# ui/app.py
import streamlit as st
import os
import sys
import time
import tempfile
import pathlib
import threading

# --- Try to import dependencies that may not be installed in some envs ---
try:
    import gdown
except Exception:
    gdown = None

try:
    from tensorflow.keras.models import load_model
    import tensorflow as tf
except Exception:
    load_model = None
    tf = None

try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    HAVE_WEBRTC = True
except Exception:
    HAVE_WEBRTC = False

import cv2
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title="ASL Real-Time (No MediaPipe)", layout="wide")
st.title("ASL Real-Time Demo — No MediaPipe (Server-side .h5 model)")

st.markdown("""
**Notes**
- This app uses your Keras `.h5` model on the server side.
- `letters` model loads from Google Drive and runs inference on webcam frames.
- `words` model is marked *under development* and is disabled.
- If your model expects **landmarks** (1D vector), you must send landmarks from the frontend — this demo only preprocesses raw frames for image-based models.
""")

# -------------------------
# CONFIG: put your drive links here
# -------------------------
# Replace these with your actual drive "view" links or direct-download link
LETTERS_DRIVE_VIEW_URL = "https://drive.google.com/file/d/1gnqGeqa-tU5WDACv38Az13LZqxYnloEZ/view?usp=sharing"
WORDS_DRIVE_VIEW_URL = ""  # leave empty or put link when ready

# Where to store downloaded models in streamlit cloud / local
MODELS_DIR = "/tmp/models"
os.makedirs(MODELS_DIR, exist_ok=True)

LETTERS_LOCAL = os.path.join(MODELS_DIR, "letters_model.h5")
WORDS_LOCAL = os.path.join(MODELS_DIR, "words_model.h5")

# -------------------------
# helper: extract file id and direct download url for Google Drive view link
# -------------------------
def gdrive_direct_from_view(url):
    # Accept either view URL or uc? id= or direct model.json etc.
    if not url:
        return None
    # If already uc? style
    if "uc?export=download" in url or "uc?id=" in url:
        return url
    # Try to extract the file id from "file/d/<id>/"
    import re
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        fid = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={fid}"
    # fallback: return original
    return url

# -------------------------
# download helper (uses gdown if available, else requests streaming)
# -------------------------
def download_file(url, dest_path, desc=None):
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    st.info(f"Downloading model to `{dest_path}` ...")
    direct = gdrive_direct_from_view(url)
    # try gdown if available (handles big files and confirm tokens)
    if gdown is not None and direct and ("drive.google.com" in direct):
        try:
            gdown.download(direct, dest_path, quiet=False, fuzzy=True)
            if os.path.exists(dest_path):
                st.success(f"Downloaded: {pathlib.Path(dest_path).name}")
                return dest_path
        except Exception as e:
            st.warning(f"gdown failed: {e}. Trying requests fallback.")
    # fallback to requests streaming
    try:
        import requests
        with requests.get(direct, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success(f"Downloaded: {pathlib.Path(dest_path).name}")
        return dest_path
    except Exception as e:
        st.error(f"Failed to download model. Error: {e}")
        return None

# -------------------------
# Load models (letters mandatory, words optional)
# -------------------------
letters_model = None
words_model = None

if load_model is None:
    st.error("TensorFlow / Keras is not installed in this environment. Please add `tensorflow` to requirements.")
else:
    # download letters
    if LETTERS_DRIVE_VIEW_URL:
        download_file(LETTERS_DRIVE_VIEW_URL, LETTERS_LOCAL)
        try:
            letters_model = load_model(LETTERS_LOCAL)
            st.success("Letters model loaded.")
        except Exception as e:
            st.error(f"Failed to load letters model: {e}")
    else:
        st.warning("No LETTERS_DRIVE_VIEW_URL provided; letters model not loaded.")

    # download words if link provided
    if WORDS_DRIVE_VIEW_URL:
        download_file(WORDS_DRIVE_VIEW_URL, WORDS_LOCAL)
        try:
            words_model = load_model(WORDS_LOCAL)
            st.success("Words model loaded.")
        except Exception as e:
            st.warning(f"Words model download/load failed (it's OK if it's under development): {e}")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")
model_option = st.sidebar.radio("Select model", ("letters", "words (under development)"))
show_fps = st.sidebar.checkbox("Show FPS", True)
speak_enabled = st.sidebar.checkbox("Enable text-to-speech (browser)", True)

use_model = None
if model_option.startswith("letters"):
    use_model = "letters"
else:
    use_model = None
    st.sidebar.warning("⚠ Word model is under development and not available yet.")

# -------------------------
# Utility: determine model input shape and preprocess
# -------------------------
def get_model_input_shape(keras_model):
    """
    Return (input_rank, shape) where shape excludes batch dim.
    e.g. for (None, 224,224,3) -> (4, (224,224,3))
         for (None, 63) -> (2, (63,))
    """
    try:
        shape = keras_model.input_shape
        # shape may be tuple or list of tuples
        if isinstance(shape, list):
            shape = shape[0]
        return (len(shape), tuple(shape[1:]))
    except Exception:
        return None

def preprocess_frame_for_model(frame_bgr, model):
    """
    Preprocess a BGR OpenCV frame for the given Keras model.
    - If model expects 4D image: resize to (H,W) and scale [0,1], return float32 array shape (1,H,W,C)
    - If model expects 2D vector: returns None (we don't have landmarks here)
    """
    meta = get_model_input_shape(model)
    if not meta:
        return None
    rank, shape = meta
    if rank == 4:
        # shape like (H,W,C) or (H,W)
        H, W, *rest = (*shape, 3)  # ensure C present
        C = rest[0] if rest else 3
        # resize
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (W, H))
        arr = resized.astype("float32") / 255.0
        # if model expects grayscale or different channels, handle minimally
        if C == 1 and arr.shape[2] == 3:
            arr = cv2.cvtColor((arr * 255).astype("uint8"), cv2.COLOR_RGB2GRAY).astype("float32") / 255.0
            arr = arr.reshape((H, W, 1))
        return np.expand_dims(arr, axis=0)
    elif rank == 2:
        # model expects 1D input (e.g., landmarks), not supported without landmarks
        return None
    else:
        return None

# -------------------------
# Text-to-speech helper (speak only on label change)
# We'll render a small HTML snippet that triggers speech when last_pred changes.
# -------------------------
def speak_label_js(label):
    if not label:
        return ""
    safe_label = str(label).replace('"', '\\"')
    js = f"""
    <script>
      var msg = new SpeechSynthesisUtterance("{safe_label}");
      window.speechSynthesis.speak(msg);
    </script>
    """
    return js

# Keep last prediction in session state
if "last_pred" not in st.session_state:
    st.session_state.last_pred = None

# -------------------------
# Video transformer for streamlit-webrtc
# -------------------------
if HAVE_WEBRTC and load_model is not None:
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.model_name = use_model
            self.last_time = None
            self.fps = 0.0

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # model selection state might change in sidebar
            if use_model is None:
                # show warning text on frame
                cv2.putText(img_bgr, "Word model under development", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # choose model
            model = letters_model if use_model == "letters" else words_model

            # preprocess
            inp = preprocess_frame_for_model(img_bgr, model)
            label = None
            if inp is None:
                # model expects 1D input (landmarks) or unsupported shape
                cv2.putText(img_bgr, "Model expects landmarks (not supported here)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                try:
                    preds = model.predict(inp)
                    # if preds shape is (1, N)
                    if preds.ndim == 2:
                        data = preds[0]
                    else:
                        data = np.array(preds).flatten()
                    idx = int(np.argmax(data))
                    if use_model == "letters":
                        label = chr(65 + idx)
                    else:
                        label = f"word_{idx}"
                    # draw label
                    cv2.putText(img_bgr, f"Pred: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(img_bgr, f"Predict error", (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)
                    print("predict error:", e, file=sys.stderr)

            # FPS
            if show_fps:
                t = time.time()
                if self.last_time is None:
                    self.last_time = t
                else:
                    dt = t - self.last_time
                    if dt > 0:
                        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                    self.last_time = t
                cv2.putText(img_bgr, f"FPS: {self.fps:.1f}", (10, img_bgr.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # speak if changed
            if label and label != st.session_state.last_pred and speak_enabled:
                st.session_state.last_pred = label
                # inject JS; components.html works but must be called from main thread, so we signal via a tiny file write hack:
                # Simpler: write hidden html via session_state placeholder later in main thread
                try:
                    st.session_state._speak = label
                except Exception:
                    pass

            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # start webrtc streamer
    st.write("Webcam stream (allow camera). If webRTC missing or fails, use the Upload fallback below.")
    webrtc_ctx = webrtc_streamer(
        key="asl-camera",
        video_transformer_factory=VideoTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True,
    )
else:
    st.warning("streamlit-webrtc not available or TensorFlow not installed — using image upload fallback.")

# -------------------------
# Image upload fallback (if webRTC not available)
# -------------------------
st.markdown("---")
st.header("Upload a snapshot (fallback)")

uploaded = st.file_uploader("Upload image (jpg/png) for prediction", type=["jpg", "jpeg", "png"])
if uploaded is not None and load_model is not None and use_model:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    model = letters_model if use_model == "letters" else words_model
    inp = preprocess_frame_for_model(frame, model)
    if inp is None:
        st.error("Model expects landmarks (1D input). This demo does not provide landmarks. You can provide an image-model or supply landmarks instead.")
    else:
        preds = model.predict(inp)
        data = preds[0] if preds.ndim == 2 else np.array(preds).flatten()
        idx = int(np.argmax(data))
        label = chr(65 + idx) if use_model == "letters" else f"word_{idx}"
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        st.success(f"Prediction: {label}")
        if speak_enabled:
            components.html(speak_label_js(label), height=0)

# -------------------------
# Speak trigger from VideoTransformer stored label
# The webrtc transformer cannot directly call components.html in its thread,
# so we use session_state._speak to trigger TTS in the main thread.
# -------------------------
if " _speak" in st.session_state:  # safety check - unlikely
    del st.session_state[" _speak"]

if "_speak" in st.session_state:
    lbl = st.session_state.pop("_speak")
    if speak_enabled:
        components.html(speak_label_js(lbl), height=0)

# small footer
st.markdown("---")
st.caption("Note: If your model expects landmarks (1D vector), this app will not produce them. For landmark-based models, you must provide the landmarks from the client-side (e.g., MediaPipe JS) and call the server API with the vector.")

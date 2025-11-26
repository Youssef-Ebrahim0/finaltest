iimport streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from PIL import Image
import gdown

# -----------------------------
# 1. Download model from Drive
# -----------------------------
url = "https://drive.google.com/uc?id=1gnqGeqa-tU5WDACv38Az13LZqxYnloEZ"
output = "my_model.h5"

@st.cache_resource
def download_and_load_model(url, path):
    # Download model if not exists
    gdown.download(url, path, quiet=False)
    # Load model
    model = tf.keras.models.load_model(path)
    return model

model = download_and_load_model(MODEL_DRIVE_LINK, MODEL_PATH)

# -----------------------------
# 2. Mediapipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("Real-Time Hand Gesture Recognition")
st.write("Take a picture or stream frames for gesture prediction:")

# Camera input (single-frame)
img_file_buffer = st.camera_input("Capture frame")

if img_file_buffer is not None:
    # Convert to OpenCV image
    frame = np.array(Image.open(img_file_buffer))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Mediapipe processing
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Prepare landmarks for model
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks).reshape(1, -1)

            # Predict gesture
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction, axis=1)[0]

            st.write(f"Predicted Gesture Class: {predicted_class}")

    st.image(frame, caption='Processed Frame', use_column_width=True)



# # ui/app.py
# import streamlit as st
# import os
# import cv2
# import numpy as np
# from collections import deque, Counter
# import time
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet50 import preprocess_input
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
# import requests

# st.set_page_config(page_title="ASL Real-Time Demo", layout="wide")
# st.title("ASL Real-Time Demo (Full Frame, No MediaPipe)")

# # -------------------------
# # Config
# # -------------------------
# LETTERS_DRIVE_URL = "https://drive.google.com/uc?export=download&id=1gnqGeqa-tU5WDACv38Az13LZqxYnloEZ"
# MODEL_PATH = "/tmp/letters_model.h5"
# CONF_THRESHOLD = 0.6
# SMOOTH_WINDOW = 8

# CLASS_LABELS = {
#      0:'A',  1:'B',  2:'C',  3:'D',  4:'E',  5:'F',  6:'G',  7:'H',  8:'I',  9:'J',
#     10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T',
#     20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z', 26:'del', 27:'nothing', 28:'space'
# }

# # -------------------------
# # Download model if needed
# # -------------------------
# if not os.path.exists(MODEL_PATH):
#     st.info("Downloading letters model from Google Drive...")
#     r = requests.get(LETTERS_DRIVE_URL, stream=True)
#     with open(MODEL_PATH, "wb") as f:
#         for chunk in r.iter_content(chunk_size=8192):
#             if chunk:
#                 f.write(chunk)
#     st.success("Model downloaded!")

# # -------------------------
# # Load model
# # -------------------------
# model = load_model(MODEL_PATH)
# IMG_H, IMG_W, IMG_C = model.input_shape[1:4]
# st.success(f"Model loaded: {IMG_H}x{IMG_W}x{IMG_C}")

# # -------------------------
# # Sidebar
# # -------------------------
# st.sidebar.header("Controls")
# model_option = st.sidebar.radio("Select model", ("letters", "words (under development)"))
# show_fps = st.sidebar.checkbox("Show FPS", True)
# tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", True)

# if model_option.startswith("letters"):
#     use_model = "letters"
# else:
#     use_model = None
#     st.sidebar.warning("⚠ Word model is under development and not available yet.")

# # -------------------------
# # Preprocessing helper
# # -------------------------
# def preprocess_frame(frame):
#     # Resize
#     img = cv2.resize(frame, (IMG_W, IMG_H))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img_arr = preprocess_input(np.expand_dims(img.astype("float32"), axis=0))
#     return img_arr

# # -------------------------
# # Text-to-Speech helper
# # -------------------------
# import streamlit.components.v1 as components
# def speak_label_js(label):
#     if not label:
#         return ""
#     safe_label = str(label).replace('"', '\\"')
#     js = f"""
#     <script>
#       var msg = new SpeechSynthesisUtterance("{safe_label}");
#       window.speechSynthesis.speak(msg);
#     </script>
#     """
#     return js

# # Session state for smoothing & TTS
# if "pred_history" not in st.session_state:
#     st.session_state.pred_history = deque(maxlen=SMOOTH_WINDOW)
# if "last_pred" not in st.session_state:
#     st.session_state.last_pred = None

# # -------------------------
# # Video Transformer
# # -------------------------
# RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# class VideoTransformer(VideoTransformerBase):
#     def __init__(self):
#         self.last_time = 0
#         self.fps = 0.0

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         label = "nothing"
#         conf = 0.0

#         if use_model:
#             inp = preprocess_frame(img)
#             preds = model.predict(inp, verbose=0)[0]
#             conf = float(np.max(preds))
#             idx = int(np.argmax(preds))
#             label = CLASS_LABELS[idx] if conf >= CONF_THRESHOLD else "nothing"

#             # smoothing
#             st.session_state.pred_history.append(label)
#             stable_label = Counter(st.session_state.pred_history).most_common(1)[0][0]
#         else:
#             stable_label = "N/A"

#         # Draw label and FPS
#         h, w, _ = img.shape
#         cv2.putText(img, f"Pred: {label} ({conf*100:.1f}%)", (10,30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
#         cv2.putText(img, f"Stable: {stable_label}", (10,70),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

#         if show_fps:
#             ctime = time.time()
#             if self.last_time != 0:
#                 self.fps = 0.9*self.fps + 0.1*(1/(ctime - self.last_time))
#             self.last_time = ctime
#             cv2.putText(img, f"FPS: {self.fps:.1f}", (10,h-20),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

#         # TTS
#         if stable_label != st.session_state.last_pred and tts_enabled and use_model:
#             st.session_state.last_pred = stable_label
#             components.html(speak_label_js(st.session_state.last_pred), height=0)

#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # -------------------------
# # Run Webcam
# # -------------------------
# webrtc_streamer(
#     key="asl-camera",
#     video_transformer_factory=VideoTransformer,
#     rtc_configuration=RTC_CONFIGURATION,
#     media_stream_constraints={"video": True, "audio": False},
#     async_transform=True,
# )

# st.markdown("---")
# st.caption("Webcam full-frame real-time ASL recognition (no MediaPipe). Word model under development.")

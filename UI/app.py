import streamlit as st
import mediapipe as mp
import os
import gdown
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from tensorflow.keras.models import load_model
from utils.preprocessing import normalize

# -----------------------------
# 1) Streamlit page config
# -----------------------------
st.set_page_config(page_title="ASL Real-Time Recognition", layout="wide")
st.title("ASL Sign Language Real-Time Recognition")
st.write("Allow camera access and select model from sidebar.")

# -----------------------------
# 2) Google Drive model links
# -----------------------------
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

model_links = {
    "letters": "https://drive.google.com/uc?id=1gnqGeqa-tU5WDACv38Az13LZqxYnloEZ",
    "words": "https://drive.google.com/uc?id=YOUR_WORDS_MODEL_ID"
}

model_paths = {}
for name, link in model_links.items():
    path = os.path.join(models_dir, f"{name}_model.h5")
    if not os.path.exists(path):
        st.write(f"Downloading {name} model from Drive...")
        gdown.download(link, path, fuzzy=True)
    model_paths[name] = path

# -----------------------------
# 3) Load models
# -----------------------------
model_manager = {
    "letters": load_model(model_paths["letters"]),
    "words": load_model(model_paths["words"])
}

# -----------------------------
# 4) Sidebar controls
# -----------------------------
# نموذج تحت التطوير
model_options = ["letters", "words (under development)"]
model_choice = st.sidebar.radio("Select Model", model_options)

# حدد الموديل الفعلي الذي سيتم استخدامه
if model_choice.startswith("letters"):
    active_model = "letters"
else:
    active_model = None
    st.warning("⚠️ Word/video model is under development and not working yet.")
    
# -----------------------------
# 5) Hand Processor
# -----------------------------


class HandProcessor:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

    def process_rgb(self, img_rgb):
        results = self.hands.process(img_rgb)
        if not results.multi_hand_landmarks:
            return None
        lm = results.multi_hand_landmarks[0].landmark
        pts = []
        for p in lm:
            pts.extend([p.x, p.y, p.z])
        return normalize(np.array(pts))


hand_processor = HandProcessor()

# -----------------------------
# 6) WebRTC configuration
# -----------------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -----------------------------
# 7) Video Transformer
# -----------------------------


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hp = hand_processor
        self.model_choice = model_choice
        self._last_time = None
        self.fps = 0

    def transform(self, frame):
        import time
        global model_choice

        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Switch model if changed in sidebar
        if self.model_choice != model_choice:
            self.model_choice = model_choice

        # Process hand
        processed = self.hp.process_rgb(img_rgb)
        label = None
        if processed is not None:
            model = model_manager[self.model_choice]
            preds = model.predict(np.array([processed]))[0]
            idx = int(np.argmax(preds))
            if self.model_choice == "letters":
                label = chr(idx + 65)
            else:
                label = f"word_{idx}"

            # Draw prediction
            cv2.putText(img, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 2)

            # Text-to-Speech
            st.components.v1.html(
                f"""
                <script>
                  var msg = new SpeechSynthesisUtterance("{label}");
                  window.speechSynthesis.speak(msg);
                </script>
                """,
                height=0,
            )

        # FPS overlay
        if show_fps:
            t = time.time()
            if self._last_time is None:
                self._last_time = t
            else:
                dt = t - self._last_time
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                self._last_time = t
            cv2.putText(img, f"FPS: {self.fps:.1f}", (10, img.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        return img


# -----------------------------
# 8) Start Streamlit Webcam
# -----------------------------
webrtc_streamer(
    key="asl-realtime",
    video_transformer_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True
)


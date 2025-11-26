import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from tensorflow.keras.models import load_model
from mediapipe_solutions import hands as mp_hands
from mediapipe_solutions import drawing_utils as mp_drawing

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("Simple Hand Gesture Recognition — Streamlit Cloud Compatible")
st.markdown(
    "This app runs on Streamlit Cloud using MediaPipe-Solutions + TensorFlow. "
    "Upload your model (.h5), labels, then start webcam.")

st.sidebar.header("Model Settings")
model_file = st.sidebar.file_uploader("Upload model (.h5)", type=["h5"])
labels_file = st.sidebar.file_uploader("Upload labels (.txt)", type=["txt"])
input_size = st.sidebar.number_input("Model input size", 32, 512, 224)
conf_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)
flip = st.sidebar.checkbox("Flip webcam", True)

@st.cache_resource
def load_model_from_file(f):
    if f is None:
        return None
    t = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    t.write(f.read())
    t.flush()
    return load_model(t.name)

model = load_model_from_file(model_file)

def load_labels(f):
    if f is None:
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return f.getvalue().decode().splitlines()

labels = load_labels(labels_file)

col1, col2 = st.columns([2, 1])
st_frame = col1.image([])
start = col1.button("Start webcam")
stop = col1.button("Stop")
pred_display = col2.empty()
fps_display = col2.empty()
box_display = col2.empty()

if "run" not in st.session_state:
    st.session_state.run = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access webcam.")
    else:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        prev = time.time()
        from collections import deque
        smooth = deque(maxlen=8)

        while st.session_state.run:
            ok, frame = cap.read()
            if not ok:
                st.warning("No webcam frame.")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            h, w, _ = frame.shape
            annotated = frame.copy()
            label = "-"
            conf = 0.0
            x1=y1=x2=y2=0

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(annotated, lm, mp_hands.HAND_CONNECTIONS)

                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                x1 = int(min(xs)*w)-20
                y1 = int(min(ys)*h)-20
                x2 = int(max(xs)*w)+20
                y2 = int(max(ys)*h)+20

                crop = frame[y1:y2, x1:x2]
                if crop.size != 0 and model is not None:
                    try:
                        img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),(input_size,input_size))
                        img = img.astype(np.float32)/255.0
                        img = np.expand_dims(img,0)
                        preds = model.predict(img)
                        idx = int(np.argmax(preds[0]))
                        conf = float(np.max(preds[0]))
                        label = labels[idx] if conf>=conf_threshold else "?"
                        smooth.append(label)
                        final = max(set(smooth), key=smooth.count)
                        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(annotated,f"{final} {conf:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

            now = time.time()
            fps = 1/(now-prev)
            prev = now

            st_frame.image(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))
            pred_display.markdown(f"**Prediction:** {label}  ")
            fps_display.markdown(f"**FPS:** {fps:.1f}")
            box_display.markdown(f"**Box:** ({x1},{y1}) → ({x2},{y2})")

        cap.release()
        hands.close()
        st.session_state.run = False
else:
    st.info("Click Start to begin.")

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
from tensorflow.keras.models import load_model
import mediapipe as mp

st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

st.title("Simple Hand Gesture Recognition — Streamlit")
st.markdown(
    "This app uses MediaPipe for hand detection and a Keras model for gesture classification.\n\n" 
    "Upload your model (.h5) and optionally a labels file (one label per line). Then press `Start` to run the webcam.")

# Sidebar: upload model + labels
st.sidebar.header("Model & settings")
model_file = st.sidebar.file_uploader("Upload Keras model (.h5)", type=["h5"], accept_multiple_files=False)
labels_file = st.sidebar.file_uploader("Upload labels (txt, one per line)", type=["txt"], accept_multiple_files=False)
input_size = st.sidebar.number_input("Model input size (square)", min_value=32, max_value=512, value=224, step=1)
conf_threshold = st.sidebar.slider("Confidence threshold (show prediction only if prob >=)", 0.0, 1.0, 0.5, 0.01)
flip_image = st.sidebar.checkbox("Flip webcam horizontally (mirror)", value=True)

# Load model (from uploaded file)
@st.cache_resource
def load_keras_model_from_file(uploaded_h5):
    if uploaded_h5 is None:
        return None
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    tfile.write(uploaded_h5.read())
    tfile.flush()
    model = load_model(tfile.name)
    return model

model = load_keras_model_from_file(model_file)

# Load labels
def load_labels(uploaded_txt):
    if uploaded_txt is None:
        # default: A-Z (26 classes)
        return [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    text = uploaded_txt.getvalue().decode("utf-8")
    labels = [line.strip() for line in text.splitlines() if line.strip()]
    return labels

labels = load_labels(labels_file)

if model is None:
    st.warning("No model uploaded. Upload a .h5 Keras model in the sidebar to enable prediction. Using demo mode (no predictions).")

col1, col2 = st.columns([2, 1])

with col1:
    stframe = st.image([])
    start_button = st.button("Start webcam")
    stop_button = st.button("Stop")

with col2:
    st.subheader("Info / Prediction")
    pred_text = st.empty()
    fps_text = st.empty()
    detected_box = st.empty()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

running = False

if start_button:
    running = True
if stop_button:
    running = False

# We'll use session state to persist running state between reruns
if "running" not in st.session_state:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# Main loop: capture frames and run detection
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam. Make sure your browser/device gives permission and no other app is using the camera.")
    else:
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)
        prev_time = time.time()
        # smoothing for prediction
        from collections import deque
        pred_buffer = deque(maxlen=8)

        while st.session_state.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Empty frame from webcam. Retrying...")
                break

            if flip_image:
                frame = cv2.flip(frame, 1)

            # convert to RGB for mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            h, w, _ = frame.shape
            # default annotation
            annotated = frame.copy()
            prediction_label = "-"
            prediction_conf = 0.0

            if results.multi_hand_landmarks:
                # draw landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # compute bounding box around landmarks
                lm = results.multi_hand_landmarks[0]
                xs = [pt.x for pt in lm.landmark]
                ys = [pt.y for pt in lm.landmark]
                x_min = int(max(0, min(xs) * w) - 20)
                x_max = int(min(w, max(xs) * w) + 20)
                y_min = int(max(0, min(ys) * h) - 20)
                y_max = int(min(h, max(ys) * h) + 20)

                # crop and prepare for model
                crop = frame[y_min:y_max, x_min:x_max]
                if crop.size != 0 and model is not None:
                    try:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(crop_rgb, (input_size, input_size))
                        img = img.astype(np.float32) / 255.0
                        img = np.expand_dims(img, axis=0)
                        preds = model.predict(img)
                        if preds.ndim == 1 or preds.shape[1] == 1:
                            # regression or single output -- not supported here
                            pred_label = "?"
                            pred_conf = 0.0
                        else:
                            pred_idx = int(np.argmax(preds[0]))
                            pred_conf = float(np.max(preds[0]))
                            if pred_conf >= conf_threshold and pred_idx < len(labels):
                                prediction_label = labels[pred_idx]
                                prediction_conf = pred_conf
                            else:
                                prediction_label = "(low conf)"
                                prediction_conf = pred_conf

                        pred_buffer.append(prediction_label)
                        # voting
                        if len(pred_buffer) > 0:
                            most_common = max(set(pred_buffer), key=pred_buffer.count)
                        else:
                            most_common = prediction_label

                        # draw bbox and label
                        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(annotated, f"{most_common} {prediction_conf:.2f}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    except Exception as e:
                        cv2.putText(annotated, f"Model error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        st.error(f"Prediction error: {e}")

            # show fps
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time

            stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
            pred_text.markdown(f"**Prediction:** {prediction_label}  
                        **Confidence:** {prediction_conf:.2f}")
            fps_text.markdown(f"**FPS:** {fps:.1f}")
            detected_box.markdown(f"**Hand box:** x={x_min},{x_max} y={y_min},{y_max}")

            # small sleep to reduce CPU usage
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        hands.close()
        st.session_state.running = False
else:
    st.info("Press 'Start webcam' to begin. Upload a model in the sidebar for live predictions.")

st.markdown("---")
st.markdown("**Notes / Troubleshooting**:\n\n- If webcam doesn't start, check camera permissions in your browser.\n- If your model fails to load, make sure it's a Keras .h5 file saved with `model.save('model.h5')`.\n- Adjust the input size or labels file if your model expects different dimensions or class ordering.\n- This app is a minimal demo — for production, consider using a proper backend, handling multiple clients, and optimizing model inference (e.g., TensorFlow Lite or ONNX).")

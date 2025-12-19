from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import os
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Path to Haar cascade XML file
CASCADE_PATH = r'D:\Projects\EmotionDetection\LearnData\Output\haarcascade_frontalface_default.xml'

# Path to trained model (MobileNetV2)
MODEL_PATH = r'D:\Projects\EmotionDetection\LearnData\model\Best_Emotion_MobileNetV2.keras'

# Video source: 0 is default webcam, 1 is external webcam (if present)
VIDEO_SOURCE = 0 

CLASS_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_roi(roi, target_h, target_w, target_c):
    # Standard preprocessing for MobileNetV2
    if roi is None or roi.size == 0:
        return None

    # 1. Resize
    roi_resized = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 2. Convert color (OpenCV BGR -> RGB)
    if target_c == 3: 
        if roi_resized.ndim == 2:
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        else:
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Normalize to [-1, 1] (MobileNetV2 standard)
    arr = roi_resized.astype('float32')
    arr = preprocess_input(arr) 

    return np.expand_dims(arr, axis=0)

def main():
    # Check files
    if not os.path.exists(CASCADE_PATH):
        print(f"ERROR: Missing file: {CASCADE_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Missing file: {MODEL_PATH}")
        return

    print('--- Loading model & camera... ---')
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    model = load_model(MODEL_PATH, compile=False)

    # Get model input size
    try:
        in_shape = model.input_shape
        H, W, C = in_shape[1], in_shape[2], in_shape[3]
    except:
        H, W, C = 224, 224, 3 # Default

    # Open Webcam
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Could not open Webcam. Check connection or permissions.")
        return

    # FPS variables
    prev_frame_time = 0
    new_frame_time = 0

    print("--- STARTING DETECTION (Press 'q' to exit) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Flip image (mirror) for mirror-like view
        frame = cv2.flip(frame, 1)

        # 2. Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w_box, h_box) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
            
            # Crop face (ROI) from color frame
            roi = frame[y:y + h_box, x:x + w_box]

            # Predict
            try:
                inp = preprocess_roi(roi, H, W, C)
                if inp is not None:
                    preds = model.predict(inp, verbose=0)[0]
                    idx = np.argmax(preds)
                    label = CLASS_LABELS[idx]
                    prob = preds[idx]

                    # Display results
                    text_label = f'{label}'
                    text_prob = f'{prob*100:.0f}%'
                    
                    # Change color based on emotion
                    if label == 'Angry': color = (0, 0, 255)       # Red
                    elif label == 'Happy': color = (0, 255, 0)     # Green
                    elif label == 'Sad': color = (255, 0, 0)       # Blue
                    else: color = (0, 255, 255)                    # Yellow

                    # Draw background for text readability
                    cv2.rectangle(frame, (x, y - 40), (x + w_box, y), color, -1)
                    cv2.putText(frame, text_label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    cv2.putText(frame, text_prob, (x + w_box - 55, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            except Exception as e:
                print(e)

        # 3. Compute and display FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Show
        cv2.imshow('Emotion Detection - Webcam', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
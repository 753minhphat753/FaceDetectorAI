from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np
import os
import time

# ==========================================
# CẤU HÌNH
# ==========================================
# Đường dẫn file xml cascade
CASCADE_PATH = r'D:\Projects\EmotionDetection\LearnData\Output\haarcascade_frontalface_default.xml'

# Đường dẫn model đã train (MobileNetV2)
MODEL_PATH = r'D:\Projects\EmotionDetection\LearnData\model\Best_Emotion_MobileNetV2.keras'

# Nguồn Video: Số 0 là Webcam mặc định, 1 là webcam cắm ngoài (nếu có)
VIDEO_SOURCE = 0 

CLASS_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def preprocess_roi(roi, target_h, target_w, target_c):
    # Hàm tiền xử lý chuẩn cho MobileNetV2
    if roi is None or roi.size == 0:
        return None

    # 1. Resize
    roi_resized = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 2. Chuyển đổi màu (OpenCV BGR -> RGB)
    if target_c == 3: 
        if roi_resized.ndim == 2:
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        else:
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Chuẩn hóa về [-1, 1] (Chuẩn MobileNetV2)
    arr = roi_resized.astype('float32')
    arr = preprocess_input(arr) 

    return np.expand_dims(arr, axis=0)

def main():
    # Kiểm tra file
    if not os.path.exists(CASCADE_PATH):
        print(f"LỖI: Không tìm thấy {CASCADE_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"LỖI: Không tìm thấy {MODEL_PATH}")
        return

    print('--- Đang load model & camera... ---')
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    model = load_model(MODEL_PATH, compile=False)

    # Lấy kích thước input của model
    try:
        in_shape = model.input_shape
        H, W, C = in_shape[1], in_shape[2], in_shape[3]
    except:
        H, W, C = 224, 224, 3 # Mặc định

    # Mở Webcam
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("LỖI: Không thể mở Webcam. Hãy kiểm tra lại kết nối hoặc quyền truy cập.")
        return

    # Biến tính FPS
    prev_frame_time = 0
    new_frame_time = 0

    print("--- BẮT ĐẦU NHẬN DIỆN (Nhấn 'q' để thoát) ---")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Lật hình (Mirror) cho giống soi gương
        frame = cv2.flip(frame, 1)

        # 2. Phát hiện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w_box, h_box) in faces:
            # Vẽ khung mặt
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
            
            # Cắt mặt (ROI) từ frame màu
            roi = frame[y:y + h_box, x:x + w_box]

            # Dự đoán
            try:
                inp = preprocess_roi(roi, H, W, C)
                if inp is not None:
                    preds = model.predict(inp, verbose=0)[0]
                    idx = np.argmax(preds)
                    label = CLASS_LABELS[idx]
                    prob = preds[idx]

                    # Hiển thị kết quả
                    text_label = f'{label}'
                    text_prob = f'{prob*100:.0f}%'
                    
                    # Đổi màu dựa trên cảm xúc
                    if label == 'Angry': color = (0, 0, 255)       # Đỏ
                    elif label == 'Happy': color = (0, 255, 0)     # Xanh lá
                    elif label == 'Sad': color = (255, 0, 0)       # Xanh dương
                    else: color = (0, 255, 255)                    # Vàng

                    # Vẽ background cho chữ dễ đọc
                    cv2.rectangle(frame, (x, y - 40), (x + w_box, y), color, -1)
                    cv2.putText(frame, text_label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
                    cv2.putText(frame, text_prob, (x + w_box - 55, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            except Exception as e:
                print(e)

        # 3. Tính và hiện FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4. Hiển thị
        cv2.imshow('Emotion Detection - Webcam', frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
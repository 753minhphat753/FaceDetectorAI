from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # <--- THÊM DÒNG NÀY
import cv2
import numpy as np
import os

# ==========================================
# CẤU HÌNH
# ==========================================
CASCADE_PATH = r'D:\Projects\EmotionDetection\LearnData\Output\haarcascade_frontalface_default.xml'
MODEL_PATH = r'D:\Projects\EmotionDetection\LearnData\model\Best_Emotion_MobileNetV2.keras'
VIDEO_SOURCE = r'D:\Projects\EmotionDetection\LearnData\Output\video3.mp4'  # Hoặc để 0 nếu dùng Webcam
OUTPUT_PATH = r'D:\Projects\EmotionDetection\LearnData\Output\video_output_3.mp4'

# LƯU Ý: Thứ tự này phải khớp với thứ tự thư mục lúc train (Alpha B)
CLASS_LABELS = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

def check_files():
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f'Haarcascade not found: {CASCADE_PATH}')
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

def preprocess_roi(roi, target_h, target_w, target_c):
    if roi is None or roi.size == 0:
        return None

    # 1. Resize về kích thước model yêu cầu (thường là 224x224)
    roi_resized = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 2. Đảm bảo đúng kênh màu (RGB)
    # MobileNetV2 luôn cần 3 kênh màu
    if target_c == 3: 
        if roi_resized.ndim == 2: # Nếu là ảnh xám -> chuyển sang màu
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2RGB)
        else: # OpenCV đọc là BGR -> chuyển sang RGB
            roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    
    # 3. CHUẨN HÓA DỮ LIỆU (QUAN TRỌNG NHẤT)
    # Chuyển sang float
    arr = roi_resized.astype('float32')
    
    # Dùng hàm chuẩn của MobileNetV2 (đưa về khoảng -1 đến 1)
    # Thay thế cho dòng arr / 255.0 cũ
    arr = preprocess_input(arr) 

    # Thêm chiều batch (1, 224, 224, 3)
    return np.expand_dims(arr, axis=0)

def main():
    check_files()
    print('Loading face cascade...')
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    print('Loading model...')
    model = load_model(MODEL_PATH, compile=False)

    # Tự động lấy kích thước input của model
    try:
        in_shape = model.input_shape
        # Input shape thường là (None, 224, 224, 3)
        if in_shape[1] is not None:
            H, W, C = in_shape[1], in_shape[2], in_shape[3]
        else:
            # Fallback nếu model shape là dynamic, MobileNetV2 chuẩn là 224
            H, W, C = 224, 224, 3 
    except Exception:
        H, W, C = 224, 224, 3

    print(f'Model input expected: {H}x{W}x{C}')

    # Mở video
    if str(VIDEO_SOURCE).isdigit():
        cap = cv2.VideoCapture(int(VIDEO_SOURCE))
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video source: {VIDEO_SOURCE}')

    writer = None
    if OUTPUT_PATH:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w, h))
        print(f'Writing output to {OUTPUT_PATH} ({w}x{h} @ {fps}fps)')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('End of stream')
            break

        # Phát hiện khuôn mặt (dùng ảnh xám để detect cho nhanh)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            pass # Không làm gì nếu không thấy mặt
        else:
            for (x, y, w_box, h_box) in faces:
                # Vẽ khung
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 255), 2)
                
                # Cắt vùng mặt (Lấy từ frame màu gốc)
                roi = frame[y:y + h_box, x:x + w_box]

                # Tiền xử lý
                inp = preprocess_roi(roi, H, W, C)
                
                if inp is not None:
                    # Dự đoán
                    preds = model.predict(inp, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    prob = preds[idx]
                    
                    label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)
                    
                    # Hiển thị text
                    text = f'{label}: {prob*100:.1f}%'
                    color = (0, 0, 255) if label == 'Angry' else (0, 255, 0) # Đỏ nếu giận, Xanh nếu khác
                    
                    # Vẽ nền đen cho chữ dễ đọc
                    cv2.rectangle(frame, (x, y - 30), (x + 180, y), (0, 0, 0), -1)
                    cv2.putText(frame, text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Emotion Detector (MobileNetV2)', frame)
        
        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "best.pt"  # cùng thư mục

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Không thấy {MODEL_PATH}. Hãy để best.pt cùng thư mục!")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows ổn định hơn
if not cap.isOpened():
    raise RuntimeError("Không mở được webcam. Thử đổi 0 -> 1 hoặc kiểm tra quyền camera.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    r = model.predict(source=frame, imgsz=224, verbose=False)[0]
    top1 = int(r.probs.top1)
    conf = float(r.probs.top1conf)
    name = r.names[top1]

    cv2.putText(frame, f"{name}  conf:{conf:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Digits Classifier (ESC to quit)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
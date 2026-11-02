import cv2
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
helmet_model = YOLO("static/saved_models/helmet_detector.pt")

sinch_path = 'static/uploads/videos/test_CAND-2026-062_sinch.mp4'
cap = cv2.VideoCapture(sinch_path)

confs = []
for _ in range(60):
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    for b in b_res.boxes:
        if int(b.cls[0]) == 0:
            rx1, ry1, rx2, ry2 = [int(x) for x in b.xyxy[0].tolist()]
            rh = ry2 - ry1
            rw = rx2 - rx1
            y_top = max(0, ry1 - int(rh * 0.05))
            y_bot = min(h, ry1 + int(rh * 0.55))
            x_left = max(0, rx1 - int(rw * 0.10))
            x_right = min(w, rx2 + int(rw * 0.10))
            upper_crop = frame[y_top:y_bot, x_left:x_right]
            if upper_crop.size > 0:
                h_res = helmet_model(upper_crop, verbose=False, conf=0.30)[0]
                for hb in h_res.boxes:
                    confs.append((helmet_model.names[int(hb.cls[0])], float(hb.conf[0])))

cap.release()
print("Sinch first 60 frames:")
for item in confs[:25]:
    print(item)

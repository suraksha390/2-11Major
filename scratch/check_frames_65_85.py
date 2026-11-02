import cv2
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("without_helmet_foottouch.mp4")

f = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    f += 1
    if 65 <= f <= 85:
        b_res = base_model(frame, verbose=False, conf=0.25)[0]
        motos = []
        riders = []
        for b in b_res.boxes:
            c = int(b.cls[0])
            conf = float(b.conf[0])
            xyxy = [int(x) for x in b.xyxy[0].tolist()]
            if c == 3:
                motos.append((conf, xyxy))
            elif c == 0:
                riders.append((conf, xyxy))
        print(f"F{f:03d}: riders={riders}, motos={motos}")
cap.release()

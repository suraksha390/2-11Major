import cv2
import os
from ultralytics import YOLO

model = YOLO('static/saved_models/helmet_detector.pt')
base_model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("static/uploads/videos/suraksha_with_helmet.mp4")

print("Testing helmet detector on suraksha_with_helmet.mp4...")
frame_idx = 0
stride = 5

detections = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    res_base = base_model(frame, verbose=False, conf=0.35)[0]
    best_rider = None
    best_conf = 0.0
    for b in res_base.boxes:
        if int(b.cls[0]) == 0 and float(b.conf[0]) > best_conf:
            best_conf = float(b.conf[0])
            best_rider = [int(x) for x in b.xyxy[0].tolist()]

    if best_rider:
        rx1, ry1, rx2, ry2 = best_rider
        rh = ry2 - ry1
        rw = rx2 - rx1
        
        y_top = max(0, ry1 - int(rh * 0.05))
        y_bot = min(frame.shape[0], ry1 + int(rh * 0.50))
        x_left = max(0, rx1 - int(rw * 0.10))
        x_right = min(frame.shape[1], rx2 + int(rw * 0.10))
        upper_crop = frame[y_top:y_bot, x_left:x_right]
        
        if upper_crop.size > 0:
            res_upper = model(upper_crop, verbose=False, conf=0.35)[0]
            upper_boxes = []
            for b in res_upper.boxes:
                upper_boxes.append((model.names[int(b.cls[0])], float(b.conf[0])))
            if upper_boxes:
                detections.append((frame_idx, upper_boxes))

cap.release()
print(f"Total detections on with_helmet video: {len(detections)}")
with_helm = sum(1 for _, bxs in detections if any(b[0] == 'With Helmet' and b[1] > 0.4 for b in bxs))
without_helm = sum(1 for _, bxs in detections if any(b[0] == 'Without Helmet' and b[1] > 0.4 for b in bxs))
print(f"With Helmet count: {with_helm}, Without Helmet count: {without_helm}")
for f_idx, bxs in detections[:10]:
    print(f"Frame {f_idx}: {bxs}")

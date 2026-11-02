import cv2
import os
import sys
from ultralytics import YOLO

model = YOLO('static/saved_models/helmet_detector.pt')
cap = cv2.VideoCapture("without_helmet_foottouch.mp4")

print("Testing helmet detector on entire video...")
frame_idx = 0
stride = 3

full_frame_detections = []
rider_crop_detections = []

from ultralytics import YOLO as BaseYOLO
base_model = BaseYOLO("yolov8n.pt")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    # 1. Direct prediction on full frame with helmet model
    res_full = model(frame, verbose=False, conf=0.35)[0]
    full_boxes = []
    for b in res_full.boxes:
        full_boxes.append((model.names[int(b.cls[0])], float(b.conf[0]), [round(x, 1) for x in b.xyxy[0].tolist()]))
    if full_boxes:
        full_frame_detections.append((frame_idx, full_boxes))

    # 2. Prediction on rider bbox
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
        
        # Test on upper body (top 50% of rider bbox with a bit of padding)
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
                rider_crop_detections.append((frame_idx, upper_boxes))

cap.release()

print("\n--- Full Frame Detections Count:", len(full_frame_detections), "---")
for f_idx, bxs in full_frame_detections[:20]:
    print(f"Frame {f_idx}: {bxs}")

print("\n--- Upper Rider Crop Detections Count:", len(rider_crop_detections), "---")
for f_idx, bxs in rider_crop_detections[:20]:
    print(f"Frame {f_idx}: {bxs}")

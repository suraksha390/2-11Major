import cv2
import os
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
helmet_model = YOLO("static/saved_models/helmet_detector.pt")

cap_video = "static/uploads/videos/test_CAND-2026-060_cap_video.mp4"
if not os.path.exists(cap_video):
    print("Cap video not found!")
    exit(0)

cap = cv2.VideoCapture(cap_video)
print(f"Testing on {cap_video}...")

frame_idx = 0
stride = 5
results = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    h, w = frame.shape[:2]
    b_res = base_model(frame, verbose=False, conf=0.35)[0]
    best_rider = None
    best_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 0 and float(b.conf[0]) > best_conf:
            best_conf = float(b.conf[0])
            best_rider = [int(x) for x in b.xyxy[0].tolist()]

    if best_rider:
        rx1, ry1, rx2, ry2 = best_rider
        rh = ry2 - ry1
        rw = rx2 - rx1
        
        y_top = max(0, ry1 - int(rh * 0.05))
        y_bot = min(h, ry1 + int(rh * 0.55))
        x_left = max(0, rx1 - int(rw * 0.10))
        x_right = min(w, rx2 + int(rw * 0.10))
        upper_crop = frame[y_top:y_bot, x_left:x_right]
        
        if upper_crop.size > 0:
            res_upper = helmet_model(upper_crop, verbose=False, conf=0.30)[0]
            boxes = []
            for b in res_upper.boxes:
                boxes.append((helmet_model.names[int(b.cls[0])], float(b.conf[0])))
            results.append((frame_idx, boxes))

cap.release()
print(f"Total sampled frames with rider: {len(results)}")
for f_idx, bxs in results[:15]:
    print(f"Frame {f_idx}: {bxs}")

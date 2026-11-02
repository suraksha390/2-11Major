import cv2
import os
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
helmet_model = YOLO("static/saved_models/helmet_detector.pt")

sinch_path = 'static/uploads/videos/test_CAND-2026-062_sinch.mp4'
cap = cv2.VideoCapture(sinch_path)

print(f"Testing sinch.mp4 with upper body crop...")
f = 0
stride = 2
with_helm = 0
without_helm = 0
total_rider = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    f += 1
    if f % stride != 0:
        continue
    h, w = frame.shape[:2]
    
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    best_rider = None
    best_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 0 and float(b.conf[0]) > best_conf:
            best_conf = float(b.conf[0])
            best_rider = [int(x) for x in b.xyxy[0].tolist()]
            
    if best_rider:
        total_rider += 1
        rx1, ry1, rx2, ry2 = best_rider
        rw = rx2 - rx1
        rh = ry2 - ry1
        
        # Upper body crop (top 55% of rider)
        y1 = max(0, ry1 - int(rh * 0.05))
        y2 = min(h, ry1 + int(rh * 0.55))
        x1 = max(0, rx1 - int(rw * 0.10))
        x2 = min(w, rx2 + int(rw * 0.10))
        upper_crop = frame[y1:y2, x1:x2]
        
        if upper_crop.size > 0:
            h_res = helmet_model(upper_crop, verbose=False, conf=0.3)[0]
            boxes = h_res.boxes
            has_with = False
            has_without = False
            for b in boxes:
                c = int(b.cls[0])
                conf = float(b.conf[0])
                c_name = helmet_model.names[c]
                if c_name == 'With Helmet' and conf >= 0.40:
                    has_with = True
                elif c_name == 'Without Helmet' and conf >= 0.40:
                    has_without = True
            if has_with:
                with_helm += 1
            elif has_without:
                without_helm += 1

cap.release()
print(f"Sinch.mp4 results: total_rider={total_rider}, with_helm={with_helm}, without_helm={without_helm}")
if total_rider > 0:
    print(f"Ratio With Helmet: {with_helm / total_rider:.2f}")

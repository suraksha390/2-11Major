import cv2
import os
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
helmet_model = YOLO("static/saved_models/helmet_detector.pt")

frame = cv2.imread("scratch/video_frames/frame_060.jpg")
h, w = frame.shape[:2]

# Rider detection
b_res = base_model(frame, verbose=False)[0]
for b in b_res.boxes:
    if int(b.cls[0]) == 0:
        rx1, ry1, rx2, ry2 = [int(x) for x in b.xyxy[0].tolist()]
        rw = rx2 - rx1
        rh = ry2 - ry1
        
        # Test 1: Upper body crop
        upper = frame[max(0, ry1 - 5): min(h, ry1 + int(rh * 0.55)), max(0, rx1 - 10): min(w, rx2 + 10)]
        h_res = helmet_model(upper, verbose=False)[0]
        print("Upper body crop detections:")
        for hb in h_res.boxes:
            print(f"  {helmet_model.names[int(hb.cls[0])]}: conf={float(hb.conf[0]):.3f}, box={hb.xyxy[0].tolist()}")

        # Test 2: Full frame
        full_res = helmet_model(frame, verbose=False)[0]
        print("Full frame detections:")
        for hb in full_res.boxes:
            print(f"  {helmet_model.names[int(hb.cls[0])]}: conf={float(hb.conf[0]):.3f}, box={hb.xyxy[0].tolist()}")

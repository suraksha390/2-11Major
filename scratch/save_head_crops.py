import cv2
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
out_dir = "scratch/head_crops_test"
os.makedirs(out_dir, exist_ok=True)

frame_idx = 0
stride = 5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    dets, _ = yolo_detector.detect_objects(frame)
    if dets['rider_found'] and dets['helmet_bbox']:
        hx1, hy1, hx2, hy2 = dets['helmet_bbox']
        crop = frame[hy1:hy2, hx1:hx2]
        if crop.size > 0:
            h_res = yolo_detector.classify_head_roi(crop)
            cls_name = h_res['class']
            stat = h_res['status']
            conf = h_res['confidence']
            fname = f"{out_dir}/f{frame_idx:03d}_{cls_name}_{stat}_{conf:.2f}.jpg"
            cv2.imwrite(fname, crop)

cap.release()
print("Saved head crops to", out_dir)

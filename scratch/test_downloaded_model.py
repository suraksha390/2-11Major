import os
import sys
from ultralytics import YOLO

model_path = 'static/saved_models/helmet_detector.pt'
model = YOLO(model_path)
print("Model loaded successfully!")
print("Model task:", model.task)
print("Model names (classes):", model.names)

# Test on a few head crops from without_helmet_foottouch.mp4
test_crops = [
    "scratch/head_crops_test/f060_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f090_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f120_helmet_PASS_0.90.jpg",
    "scratch/head_crops_test/f170_no_helmet_FAIL_0.31.jpg",
    "scratch/head_crops_test/f195_helmet_PASS_0.90.jpg"
]

print("\n--- Testing on head crops ---")
for cp in test_crops:
    if not os.path.exists(cp):
        continue
    res = model(cp, verbose=False)[0]
    boxes = res.boxes
    print(f"\nImage: {cp}")
    if len(boxes) == 0:
        print("  No detection on crop (might be bare head or head not recognized as helmet)")
    for b in boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = model.names[cls_id]
        print(f"  Box: {name} (cls={cls_id}), conf={conf:.3f}, xyxy={b.xyxy[0].tolist()}")

# Also test on full frames!
import cv2
test_frames = [
    "scratch/video_frames/frame_060.jpg",
    "scratch/video_frames/frame_120.jpg",
    "scratch/video_frames/frame_180.jpg"
]
print("\n--- Testing on full video frames ---")
for fp in test_frames:
    if not os.path.exists(fp):
        continue
    res = model(fp, verbose=False)[0]
    boxes = res.boxes
    print(f"\nFrame: {fp}")
    for b in boxes:
        cls_id = int(b.cls[0])
        conf = float(b.conf[0])
        name = model.names[cls_id]
        print(f"  Detected: {name} (cls={cls_id}), conf={conf:.3f}, xyxy={b.xyxy[0].tolist()}")

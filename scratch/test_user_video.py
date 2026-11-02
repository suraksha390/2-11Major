import cv2
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath('.'))

from models.yolo_detector import yolo_detector
from models.pose_analyzer import pose_analyzer
from models.lane_detector import lane_detector
from models.ml_evaluator import ml_evaluator

video_path = "without_helmet_foottouch.mp4"
if not os.path.exists(video_path):
    print(f"Error: {video_path} does not exist!")
    sys.exit(1)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video Info: {video_path}")
print(f"Resolution: {w}x{h}, FPS: {fps}, Total Frames: {total_frames}, Duration: {total_frames/fps:.2f}s")

frame_idx = 0
stride = 3

rider_frames = 0
helmet_classes = {}
foot_touches = 0
ground_distances = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    yolo_dets, _ = yolo_detector.detect_objects(frame)
    pose_info, _ = pose_analyzer.analyze_pose(frame, yolo_dets['ground_line_y'])

    if yolo_dets['rider_found']:
        rider_frames += 1
        cls = yolo_dets.get('helmet_class', 'none')
        stat = yolo_dets.get('helmet_status', 'FAIL')
        conf = yolo_dets.get('helmet_conf', 0.0)
        det_str = f"{cls} ({stat}, conf={conf:.2f})"
        helmet_classes[det_str] = helmet_classes.get(det_str, 0) + 1
        
        # Log every 15 frames or when foot touch occurs
        dist = pose_info.get('min_foot_ground_distance', 999)
        ground_distances.append(dist)
        if pose_info['foot_touch']:
            foot_touches += 1
            print(f"[Frame {frame_idx}] Rider detected: Helmet={det_str} | Foot touch=TRUE (dist={dist:.1f}, ground_y={yolo_dets['ground_line_y']})")
        elif frame_idx % 30 == 0:
            print(f"[Frame {frame_idx}] Rider detected: Helmet={det_str} | Foot touch=False (dist={dist:.1f}, ground_y={yolo_dets['ground_line_y']})")

cap.release()

print("\n--- Summary ---")
print(f"Total sampled frames: {frame_idx // stride}")
print(f"Rider detected frames: {rider_frames}")
print("Helmet class breakdown:", helmet_classes)
print(f"Foot touches detected: {foot_touches}")
if ground_distances:
    print(f"Min foot-ground dist: {min(ground_distances):.1f}, Max: {max(ground_distances):.1f}, Mean: {np.mean(ground_distances):.1f}")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from models.yolo_detector import YoloDetector
from models.pose_analyzer import PoseAnalyzer

def test_video():
    video_path = "without_helmet_foottouch.mp4"
    if not os.path.exists(video_path):
        print("Video not found:", video_path)
        return

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Testing video {video_path}: {total_frames} frames")

    yolo = YoloDetector()
    pose = PoseAnalyzer()

    frame_idx = 0
    stride = 3
    touch_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        dets, _ = yolo.detect_objects(frame)
        res, _ = pose.analyze_pose(
            frame, 
            ground_y=dets.get('ground_line_y'),
            moto_bbox=dets.get('motorcycle_bbox'),
            rider_bbox=dets.get('rider_bbox'),
            frame_idx=frame_idx
        )

        if res.get('foot_touch'):
            touch_frames.append((frame_idx, res.get('min_foot_ground_distance'), res.get('active_foot')))

    cap.release()
    print("Touch frames count:", len(touch_frames))
    print("Confirmed events:", getattr(pose, 'confirmed_events', 'N/A'))
    if touch_frames:
        print("First 5 touch frames:", touch_frames[:5])
        print("Last 5 touch frames:", touch_frames[-5:])

if __name__ == '__main__':
    test_video()

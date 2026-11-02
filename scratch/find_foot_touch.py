import cv2
import mediapipe as mp
import numpy as np
import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

out_dir = "scratch/foot_analysis"
os.makedirs(out_dir, exist_ok=True)

idx = 0
results_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    idx += 1
    
    yolo_dets, _ = yolo_detector.detect_objects(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_res = pose.process(rgb)
    
    l_ankle = r_ankle = l_foot = r_foot = None
    l_heel = r_heel = None
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        l_ankle = (int(lm[27].x * w), int(lm[27].y * h), lm[27].visibility)
        r_ankle = (int(lm[28].x * w), int(lm[28].y * h), lm[28].visibility)
        l_heel = (int(lm[29].x * w), int(lm[29].y * h), lm[29].visibility)
        r_heel = (int(lm[30].x * w), int(lm[30].y * h), lm[30].visibility)
        l_foot = (int(lm[31].x * w), int(lm[31].y * h), lm[31].visibility)
        r_foot = (int(lm[32].x * w), int(lm[32].y * h), lm[32].visibility)
        
        # Lowest foot point
        feet_y = [p[1] for p in [l_ankle, r_ankle, l_heel, r_heel, l_foot, r_foot] if p[2] > 0.4]
        lowest_y = max(feet_y) if feet_y else None
    else:
        lowest_y = None
        
    moto_bbox = yolo_dets.get('motorcycle_bbox')
    moto_bottom = moto_bbox[3] if moto_bbox else None
    rider_bbox = yolo_dets.get('rider_bbox')
    
    # Save frame if lowest_y is close to moto_bottom or in key frames
    diff = (moto_bottom - lowest_y) if (moto_bottom and lowest_y) else 999
    
    results_log.append({
        'frame': idx,
        'lowest_y': lowest_y,
        'moto_bottom': moto_bottom,
        'diff': diff,
        'l_foot': l_foot,
        'r_foot': r_foot,
        'l_heel': l_heel,
        'r_heel': r_heel,
        'moto_bbox': moto_bbox,
        'rider_bbox': rider_bbox
    })
    
    if diff < 60 and idx % 2 == 0:
        # draw and save
        dbg = frame.copy()
        if moto_bbox:
            cv2.rectangle(dbg, (moto_bbox[0], moto_bbox[1]), (moto_bbox[2], moto_bbox[3]), (255, 0, 255), 2)
            cv2.line(dbg, (0, moto_bottom), (w, moto_bottom), (0, 255, 255), 1)
        if lowest_y:
            cv2.line(dbg, (0, lowest_y), (w, lowest_y), (0, 0, 255), 1)
            for pt, col in [(l_foot, (0, 255, 0)), (r_foot, (255, 255, 0)), (l_heel, (0, 200, 200)), (r_heel, (200, 200, 0))]:
                if pt and pt[2] > 0.4:
                    cv2.circle(dbg, (pt[0], pt[1]), 5, col, -1)
        cv2.putText(dbg, f"F:{idx} diff:{diff} lowest:{lowest_y} moto_bot:{moto_bottom}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imwrite(f"{out_dir}/frame_{idx:03d}.jpg", dbg)

cap.release()
pose.close()

# Find frames with lowest diff
results_log.sort(key=lambda x: x['diff'])
print("Top 15 frames with lowest foot-to-moto_bottom difference:")
for r in results_log[:15]:
    print(f"Frame {r['frame']}: lowest_y={r['lowest_y']}, moto_bottom={r['moto_bottom']}, diff={r['diff']}, L_foot={r['l_foot']}, R_foot={r['r_foot']}")

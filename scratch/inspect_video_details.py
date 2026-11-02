import cv2, os
import mediapipe as mp
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('.'))
from models.yolo_detector import yolo_detector

video_path = r'static/uploads/videos/test_CAND-2026-083_sinchana_bbreak.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3)

frame_idx = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    # Check every frame or every 2 frames
    if frame_idx % 2 != 0:
        continue

    h, w, _ = frame.shape
    yolo_dets, _ = yolo_detector.detect_objects(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)

    moto = yolo_dets.get('motorcycle_bbox')
    rider = yolo_dets.get('rider_bbox')

    l_ankle = r_ankle = l_heel = r_heel = l_toe = r_toe = None
    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        l_ankle = (lm[27].x*w, lm[27].y*h, lm[27].visibility)
        r_ankle = (lm[28].x*w, lm[28].y*h, lm[28].visibility)
        l_heel = (lm[29].x*w, lm[29].y*h, lm[29].visibility)
        r_heel = (lm[30].x*w, lm[30].y*h, lm[30].visibility)
        l_toe = (lm[31].x*w, lm[31].y*h, lm[31].visibility)
        r_toe = (lm[32].x*w, lm[32].y*h, lm[32].visibility)

    data.append({
        'f': frame_idx,
        'moto': moto,
        'rider': rider,
        'l_ankle': l_ankle,
        'r_ankle': r_ankle,
        'l_heel': l_heel,
        'r_heel': r_heel,
        'l_toe': l_toe,
        'r_toe': r_toe
    })

cap.release()
pose.close()

print(f"Processed {len(data)} sampled frames out of {total_frames}.")

print("\n--- Motion trajectory (sampled every 20 frames) ---")
for d in data[::10]:
    f = d['f']
    moto = d['moto']
    rider = d['rider']
    moto_str = f"({moto[0]},{moto[1]},{moto[2]},{moto[3]})" if moto else "None"
    rider_str = f"({rider[0]},{rider[1]},{rider[2]},{rider[3]})" if rider else "None"
    la = f"{d['l_ankle'][1]:.1f}" if d['l_ankle'] else "N"
    ra = f"{d['r_ankle'][1]:.1f}" if d['r_ankle'] else "N"
    print(f"F {f:03d} | Moto: {moto_str} | Rider: {rider_str} | L_Ankle_Y: {la} | R_Ankle_Y: {ra}")

# Let's save data as npy or pkl for quick querying
import pickle
with open('scratch/sinchana_bbreak_analysis.pkl', 'wb') as pf:
    pickle.dump(data, pf)
print("Saved analysis data to scratch/sinchana_bbreak_analysis.pkl")

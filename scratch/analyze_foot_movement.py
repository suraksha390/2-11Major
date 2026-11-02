import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

base_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)

frame_idx = 0
data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 2 != 0:
        continue

    # Motorcycle
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    moto_box = None
    best_moto_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 3 and float(b.conf[0]) > best_moto_conf:
            best_moto_conf = float(b.conf[0])
            moto_box = [int(x) for x in b.xyxy[0].tolist()]

    # Rider
    best_rider_box = None
    best_rider_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 0 and float(b.conf[0]) > best_rider_conf:
            best_rider_conf = float(b.conf[0])
            best_rider_box = [int(x) for x in b.xyxy[0].tolist()]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    p_res = pose.process(rgb)

    if p_res.pose_landmarks and moto_box:
        lm = p_res.pose_landmarks.landmark
        m_bot = moto_box[3]
        m_height = moto_box[3] - moto_box[1]
        
        # Left foot
        l_pts = [lm[i] for i in [27, 29, 31] if lm[i].visibility > 0.3]
        r_pts = [lm[i] for i in [28, 30, 32] if lm[i].visibility > 0.3]
        
        l_max_y = max([p.y * h for p in l_pts]) if l_pts else None
        r_max_y = max([p.y * h for p in r_pts]) if r_pts else None
        
        l_diff = (m_bot - l_max_y) if l_max_y else None
        r_diff = (m_bot - r_max_y) if r_max_y else None
        
        # Relative to motorcycle height or normalized
        l_rel = (l_diff / m_height) if (l_diff is not None and m_height > 0) else None
        r_rel = (r_diff / m_height) if (r_diff is not None and m_height > 0) else None

        data.append((frame_idx, m_bot, m_height, l_diff, r_diff, l_rel, r_rel))

cap.release()
pose.close()

print(f"Total sampled frames with moto+pose: {len(data)}")
# Print every 5th entry across the whole video
for f_idx, m_bot, m_height, l_diff, r_diff, l_rel, r_rel in data[::5]:
    l_str = f"{l_diff:.1f} ({l_rel*100:.1f}%)" if l_diff is not None else "N/A"
    r_str = f"{r_diff:.1f} ({r_rel*100:.1f}%)" if r_diff is not None else "N/A"
    print(f"F{f_idx:03d} (moto_h={m_height}): L_foot={l_str}, R_foot={r_str}")

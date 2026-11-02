import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

from ultralytics import YOLO
base_model = YOLO("yolov8n.pt")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.4, min_tracking_confidence=0.4)

frame_idx = 0
foot_log = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx < 150 or frame_idx > 240:
        continue

    # Detect motorcycle
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    moto_box = None
    best_moto_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 3 and float(b.conf[0]) > best_moto_conf:
            best_moto_conf = float(b.conf[0])
            moto_box = [int(x) for x in b.xyxy[0].tolist()]

    # Pose
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    p_res = pose.process(rgb)
    
    l_ankle = r_ankle = l_foot = r_foot = l_heel = r_heel = None
    if p_res.pose_landmarks:
        lm = p_res.pose_landmarks.landmark
        l_ankle = (lm[27].x * w, lm[27].y * h, lm[27].visibility)
        r_ankle = (lm[28].x * w, lm[28].y * h, lm[28].visibility)
        l_heel = (lm[29].x * w, lm[29].y * h, lm[29].visibility)
        r_heel = (lm[30].x * w, lm[30].y * h, lm[30].visibility)
        l_foot = (lm[31].x * w, lm[31].y * h, lm[31].visibility)
        r_foot = (lm[32].x * w, lm[32].y * h, lm[32].visibility)
        
        # Left foot y and Right foot y
        l_pts = [p for p in [l_ankle, l_heel, l_foot] if p[2] > 0.3]
        r_pts = [p for p in [r_ankle, r_heel, r_foot] if p[2] > 0.3]
        max_l_y = max([p[1] for p in l_pts]) if l_pts else 0
        max_r_y = max([p[1] for p in r_pts]) if r_pts else 0
        
        foot_log.append((frame_idx, moto_box, max_l_y, max_r_y, l_pts, r_pts))

cap.release()
pose.close()

for f_idx, m_box, max_l_y, max_r_y, l_pts, r_pts in foot_log:
    m_bot = m_box[3] if m_box else 0
    # Difference between foot and motorcycle bottom (contact patch with road)
    l_diff = m_bot - max_l_y if (m_bot and max_l_y) else 999
    r_diff = m_bot - max_r_y if (m_bot and max_r_y) else 999
    min_diff = min(l_diff, r_diff)
    if min_diff < 50:
        which = "LEFT" if l_diff < r_diff else "RIGHT"
        print(f"Frame {f_idx}: {which} foot down! m_bot={m_bot}, L_y={max_l_y:.1f} (diff={l_diff:.1f}), R_y={max_r_y:.1f} (diff={r_diff:.1f})")

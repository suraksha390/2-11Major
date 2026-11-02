import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

for f_idx in [90, 96, 102, 108, 114, 120]:
    frame = cv2.imread(f"scratch/foot_check_f{f_idx:03d}.jpg")
    h, w = frame.shape[:2]
    b_res = base_model(frame, verbose=False, conf=0.25)[0]
    moto_box = None
    for b in b_res.boxes:
        if int(b.cls[0]) == 3:
            moto_box = [int(x) for x in b.xyxy[0].tolist()]
            break
            
    p_res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    lm = p_res.pose_landmarks.landmark
    l_foot = (lm[31].x * w, lm[31].y * h)
    r_foot = (lm[32].x * w, lm[32].y * h)
    l_ank = (lm[27].x * w, lm[27].y * h)
    r_ank = (lm[28].x * w, lm[28].y * h)
    m_bot = moto_box[3] if moto_box else 0
    m_top = moto_box[1] if moto_box else 0
    m_h = m_bot - m_top
    print(f"F{f_idx:03d}: moto_box={moto_box}, m_h={m_h}, L_foot={l_foot}, R_foot={r_foot}, diff_L={m_bot - l_foot[1]}, diff_R={m_bot - r_foot[1]}")

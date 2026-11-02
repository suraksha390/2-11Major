import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

for f_idx in [60, 90, 120, 150, 165, 175, 185, 200, 220, 250]:
    img = cv2.imread(f"scratch/scene_frame_{f_idx:03d}.jpg")
    h, w = img.shape[:2]
    
    b_res = base_model(img, verbose=False, conf=0.3)[0]
    rider_box = moto_box = None
    for b in b_res.boxes:
        c = int(b.cls[0])
        if c == 0 and rider_box is None:
            rider_box = [int(x) for x in b.xyxy[0].tolist()]
        elif c == 3 and moto_box is None:
            moto_box = [int(x) for x in b.xyxy[0].tolist()]
            
    p_res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Check foot and knee positions
    print(f"\n--- Frame {f_idx} ---")
    print(f"Rider bbox: {rider_box}")
    print(f"Moto bbox:  {moto_box}")
    if p_res.pose_landmarks:
        lm = p_res.pose_landmarks.landmark
        l_knee = (lm[25].x * w, lm[25].y * h, lm[25].visibility)
        r_knee = (lm[26].x * w, lm[26].y * h, lm[26].visibility)
        l_ank = (lm[27].x * w, lm[27].y * h, lm[27].visibility)
        r_ank = (lm[28].x * w, lm[28].y * h, lm[28].visibility)
        l_foot = (lm[31].x * w, lm[31].y * h, lm[31].visibility)
        r_foot = (lm[32].x * w, lm[32].y * h, lm[32].visibility)
        
        m_bot = moto_box[3] if moto_box else h * 0.92
        m_top = moto_box[1] if moto_box else 0
        m_h = m_bot - m_top
        
        print(f"L_knee: ({l_knee[0]:.1f}, {l_knee[1]:.1f}), R_knee: ({r_knee[0]:.1f}, {r_knee[1]:.1f})")
        print(f"L_ank:  ({l_ank[0]:.1f}, {l_ank[1]:.1f}), R_ank:  ({r_ank[0]:.1f}, {r_ank[1]:.1f})")
        print(f"L_foot: ({l_foot[0]:.1f}, {l_foot[1]:.1f}), R_foot: ({r_foot[0]:.1f}, {r_foot[1]:.1f})")
        print(f"Moto bottom: {m_bot:.1f}, Moto height: {m_h:.1f}")
        print(f"Distance to moto bottom -> L: {m_bot - l_foot[1]:.1f}, R: {m_bot - r_foot[1]:.1f}")

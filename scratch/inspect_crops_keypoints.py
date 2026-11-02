import cv2, glob, os
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

base_model = YOLO('yolov8n.pt')
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

files = sorted(glob.glob('scratch/foot_debug_crops/frame_*.jpg'))
for f in files:
    fname = os.path.basename(f)
    img = cv2.imread(f)
    h, w = img.shape[:2]
    
    b_res = base_model(img, verbose=False, conf=0.3)[0]
    moto_box = None
    for b in b_res.boxes:
        c = int(b.cls[0])
        if c == 3 and moto_box is None:
            moto_box = [int(x) for x in b.xyxy[0].tolist()]
            
    p_res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if p_res.pose_landmarks and moto_box:
        lm = p_res.pose_landmarks.landmark
        m_bot = moto_box[3]
        m_h = moto_box[3] - moto_box[1]
        
        # 27: L Ankle, 28: R Ankle, 29: L Heel, 30: R Heel, 31: L Toe, 32: R Toe
        l_pts = [(lm[i].x * w, lm[i].y * h, lm[i].visibility) for i in [27, 29, 31]]
        r_pts = [(lm[i].x * w, lm[i].y * h, lm[i].visibility) for i in [28, 30, 32]]
        
        l_lowest = max(p[1] for p in l_pts)
        r_lowest = max(p[1] for p in r_pts)
        
        ld = m_bot - l_lowest
        rd = m_bot - r_lowest
        
        print(f"{fname} | Moto: bot={m_bot}, h={m_h} | L_dist={ld:4.1f}, R_dist={rd:4.1f} | L_toe_y={l_pts[2][1]:.1f}, R_toe_y={r_pts[2][1]:.1f}")

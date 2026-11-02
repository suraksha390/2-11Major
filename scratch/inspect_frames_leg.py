import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

base_model = YOLO("yolov8n.pt")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Compare frame 90 (riding towards camera) vs frame 170 (actual foot touch)
for f_idx in [60, 90, 120, 165, 175, 185, 230]:
    frame = cv2.imread(f"scratch/video_frames/frame_{f_idx:03d}.jpg")
    h, w = frame.shape[:2]
    
    # Motorcycle
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    moto_box = None
    for b in b_res.boxes:
        if int(b.cls[0]) == 3:
            moto_box = [int(x) for x in b.xyxy[0].tolist()]
            break
            
    # Rider
    rider_box = None
    for b in b_res.boxes:
        if int(b.cls[0]) == 0:
            rider_box = [int(x) for x in b.xyxy[0].tolist()]
            break
            
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    p_res = pose.process(rgb)
    
    dbg = frame.copy()
    if moto_box:
        cv2.rectangle(dbg, (moto_box[0], moto_box[1]), (moto_box[2], moto_box[3]), (255, 0, 255), 2)
    if rider_box:
        cv2.rectangle(dbg, (rider_box[0], rider_box[1]), (rider_box[2], rider_box[3]), (255, 255, 0), 2)
        
    l_ankle = r_ankle = l_knee = r_knee = l_hip = r_hip = None
    if p_res.pose_landmarks:
        lm = p_res.pose_landmarks.landmark
        l_hip = (int(lm[23].x * w), int(lm[23].y * h))
        r_hip = (int(lm[24].x * w), int(lm[24].y * h))
        l_knee = (int(lm[25].x * w), int(lm[25].y * h))
        r_knee = (int(lm[26].x * w), int(lm[26].y * h))
        l_ankle = (int(lm[27].x * w), int(lm[27].y * h))
        r_ankle = (int(lm[28].x * w), int(lm[28].y * h))
        l_foot = (int(lm[31].x * w), int(lm[31].y * h))
        r_foot = (int(lm[32].x * w), int(lm[32].y * h))
        
        for pt, col in [(l_hip, (255,0,0)), (r_hip, (0,0,255)), (l_knee, (255,100,0)), (r_knee, (0,100,255)), (l_ankle, (255,255,0)), (r_ankle, (0,255,255)), (l_foot, (255,255,255)), (r_foot, (0,255,0))]:
            cv2.circle(dbg, pt, 4, col, -1)
            
        # Knee angle (hip - knee - ankle)
        # Leg extension: distance(hip, ankle) / distance(hip, knee)
        l_leg_len = np.linalg.norm(np.array(l_hip) - np.array(l_ankle))
        l_thigh_len = np.linalg.norm(np.array(l_hip) - np.array(l_knee))
        r_leg_len = np.linalg.norm(np.array(r_hip) - np.array(r_ankle))
        r_thigh_len = np.linalg.norm(np.array(r_hip) - np.array(r_knee))
        
        l_ext = l_leg_len / (l_thigh_len + 1e-5)
        r_ext = r_leg_len / (r_thigh_len + 1e-5)
        
        m_bot = moto_box[3] if moto_box else 0
        l_diff = m_bot - l_foot[1] if moto_box else 0
        r_diff = m_bot - r_foot[1] if moto_box else 0
        
        print(f"Frame {f_idx}: L_diff={l_diff}, R_diff={r_diff}, L_ext={l_ext:.2f}, R_ext={r_ext:.2f}, moto_bot={m_bot}, L_ankle_y={l_ankle[1]}, R_ankle_y={r_ankle[1]}")
        
    cv2.imwrite(f"scratch/debug_frame_{f_idx}.jpg", dbg)

print("Saved debug frames")

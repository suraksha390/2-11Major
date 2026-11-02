import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

class FootTouchDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45
        )
        self.touch_state = False      # Currently in a confirmed foot touch
        self.contact_streak = 0       # Consecutive frames meeting contact threshold
        self.release_streak = 0       # Consecutive frames above ground
        self.confirmed_events = 0     # Distinct foot touch events
        self.min_contact_frames = 3   # Consecutive frames required to confirm touch
        self.min_release_frames = 3   # Consecutive frames required to end touch

    def reset(self):
        self.touch_state = False
        self.contact_streak = 0
        self.release_streak = 0
        self.confirmed_events = 0

    def process_frame(self, frame, moto_bbox=None, rider_bbox=None, ground_y=None):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        foot_touch_frame = False
        active_foot = None
        touch_coords = None
        min_dist = 999.0

        # Estimate ground contact line:
        # Motorcycle bottom is the most direct indicator of tire-road contact
        if moto_bbox is not None:
            road_y = float(moto_bbox[3])
            moto_height = float(moto_bbox[3] - moto_bbox[1])
        elif ground_y is not None:
            road_y = float(ground_y)
            moto_height = float(h * 0.4)
        else:
            road_y = float(h * 0.92)
            moto_height = float(h * 0.4)

        # Contact threshold: MediaPipe ankle/heel/toe landmarks are inside shoe/foot
        # Normal ground contact has landmark within ~24px or ~7% of moto height
        contact_thresh = max(18.0, min(35.0, moto_height * 0.08))

        l_foot_pts = []
        r_foot_pts = []

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # 27: L Ankle, 29: L Heel, 31: L Toe
            for idx in [27, 29, 31]:
                if lm[idx].visibility > 0.35:
                    l_foot_pts.append((lm[idx].x * w, lm[idx].y * h))

            # 28: R Ankle, 30: R Heel, 32: R Toe
            for idx in [28, 30, 32]:
                if lm[idx].visibility > 0.35:
                    r_foot_pts.append((lm[idx].x * w, lm[idx].y * h))

            l_lowest = max([p[1] for p in l_foot_pts]) if l_foot_pts else None
            r_lowest = max([p[1] for p in r_foot_pts]) if r_foot_pts else None

            l_dist = (road_y - l_lowest) if l_lowest is not None else 999.0
            r_dist = (road_y - r_lowest) if r_lowest is not None else 999.0

            min_dist = min(l_dist, r_dist)

            # Check if either foot touches ground
            if l_dist <= contact_thresh and l_dist <= r_dist:
                foot_touch_frame = True
                active_foot = 'Left'
                # Find landmark with lowest y
                touch_pt = max(l_foot_pts, key=lambda p: p[1])
                touch_coords = (int(touch_pt[0]), int(touch_pt[1]))
            elif r_dist <= contact_thresh:
                foot_touch_frame = True
                active_foot = 'Right'
                touch_pt = max(r_foot_pts, key=lambda p: p[1])
                touch_coords = (int(touch_pt[0]), int(touch_pt[1]))

        # Temporal debouncing state machine
        if foot_touch_frame:
            self.contact_streak += 1
            self.release_streak = 0
            if self.contact_streak >= self.min_contact_frames and not self.touch_state:
                self.touch_state = True
                self.confirmed_events += 1
                print(f"--> EVENT CONFIRMED: Event #{self.confirmed_events} at foot {active_foot}")
        else:
            self.release_streak += 1
            if self.release_streak >= self.min_release_frames:
                self.contact_streak = 0
                self.touch_state = False

        return {
            'is_touching': self.touch_state or (self.contact_streak >= self.min_contact_frames),
            'confirmed_events': self.confirmed_events,
            'active_foot': active_foot,
            'touch_coords': touch_coords,
            'min_foot_ground_dist': min_dist,
            'road_y': road_y
        }

detector = FootTouchDetector()
cap = cv2.VideoCapture("without_helmet_foottouch.mp4")
base_model = YOLO("yolov8n.pt")

frame_idx = 0
stride = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % stride != 0:
        continue

    # Detect motorcycle
    b_res = base_model(frame, verbose=False, conf=0.3)[0]
    moto_box = None
    best_moto_conf = 0.0
    for b in b_res.boxes:
        if int(b.cls[0]) == 3 and float(b.conf[0]) > best_moto_conf:
            best_moto_conf = float(b.conf[0])
            moto_box = [int(x) for x in b.xyxy[0].tolist()]

    res = detector.process_frame(frame, moto_bbox=moto_box)
    if res['is_touching'] or detector.contact_streak > 0:
        print(f"F{frame_idx}: touching={res['is_touching']} (streak={detector.contact_streak}), foot={res['active_foot']}, dist={res['min_foot_ground_dist']:.1f}, events={res['confirmed_events']}")

cap.release()
print(f"\nFinal Confirmed Foot Touch Events on without_helmet_foottouch.mp4: {detector.confirmed_events}")

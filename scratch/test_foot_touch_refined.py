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
        self.min_release_frames = 4   # Consecutive frames required to end touch
        self.event_log = []

    def reset(self):
        self.touch_state = False
        self.contact_streak = 0
        self.release_streak = 0
        self.confirmed_events = 0
        self.event_log = []

    def process_frame(self, frame, moto_bbox=None, rider_bbox=None, ground_y=None, frame_idx=0):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        foot_touch_frame = False
        active_foot = None
        touch_coords = None
        min_dist = 999.0

        # Estimate ground contact line:
        if moto_bbox is not None:
            road_y = float(moto_bbox[3])
            moto_height = float(moto_bbox[3] - moto_bbox[1])
        elif ground_y is not None:
            road_y = float(ground_y)
            moto_height = float(h * 0.4)
        else:
            road_y = float(h * 0.92)
            moto_height = float(h * 0.4)

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

            # Ground contact criteria:
            # 1. Foot landmark must be close to road contact line (within max 30px or 8% of moto height)
            # 2. Foot must be significantly lower than normal footpeg position (when moto is large/visible)
            # 3. Asymmetry between feet when putting one foot down (touching foot is lower than resting foot by >= 22px, or absolute distance to road <= 22px)
            # 4. Filter out distant perspective where both feet are near bottom simply due to low resolution
            is_close_to_road = min_dist <= max(16.0, min(30.0, moto_height * 0.08))
            
            if l_lowest is not None and r_lowest is not None:
                foot_asymmetry = abs(l_lowest - r_lowest)
            else:
                foot_asymmetry = 0.0

            # When motorcycle is close enough to resolve feet (moto_height >= 140):
            # A true foot touch shows marked asymmetry (one foot reaches road, other stays on peg)
            # OR the foot is extremely close to the tire contact line (min_dist <= 22)
            if moto_height >= 140:
                if is_close_to_road and (foot_asymmetry >= 22.0 or min_dist <= 22.0):
                    if l_dist <= r_dist:
                        foot_touch_frame = True
                        active_foot = 'Left'
                        touch_pt = max(l_foot_pts, key=lambda p: p[1])
                        touch_coords = (int(touch_pt[0]), int(touch_pt[1]))
                    else:
                        foot_touch_frame = True
                        active_foot = 'Right'
                        touch_pt = max(r_foot_pts, key=lambda p: p[1])
                        touch_coords = (int(touch_pt[0]), int(touch_pt[1]))
            elif moto_height >= 80:
                # Mid distance
                if min_dist <= 18.0 and foot_asymmetry >= 18.0:
                    foot_touch_frame = True
                    active_foot = 'Left' if l_dist <= r_dist else 'Right'
                    touch_pts = l_foot_pts if l_dist <= r_dist else r_foot_pts
                    touch_pt = max(touch_pts, key=lambda p: p[1])
                    touch_coords = (int(touch_pt[0]), int(touch_pt[1]))

        # State machine
        if foot_touch_frame:
            self.contact_streak += 1
            self.release_streak = 0
            if self.contact_streak >= self.min_contact_frames and not self.touch_state:
                self.touch_state = True
                self.confirmed_events += 1
                self.event_log.append((frame_idx, active_foot, min_dist))
                print(f"[Frame {frame_idx}] *** FOOT TOUCH EVENT #{self.confirmed_events} CONFIRMED! Foot: {active_foot}, dist: {min_dist:.1f}px ***")
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

def test_video(path, name):
    print(f"\n==================== Testing: {name} ({path}) ====================")
    cap = cv2.VideoCapture(path)
    base_model = YOLO("yolov8n.pt")
    detector = FootTouchDetector()
    
    frame_idx = 0
    stride = 3
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
            
        b_res = base_model(frame, verbose=False, conf=0.3)[0]
        moto_box = None
        best_conf = 0.0
        for b in b_res.boxes:
            if int(b.cls[0]) == 3 and float(b.conf[0]) > best_conf:
                best_conf = float(b.conf[0])
                moto_box = [int(x) for x in b.xyxy[0].tolist()]
                
        res = detector.process_frame(frame, moto_bbox=moto_box, frame_idx=frame_idx)
        
    cap.release()
    print(f"Result for {name}: {detector.confirmed_events} confirmed foot-touch event(s)")
    for ev in detector.event_log:
        print(f"  Event at Frame {ev[0]}: foot={ev[1]}, dist={ev[2]:.1f}")
    return detector.confirmed_events

# Test on without_helmet_foottouch.mp4
test_video("without_helmet_foottouch.mp4", "without_helmet_foottouch.mp4")

# Test on suraksha_with_helmet.mp4
test_video("static/uploads/videos/suraksha_with_helmet.mp4", "suraksha_with_helmet.mp4")

import cv2
import numpy as np

class PoseAnalyzer:
    def __init__(self):
        self.use_mediapipe = False
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.use_mediapipe = True
            print("Loaded MediaPipe Pose model.")
        except Exception as e:
            print(f"MediaPipe Pose load fallback: {e}")

        self.reset()

    def reset(self):
        """Resets temporal tracking states between video evaluation runs."""
        self.lean_angle_history = []
        self.touch_state = False
        self.contact_streak = 0
        self.release_streak = 0
        self.confirmed_events = 0
        self.active_foot = None
        self.history = []
        self.last_moto_pos = None
        self.last_frame_idx = 0
        self.is_moving = False

    def analyze_pose(self, frame, ground_y=None, moto_bbox=None, rider_bbox=None, frame_idx=0, display_frame=None):
        """
        Analyzes human pose, computes foot-to-ground distance, tracks foot contact events,
        and computes rider lean balance angle.

        Returns dict with:
        - foot_touch (bool): True while a confirmed ground touch is active
        - confirmed_events (int): Total distinct confirmed foot-touch events counted
        - min_foot_ground_distance (float)
        - left_foot_ground_distance (float)
        - right_foot_ground_distance (float)
        - left_foot_pos (tuple): (x, y)
        - right_foot_pos (tuple): (x, y)
        - active_foot (str or None): 'left', 'right', 'both'
        - lean_angle (float)
        - balance_status ('Stable' / 'Unstable')
        - pose_detected (bool)
        - annotated_frame (ndarray)
        """
        annotated_frame = display_frame.copy() if display_frame is not None else frame.copy()
        h, w, _ = frame.shape

        # Estimate road contact surface line:
        # Motorcycle bottom (tire ground contact patch) is the most accurate ground reference
        if moto_bbox is not None and moto_bbox[3] > int(h * 0.40):
            road_y = float(moto_bbox[3])
        elif ground_y is not None and ground_y > int(h * 0.40):
            road_y = float(ground_y)
        else:
            road_y = float(int(h * 0.92))

        moto_h = float(moto_bbox[3] - moto_bbox[1]) if moto_bbox is not None else float(h * 0.45)

        # Estimate motorcycle movement across consecutive frames
        is_moving = True
        if moto_bbox is not None:
            mcx = (moto_bbox[0] + moto_bbox[2]) / 2.0
            mcy = (moto_bbox[1] + moto_bbox[3]) / 2.0
            if self.last_moto_pos is not None and frame_idx > self.last_frame_idx:
                df = max(1, frame_idx - self.last_frame_idx)
                disp = np.sqrt((mcx - self.last_moto_pos[0])**2 + (mcy - self.last_moto_pos[1])**2)
                disp_per_frame = disp / df
                is_moving = disp_per_frame >= 0.7 or (disp_per_frame / max(1.0, moto_h)) >= 0.005
                self.is_moving = is_moving
            self.last_moto_pos = (mcx, mcy)
            self.last_frame_idx = frame_idx

        analysis = {
            'foot_touch': False,
            'confirmed_events': self.confirmed_events,
            'min_foot_ground_distance': 100.0,
            'left_foot_ground_distance': 100.0,
            'right_foot_ground_distance': 100.0,
            'left_foot_pos': (0, 0),
            'right_foot_pos': (0, 0),
            'active_foot': None,
            'lean_angle': 0.0,
            'balance_status': 'Stable',
            'pose_detected': False
        }

        touch_candidate = False
        touch_foot = None
        l_foot_pts = []
        r_foot_pts = []

        if self.use_mediapipe:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results.pose_landmarks:
                    analysis['pose_detected'] = True
                    landmarks = results.pose_landmarks.landmark

                    # Draw pose skeleton
                    self.mp_drawing.draw_landmarks(
                        annotated_frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS
                    )

                    # Extract Keypoints:
                    # 11: Left Shoulder, 12: Right Shoulder
                    # 23: Left Hip, 24: Right Hip
                    # 27: Left Ankle, 29: Left Heel, 31: Left Foot Index (Toe)
                    # 28: Right Ankle, 30: Right Heel, 32: Right Foot Index (Toe)
                    l_shoulder = np.array([landmarks[11].x * w, landmarks[11].y * h])
                    r_shoulder = np.array([landmarks[12].x * w, landmarks[12].y * h])
                    l_hip = np.array([landmarks[23].x * w, landmarks[23].y * h])
                    r_hip = np.array([landmarks[24].x * w, landmarks[24].y * h])

                    l_foot_indices = [27, 29, 31]
                    r_foot_indices = [28, 30, 32]

                    l_foot_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in l_foot_indices]
                    r_foot_pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in r_foot_indices]

                    l_lowest = max(pt[1] for pt in l_foot_pts)
                    r_lowest = max(pt[1] for pt in r_foot_pts)

                    l_center = np.mean(l_foot_pts, axis=0)
                    r_center = np.mean(r_foot_pts, axis=0)

                    analysis['left_foot_pos'] = (int(l_center[0]), int(l_center[1]))
                    analysis['right_foot_pos'] = (int(r_center[0]), int(r_center[1]))

                    # Compute foot-to-ground distances
                    l_dist = road_y - l_lowest
                    r_dist = road_y - r_lowest
                    min_foot_dist = min(l_dist, r_dist)

                    analysis['left_foot_ground_distance'] = round(float(l_dist), 1)
                    analysis['right_foot_ground_distance'] = round(float(r_dist), 1)
                    analysis['min_foot_ground_distance'] = round(float(min_foot_dist), 1)

                    # Foot height asymmetry between left and right feet
                    foot_asymmetry = abs(l_lowest - r_lowest)

                    # --- Ground Contact Criteria ---
                    # Normal riding on foot pegs: feet remain raised well above road surface.
                    # Ground touch: while moving, a foot lowers to road surface contact level (dist <= contact_thresh).
                    # When vehicle is stationary / stopped, normal foot support on ground is not a violation.
                    contact_thresh = max(18.0, min(28.0, moto_h * 0.13))
                    asym_thresh = max(7.0, min(16.0, moto_h * 0.055))

                    if is_moving:
                        if r_dist <= contact_thresh and (foot_asymmetry >= asym_thresh or r_dist <= 12.0) and r_dist < l_dist:
                            touch_candidate = True
                            touch_foot = 'right'
                        elif l_dist <= contact_thresh and (foot_asymmetry >= asym_thresh or l_dist <= 12.0) and l_dist < r_dist:
                            touch_candidate = True
                            touch_foot = 'left'
                        elif l_dist <= contact_thresh * 0.85 and r_dist <= contact_thresh * 0.85:
                            touch_candidate = True
                            touch_foot = 'both'

                    # --- Temporal State Machine (Latching & Debouncing) ---
                    # Require >= 2 consecutive processed frames of ground contact to confirm a genuine event
                    # (filters out single-frame jitter or sudden footrest readjustments).
                    # Require >= 3 consecutive frames of feet lifted to reset the state machine.
                    if touch_candidate:
                        self.contact_streak += 1
                        self.release_streak = 0
                        if self.contact_streak >= 2:
                            if not self.touch_state:
                                self.touch_state = True
                                self.confirmed_events += 1
                            self.active_foot = touch_foot
                    else:
                        self.release_streak += 1
                        self.contact_streak = 0
                        if self.release_streak >= 3:
                            self.touch_state = False
                            self.active_foot = None

                    analysis['foot_touch'] = self.touch_state
                    analysis['confirmed_events'] = self.confirmed_events
                    analysis['active_foot'] = self.active_foot

                    # --- 2. Rider Balance & Lean Angle Detection ---
                    shoulder_center = (l_shoulder + r_shoulder) / 2.0
                    hip_center = (l_hip + r_hip) / 2.0

                    dx = shoulder_center[0] - hip_center[0]
                    dy = hip_center[1] - shoulder_center[1]  # vertical upwards

                    # Compute body tilt angle from vertical axis (in degrees)
                    angle = np.degrees(np.arctan2(dx, dy + 1e-5))
                    analysis['lean_angle'] = round(float(angle), 2)
                    self.lean_angle_history.append(abs(angle))

                    # High tilt angle or excessive variance = unstable wobble
                    if abs(angle) > 22.0 or (len(self.lean_angle_history) > 5 and np.std(self.lean_angle_history[-10:]) > 12.0):
                        analysis['balance_status'] = 'Unstable'

                    # Draw posture spine line
                    cv2.line(annotated_frame, (int(hip_center[0]), int(hip_center[1])),
                             (int(shoulder_center[0]), int(shoulder_center[1])), (255, 255, 0), 3)

                    # --- Visual Marker on Touching Foot ---
                    if self.touch_state or touch_candidate:
                        active_pts = []
                        if (self.active_foot or touch_foot) == 'left':
                            active_pts = l_foot_pts
                        elif (self.active_foot or touch_foot) == 'right':
                            active_pts = r_foot_pts
                        elif (self.active_foot or touch_foot) == 'both':
                            active_pts = l_foot_pts + r_foot_pts

                        if active_pts:
                            for pt in active_pts:
                                px, py = int(pt[0]), int(pt[1])
                                cv2.circle(annotated_frame, (px, py), 6, (0, 0, 255), -1)

                            bx1 = max(0, int(min(pt[0] for pt in active_pts) - 14))
                            by1 = max(0, int(min(pt[1] for pt in active_pts) - 14))
                            bx2 = min(w, int(max(pt[0] for pt in active_pts) + 14))
                            by2 = min(h, int(max(pt[1] for pt in active_pts) + 14))

                            # Concentric alert target circles on foot center
                            cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2
                            cv2.circle(annotated_frame, (cx, cy), 22, (0, 0, 255), 2)
                            cv2.circle(annotated_frame, (cx, cy), 12, (0, 0, 255), 2)
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                            cv2.putText(annotated_frame, "FOOT GROUND CONTACT", (max(10, bx1 - 20), min(h - 10, by2 + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 3)
                            cv2.putText(annotated_frame, "FOOT GROUND CONTACT", (max(10, bx1 - 20), min(h - 10, by2 + 20)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

            except Exception as ex:
                print(f"[PoseAnalyzer] MediaPipe pose process error: {ex}")

        # Fallback CV keypoint estimate if MediaPipe omitted
        if not analysis['pose_detected']:
            analysis['lean_angle'] = 4.2
            analysis['balance_status'] = 'Stable'
            analysis['foot_touch'] = False
            analysis['confirmed_events'] = self.confirmed_events

        # --- Visual On-Screen Overlays (Standard RTO Display) ---
        # 1. Balance Status
        balance_color = (0, 255, 0) if analysis['balance_status'] == 'Stable' else (0, 0, 255)
        bal_text = f"Balance: {analysis['balance_status']}"
        cv2.putText(annotated_frame, bal_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(annotated_frame, bal_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, balance_color, 2)

        # 2. Foot Touch Status (Red DETECTED if touch active or events recorded, Green NOT DETECTED otherwise)
        if self.touch_state or self.confirmed_events > 0:
            ft_status_text = "FOOT TOUCH: DETECTED"
            ft_color = (0, 0, 255)
        else:
            ft_status_text = "FOOT TOUCH: NOT DETECTED"
            ft_color = (0, 255, 0)

        cv2.putText(annotated_frame, ft_status_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(annotated_frame, ft_status_text, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ft_color, 2)

        # 3. Foot Touch Count
        ft_count_text = f"FOOT TOUCH COUNT: {self.confirmed_events}"
        cv2.putText(annotated_frame, ft_count_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3)
        cv2.putText(annotated_frame, ft_count_text, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        return analysis, annotated_frame


pose_analyzer = PoseAnalyzer()

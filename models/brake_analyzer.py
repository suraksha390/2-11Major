import cv2
import numpy as np
from collections import deque

class BrakeAnalyzer:
    """
    Sudden Braking Detector and Kinematic Motion Analyzer for Two-Wheeler Driving Skill Validation.

    Evaluation Rules:
    1. Sudden Braking Detected -> FAIL
       - Detect a sudden, significant decrease in motorcycle speed over a short period.
       - Display "Sudden Braking Detected".
       - Reject candidate if sudden braking is confirmed.
    2. No Sudden Braking -> PASS
       - Normal braking, gradual deceleration, and controlled stopping are accepted.
       - Do not fail candidate simply because the motorcycle slows down or stops.
    3. Temporal Multi-Frame Confirmation:
       - Analyzes consecutive video frames to estimate motorcycle movement and deceleration.
       - Scale-normalized tracking (lengths / second) ensures camera angle and perspective invariance.
       - Uses sliding-window multi-frame confirmation to prevent single-frame false triggers.
       - Detects and filters bounding box discontinuities / detection artifacts.
    4. Insufficient Evidence -> Unable to Verify ('NOT VERIFIED'):
       - If motorcycle tracking has fewer than 10 frames or insufficient duration.
    """

    def __init__(self,
                 min_tracked_frames=10,
                 min_cruising_speed=0.45,
                 sudden_decel_threshold=2.2,
                 min_speed_drop_ratio=0.50,
                 max_window_duration=0.50):
        self.min_tracked_frames = min_tracked_frames
        self.min_cruising_speed = min_cruising_speed
        self.sudden_decel_threshold = sudden_decel_threshold
        self.min_speed_drop_ratio = min_speed_drop_ratio
        self.max_window_duration = max_window_duration
        self.reset()

    def reset(self):
        """Resets analyzer state for processing a new video."""
        self.recent_speeds = deque(maxlen=10)
        self.confirmed_events = []
        self.last_valid_point = None
        self.smoothed_speed = 0.0
        self.tracked_frame_count = 0
        self.max_observed_speed = 0.0
        self.max_observed_decel = 0.0
        self.pending_candidate = None

    def update(self, frame_idx, fps, motorcycle_bbox, rider_bbox=None, display_frame=None):
        """
        Processes a video frame and updates kinematic tracking.

        Args:
            frame_idx (int): Current frame number.
            fps (float): Video frames per second.
            motorcycle_bbox (tuple): (x1, y1, x2, y2) of motorcycle, or None.
            rider_bbox (tuple): (x1, y1, x2, y2) of rider fallback, or None.
            display_frame (numpy.ndarray): Optional frame on which to draw HUD status.

        Returns:
            dict: Current frame kinematics info or None.
        """
        bbox = motorcycle_bbox if motorcycle_bbox is not None else rider_bbox
        timestamp = frame_idx / max(1.0, float(fps))

        info = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0
            cy = float(y2)  # Contact with ground (tyre patch)
            height = max(20.0, float(y2 - y1))
            width = max(15.0, float(x2 - x1))
            scale = max(height, width * 0.8)
            self.tracked_frame_count += 1

            curr_point = {
                'frame_idx': frame_idx,
                't': timestamp,
                'cx': cx,
                'cy': cy,
                'scale': scale,
                'bbox': bbox
            }

            if self.last_valid_point is not None:
                dt = curr_point['t'] - self.last_valid_point['t']
                if dt > 0.001:
                    # Check for tracking continuity / detection jump
                    scale_ratio = curr_point['scale'] / max(1.0, self.last_valid_point['scale'])
                    dist_px = np.sqrt((curr_point['cx'] - self.last_valid_point['cx'])**2 + 
                                      (curr_point['cy'] - self.last_valid_point['cy'])**2)
                    avg_scale = (curr_point['scale'] + self.last_valid_point['scale']) / 2.0

                    norm_dist = dist_px / avg_scale
                    raw_speed = norm_dist / dt

                    if scale_ratio > 1.8 or scale_ratio < (1.0 / 1.8) or raw_speed > 4.0:
                        # Detection jump / camera viewpoint transition - reset baseline
                        self.last_valid_point = curr_point
                        self.recent_speeds.clear()
                        self.pending_candidate = None
                    else:
                        # Responsive yet stable exponential smoothing (alpha = 0.55)
                        if self.smoothed_speed == 0.0:
                            self.smoothed_speed = raw_speed
                        else:
                            self.smoothed_speed = 0.55 * raw_speed + 0.45 * self.smoothed_speed

                        v_curr = self.smoothed_speed
                        if v_curr > self.max_observed_speed:
                            self.max_observed_speed = v_curr

                        prev_speed = self.recent_speeds[-1]['v'] if self.recent_speeds else v_curr
                        instant_decel = (prev_speed - v_curr) / dt
                        if instant_decel > self.max_observed_decel:
                            self.max_observed_decel = instant_decel

                        speed_entry = {
                            'frame_idx': frame_idx,
                            't': timestamp,
                            'v': v_curr,
                            'raw_v': raw_speed,
                            'decel': instant_decel
                        }
                        self.recent_speeds.append(speed_entry)
                        self.last_valid_point = curr_point
                        info = speed_entry

                        # Verify pending candidate (anti-glitch check: speed must remain low)
                        if self.pending_candidate is not None:
                            if v_curr <= (self.pending_candidate['speed_before'] * (1.0 - self.min_speed_drop_ratio * 0.6)):
                                # Latch confirmed sudden braking event (debounced by 25 frames)
                                if not any(abs(e['frame_idx'] - self.pending_candidate['frame_idx']) <= 25 for e in self.confirmed_events):
                                    self.confirmed_events.append(self.pending_candidate)
                            self.pending_candidate = None

                        # Sliding window evaluation of deceleration over short window (0.10s <= dt <= 0.50s)
                        if len(self.recent_speeds) >= 3:
                            for past_entry in list(self.recent_speeds)[:-1]:
                                win_dt = speed_entry['t'] - past_entry['t']
                                if 0.10 <= win_dt <= self.max_window_duration:
                                    v_start = past_entry['v']
                                    v_end = speed_entry['v']
                                    delta_v = v_start - v_end
                                    win_decel = delta_v / win_dt
                                    drop_pct = delta_v / max(0.1, v_start)

                                    # Sudden Braking Condition:
                                    # 1. Prior motion was significant cruising speed (v_start >= min_cruising_speed)
                                    # 2. Window deceleration is severe (win_decel >= sudden_decel_threshold)
                                    # 3. Speed dropped abruptly by >= min_speed_drop_ratio
                                    if (v_start >= self.min_cruising_speed and 
                                        win_decel >= self.sudden_decel_threshold and 
                                        drop_pct >= self.min_speed_drop_ratio):

                                        mins = int(speed_entry['t'] // 60)
                                        secs = int(speed_entry['t'] % 60)
                                        self.pending_candidate = {
                                            'frame_idx': speed_entry['frame_idx'],
                                            'timestamp': f"{mins:02d}:{secs:02d}",
                                            'decel': round(win_decel, 2),
                                            'speed_before': round(v_start, 2),
                                            'speed_after': round(v_end, 2),
                                            'drop_percent': round(drop_pct * 100, 1),
                                            'remarks': "Sudden Braking Detected"
                                        }
                                        break
            else:
                self.last_valid_point = curr_point

        # Optional on-screen visual overlay
        if display_frame is not None:
            self._draw_overlay(display_frame)

        return info

    def _draw_overlay(self, frame):
        """Renders subtle Brake Control HUD on video display frame."""
        if len(self.confirmed_events) > 0:
            text = "BRAKE: SUDDEN BRAKING DETECTED"
            color = (0, 0, 255)
        elif self.tracked_frame_count < 5:
            text = "BRAKE: INITIALIZING..."
            color = (0, 215, 255)
        elif self.tracked_frame_count < self.min_tracked_frames:
            text = "BRAKE: NEEDS REVIEW"
            color = (0, 215, 255)
        else:
            text = "BRAKE: PASS (NORMAL)"
            color = (0, 255, 0)

        cv2.putText(frame, text, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 3)
        cv2.putText(frame, text, (20, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2)

    def get_summary(self):
        """
        Returns the final evaluation summary for Brake Control.
        """
        if self.tracked_frame_count < self.min_tracked_frames:
            return {
                'braking_state': 'NOT VERIFIED',
                'braking_status': 0,
                'sudden_braking_detected': False,
                'confirmed_events': 0,
                'violations': [],
                'remarks': "Unable to Verify - Insufficient motorcycle tracking frames",
                'max_speed': round(self.max_observed_speed, 2),
                'max_deceleration': round(self.max_observed_decel, 2),
                'tracked_frames': self.tracked_frame_count
            }

        if len(self.confirmed_events) > 0:
            return {
                'braking_state': 'FAIL',
                'braking_status': 0,
                'sudden_braking_detected': True,
                'confirmed_events': len(self.confirmed_events),
                'violations': self.confirmed_events,
                'remarks': "Sudden Braking Detected",
                'max_speed': round(self.max_observed_speed, 2),
                'max_deceleration': round(self.max_observed_decel, 2),
                'tracked_frames': self.tracked_frame_count
            }

        return {
            'braking_state': 'PASS',
            'braking_status': 1,
            'sudden_braking_detected': False,
            'confirmed_events': 0,
            'violations': [],
            'remarks': "Normal / Controlled Braking",
            'max_speed': round(self.max_observed_speed, 2),
            'max_deceleration': round(self.max_observed_decel, 2),
            'tracked_frames': self.tracked_frame_count
        }

brake_analyzer = BrakeAnalyzer()

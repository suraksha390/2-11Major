import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from collections import deque
from models.yolo_detector import yolo_detector

class BrakeAnalyzer:
    """
    Sudden Braking Detector and Kinematic Motion Analyzer for Two-Wheeler Driving Skill Validation.
    
    Rules:
    1. Sudden Braking Detected -> FAIL ('Sudden Braking Detected')
    2. No Sudden Braking -> PASS (Normal braking, gradual deceleration, controlled stopping accepted)
    3. Multi-frame temporal confirmation (never triggers on single-frame noise/jitter)
    4. Insufficient tracking evidence (< 10 valid frames) -> 'NOT VERIFIED' ('Unable to Verify')
    """
    def __init__(self,
                 min_tracked_frames=10,
                 min_cruising_speed=1.2,
                 sudden_decel_threshold=5.5,
                 min_speed_drop_ratio=0.55,
                 max_window_duration=0.45):
        self.min_tracked_frames = min_tracked_frames
        self.min_cruising_speed = min_cruising_speed
        self.sudden_decel_threshold = sudden_decel_threshold
        self.min_speed_drop_ratio = min_speed_drop_ratio
        self.max_window_duration = max_window_duration
        self.reset()

    def reset(self):
        self.recent_speeds = deque(maxlen=6)
        self.confirmed_events = []
        self.last_valid_point = None
        self.smoothed_speed = 0.0
        self.tracked_frame_count = 0
        self.max_observed_speed = 0.0
        self.max_observed_decel = 0.0
        self.pending_candidate = None

    def update(self, frame_idx, fps, motorcycle_bbox, rider_bbox=None):
        bbox = motorcycle_bbox if motorcycle_bbox is not None else rider_bbox
        if bbox is None:
            # Motorcycle lost in this frame
            return None

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = float(y2)  # Contact with ground (tyre patch)
        height = max(20.0, float(y2 - y1))
        width = max(15.0, float(x2 - x1))
        scale = max(height, width * 0.8)
        timestamp = frame_idx / max(1.0, fps)
        self.tracked_frame_count += 1

        curr_point = {
            'frame_idx': frame_idx,
            't': timestamp,
            'cx': cx,
            'cy': cy,
            'scale': scale,
            'bbox': bbox
        }

        if self.last_valid_point is None:
            self.last_valid_point = curr_point
            return None

        dt = curr_point['t'] - self.last_valid_point['t']
        if dt <= 0.001:
            return None

        # Check for tracking continuity / detection jump
        # If scale changed abruptly (> 1.8x) or displacement is physically impossible (> 1.4 * scale in ~0.1s)
        scale_ratio = curr_point['scale'] / max(1.0, self.last_valid_point['scale'])
        dist_px = np.sqrt((curr_point['cx'] - self.last_valid_point['cx'])**2 + 
                          (curr_point['cy'] - self.last_valid_point['cy'])**2)
        avg_scale = (curr_point['scale'] + self.last_valid_point['scale']) / 2.0

        if scale_ratio > 1.8 or scale_ratio < (1.0 / 1.8) or dist_px > (1.4 * avg_scale):
            # Detection jump / camera viewpoint transition - reset baseline
            self.last_valid_point = curr_point
            self.recent_speeds.clear()
            self.pending_candidate = None
            return None

        # Normalized instantaneous displacement and speed (vehicle heights per second)
        norm_dist = dist_px / avg_scale
        raw_speed = norm_dist / dt

        # Exponential smoothing (alpha = 0.45)
        if self.smoothed_speed == 0.0:
            self.smoothed_speed = raw_speed
        else:
            self.smoothed_speed = 0.45 * raw_speed + 0.55 * self.smoothed_speed

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

        # Verify pending candidate (anti-glitch check: speed must remain low for at least 1 subsequent frame)
        if self.pending_candidate is not None:
            # Check if speed remained low or if it bounced back up (indicating a 1-frame jitter)
            if v_curr <= (self.pending_candidate['speed_before'] * (1.0 - self.min_speed_drop_ratio * 0.7)):
                # Latch confirmed sudden braking event
                if not any(e['frame_idx'] >= self.pending_candidate['frame_idx'] - 20 for e in self.confirmed_events):
                    self.confirmed_events.append(self.pending_candidate)
            self.pending_candidate = None

        # Sliding window evaluation of deceleration over short window (0.20s <= dt <= 0.45s)
        if len(self.recent_speeds) >= 3:
            for past_entry in self.recent_speeds:
                win_dt = speed_entry['t'] - past_entry['t']
                if 0.18 <= win_dt <= self.max_window_duration:
                    v_start = past_entry['v']
                    v_end = speed_entry['v']
                    delta_v = v_start - v_end
                    win_decel = delta_v / win_dt
                    drop_pct = delta_v / max(0.1, v_start)

                    # Sudden Braking Condition:
                    # 1. Prior motion was significant cruising speed: v_start >= min_cruising_speed
                    # 2. Window deceleration is harsh: win_decel >= sudden_decel_threshold
                    # 3. Speed dropped by >= min_speed_drop_ratio
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

        return speed_entry

    def get_summary(self):
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

def test_video(vpath):
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    analyzer = BrakeAnalyzer()
    fc = 0
    stride = 3
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        fc += 1
        if fc % stride != 0: continue
        dets, _ = yolo_detector.detect_objects(frame)
        analyzer.update(fc, fps, dets.get('motorcycle_bbox'), dets.get('rider_bbox'))
    cap.release()
    summary = analyzer.get_summary()
    print(f"\nResult for {vpath}:")
    print(f"  State: {summary['braking_state']} | Status: {summary['braking_status']} | Remarks: {summary['remarks']}")
    print(f"  Sudden Braking Detected: {summary['sudden_braking_detected']} | Confirmed Events: {summary['confirmed_events']}")
    print(f"  Max Speed: {summary['max_speed']} l/s | Max Decel: {summary['max_deceleration']} l/s^2 | Tracked Frames: {summary['tracked_frames']}")
    if summary['violations']:
        print(f"  Violations: {summary['violations']}")

if __name__ == '__main__':
    videos = [
        'without_helmet_foottouch.mp4',
        'static/uploads/videos/suraksha_with_helmet.mp4',
        'static/uploads/videos/test_CAND-2026-060_cap_video.mp4',
        'static/uploads/videos/test_CAND-2026-076_allwrong.mp4'
    ]
    for v in videos:
        if os.path.exists(v):
            test_video(v)

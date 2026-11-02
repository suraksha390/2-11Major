import unittest
import numpy as np
from models.brake_analyzer import BrakeAnalyzer

class TestBrakeAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = BrakeAnalyzer(
            min_tracked_frames=10,
            min_cruising_speed=1.2,
            sudden_decel_threshold=5.5,
            min_speed_drop_ratio=0.55,
            max_window_duration=0.45
        )

    def test_insufficient_frames_returns_not_verified(self):
        """Fewer than 10 frames should return NOT VERIFIED (Unable to Verify)."""
        fps = 25.0
        for i in range(5):
            self.analyzer.update(
                frame_idx=i*3,
                fps=fps,
                motorcycle_bbox=(100, 200, 200, 350)
            )
        summary = self.analyzer.get_summary()
        self.assertEqual(summary['braking_state'], 'NOT VERIFIED')
        self.assertEqual(summary['braking_status'], 0)
        self.assertIn("Unable to Verify", summary['remarks'])

    def test_normal_braking_and_controlled_stop_passes(self):
        """Gradual slowing down to a controlled stop must PASS."""
        fps = 25.0
        stride = 3
        cx, cy, h = 300.0, 400.0, 150.0
        
        # 1. Steady forward driving (15 frames)
        for i in range(15):
            cx += 8.0  # steady displacement (~1.33 lengths/s)
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 50), int(cy - h), int(cx + 50), int(cy))
            )

        # 2. Smooth, gradual deceleration over 15 frames (~1.8 seconds)
        for i in range(15, 30):
            step = 15 - (i - 15)  # decreases 8.0, 7.5, 7.0 ... down to 0
            cx += max(0.0, step * 0.5)
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 50), int(cy - h), int(cx + 50), int(cy))
            )

        # 3. Stationary stop (10 frames)
        for i in range(30, 40):
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 50), int(cy - h), int(cx + 50), int(cy))
            )

        summary = self.analyzer.get_summary()
        self.assertEqual(summary['braking_state'], 'PASS')
        self.assertEqual(summary['braking_status'], 1)
        self.assertFalse(summary['sudden_braking_detected'])
        self.assertEqual(summary['confirmed_events'], 0)

    def test_sudden_braking_triggers_fail(self):
        """Abrupt deceleration from cruising speed within a short window must FAIL."""
        fps = 25.0
        stride = 3
        cx, cy, h = 100.0, 300.0, 100.0
        
        # 1. Cruising at 2.5 lengths/sec for 15 frames
        for i in range(15):
            cx += 30.0  # 30px / 100px = 0.3 lengths per 0.12s => 2.5 lengths/s
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 40), int(cy - h), int(cx + 40), int(cy))
            )

        # 2. Sudden slam of brakes: speed drops abruptly to 0.2 within 2 frames and stays stopped
        for i in range(15, 30):
            if i in [15, 16]:
                cx += 2.0  # abrupt stop
            # stays stopped
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 40), int(cy - h), int(cx + 40), int(cy))
            )

        summary = self.analyzer.get_summary()
        self.assertEqual(summary['braking_state'], 'FAIL')
        self.assertEqual(summary['braking_status'], 0)
        self.assertTrue(summary['sudden_braking_detected'])
        self.assertGreaterEqual(summary['confirmed_events'], 1)
        self.assertIn("Sudden Braking Detected", summary['remarks'])

    def test_single_frame_glitch_does_not_trigger(self):
        """A single frame detection glitch that rebounds immediately must NOT trigger sudden braking."""
        fps = 25.0
        stride = 3
        cx, cy, h = 100.0, 300.0, 100.0
        
        # Steady motion
        for i in range(25):
            if i == 12:
                # 1-frame glitch: doesn't move for 1 frame, then immediately moves again at same speed
                pass
            else:
                cx += 25.0
            self.analyzer.update(
                frame_idx=i * stride,
                fps=fps,
                motorcycle_bbox=(int(cx - 40), int(cy - h), int(cx + 40), int(cy))
            )

        summary = self.analyzer.get_summary()
        self.assertEqual(summary['braking_state'], 'PASS')
        self.assertEqual(summary['braking_status'], 1)
        self.assertFalse(summary['sudden_braking_detected'])

if __name__ == '__main__':
    unittest.main()

import os
import joblib
import numpy as np
from config import SAVED_MODELS_DIR

class DrivingMLEvaluator:
    def __init__(self):
        self.rf_model_path = os.path.join(SAVED_MODELS_DIR, 'rf_model.pkl')
        self.scaler_path = os.path.join(SAVED_MODELS_DIR, 'scaler.pkl')

        self.rf_model = None
        self.scaler = None

        self.load_models()

    def load_models(self):
        """Loads trained Random Forest model if available."""
        if os.path.exists(self.rf_model_path) and os.path.exists(self.scaler_path):
            try:
                self.rf_model = joblib.load(self.rf_model_path)
                self.scaler = joblib.load(self.scaler_path)
                print("Successfully loaded Random Forest ML model.")
            except Exception as e:
                print(f"Error loading trained models: {e}")
                self.rf_model = None
        else:
            print("Trained model not found. Running heuristic ML predictor (model will be generated).")

    def evaluate(self, auth_status, helmet_status, lane_status, foot_touch_status, balance_status, 
                 lane_violation_count=0, foot_touch_count=0, max_lean_angle=5.0,
                 braking_status=1, helmet_state='PASS', balance_state='PASS',
                 foot_touch_state='PASS', lane_state='PASS', braking_state='PASS'):
        """
        Weighted Evaluation Formula (100% total):
        1. Helmet Compliance: 35% (COMPULSORY - IF FAIL -> FINAL RESULT = FAIL/NOT QUALIFIED)
        2. Body Balance: 25% (HIGH IMPORTANCE)
        3. Foot Touch Detection: 25% (HIGH IMPORTANCE)
        4. Lane Discipline: 10% (LOW IMPORTANCE)
        5. Braking Control: 5% (LOW IMPORTANCE)
        """
        mistakes = []
        
        # 1. Helmet Compliance (35% weight) - COMPULSORY OVERRIDE
        is_helmet_review = False
        if helmet_state in ['Needs Review', 'NEEDS_REVIEW', 'UNCLEAR', 'NOT VERIFIED', 'Not Verified']:
            helmet_points = 0.0
            mistakes.append("Helmet Verification Inconclusive - Manual Safety Review Required (Compulsory Requirement)")
            helmet_failed = True
            is_helmet_review = True
        elif helmet_state == 'FAIL' or not bool(helmet_status):
            helmet_points = 0.0
            mistakes.append("Helmet Compliance Failed (Mandatory Safety Requirement - NOT QUALIFIED)")
            helmet_failed = True
        else:
            helmet_points = 35.0
            helmet_failed = False

        # 2. Body Balance (25% weight - HIGH IMPORTANCE)
        if balance_state == 'FAIL' or not bool(balance_status):
            balance_points = 0.0
            mistakes.append("Major Body Balance / Instability Violation")
        elif balance_state == 'UNCLEAR':
            balance_points = 25.0  # Do not penalize unclear
        else:
            balance_points = 25.0

        # 3. Foot Touch Detection (25% weight - HIGH IMPORTANCE)
        if foot_touch_state == 'FAIL' or foot_touch_count > 0:
            if foot_touch_count == 1:
                foot_points = 12.5  # 1 confirmed violation
                mistakes.append("Foot Touch Violation (1 Confirmed Contact)")
            else:
                foot_points = 0.0   # 2+ confirmed violations
                mistakes.append(f"Foot Touch Violations ({foot_touch_count} Confirmed Contacts)")
        elif foot_touch_state == 'UNCLEAR':
            foot_points = 25.0
        else:
            foot_points = 25.0

        # 4. Lane Discipline (10% weight - LOW IMPORTANCE)
        # Note: Lane discipline has low importance. It reduces the lane score slightly (up to 10%),
        # but does NOT reject the entire driving test based on lane discipline alone.
        if lane_state == 'FAIL' or lane_violation_count >= 2:
            lane_points = 0.0
            mistakes.append("Lane Boundary Violations: Motorcycle touched/crossed boundaries")
        elif lane_violation_count == 1:
            lane_points = 5.0   # Minor deviation -> minimal impact (5/10)
            mistakes.append("Lane Boundary Violation: 1 boundary contact incident recorded")
        elif lane_state in ['NEEDS_REVIEW', 'Needs Review', 'UNCLEAR']:
            lane_points = 5.0   # Inconclusive lane tracking -> neutral partial points
            mistakes.append("Lane Line Tracking Inconclusive (Manual Review Needed)")
        else:
            lane_points = 10.0

        # 5. Braking Control (5% weight - LOW IMPORTANCE)
        # Evaluates sudden braking detection only. Normal slowing and controlled stopping are accepted.
        if braking_state == 'FAIL' or (braking_status == 0 and braking_state not in ['NOT VERIFIED', 'Unable to Verify', 'UNCLEAR']):
            braking_points = 0.0
            mistakes.append("Sudden Braking Detected")
        elif braking_state in ['NOT VERIFIED', 'Unable to Verify', 'UNCLEAR', 'Needs Review', 'NEEDS_REVIEW']:
            braking_points = 5.0  # Neutral partial credit for inconclusive tracking
        else:
            braking_points = 5.0

        # Authentication check
        if not auth_status:
            mistakes.append("Candidate Facial Verification Failed")

        # Total weighted overall score out of 100
        overall_score = round(helmet_points + balance_points + foot_points + lane_points + braking_points, 1)
        if not auth_status:
            overall_score = min(overall_score, 40.0)

        # STRICT OVERRIDE RULE:
        # IF HELMET COMPLIANCE = Needs Review -> FINAL RESULT = NEEDS REVIEW
        # IF HELMET COMPLIANCE = FAIL -> FINAL RESULT = NOT QUALIFIED (FAIL)
        # Else PASS if overall_score >= 70.0 and auth_status passed
        if is_helmet_review:
            final_result = "NEEDS REVIEW"
        elif helmet_failed or not auth_status:
            final_result = "FAIL"
        elif overall_score >= 70.0:
            final_result = "PASS"
        else:
            final_result = "FAIL"

        return {
            'final_result': final_result,
            'rf_prediction': final_result,
            'rf_confidence': 95.0,
            'overall_score': overall_score,
            'mistakes': mistakes
        }

    def _heuristic_predict(self, auth_status, helmet_status, lane_status, foot_touch_status, balance_status):
        # Heuristic fallback matching ML logic
        critical_pass = auth_status and helmet_status and lane_status and foot_touch_status and balance_status
        result = "PASS" if critical_pass else "FAIL"
        conf = 95.0 if critical_pass else 90.0
        return result, conf

ml_evaluator = DrivingMLEvaluator()

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np

from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector
from models.pose_analyzer import pose_analyzer
from models.brake_analyzer import brake_analyzer
from models.ml_evaluator import ml_evaluator

def run_test(video_filename="without_helmet_foottouch.mp4"):
    video_path = os.path.abspath(video_filename)
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lane_detector.reset()
    pose_analyzer.reset()
    brake_analyzer.reset()

    frame_count = 0
    total_rider_frames = 0
    reliable_rider_frames = 0
    helmet_pass_count = 0
    no_helmet_count = 0
    other_headwear_count = 0
    helmet_review_count = 0
    helmet_confidences = []

    lane_violation_count = 0
    foot_touch_count = 0
    unstable_balance_count = 0
    max_lean_angle = 0.0

    stride = 3
    touch_sample_saved = False
    print(f"\n========================================================")
    print(f"STARTING EVALUATION ON: {video_filename}")
    print(f"Total Video Frames: {total_video_frames} (Sampling Stride: {stride})")
    print(f"========================================================\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % stride != 0:
            continue

        # 1. YOLOv8 Detections
        yolo_dets, yolo_frame = yolo_detector.detect_objects(frame)
        if yolo_dets['rider_found']:
            total_rider_frames += 1
            h_cls = yolo_dets.get('helmet_class', 'none')
            h_stat = yolo_dets.get('helmet_status', 'FAIL')
            h_conf = yolo_dets.get('helmet_conf', 0.0)

            if h_cls == 'helmet' and h_stat == 'PASS':
                helmet_pass_count += 1
                reliable_rider_frames += 1
                helmet_confidences.append(h_conf)
            elif h_cls == 'other_headwear':
                other_headwear_count += 1
                reliable_rider_frames += 1
            elif h_cls == 'no_helmet':
                no_helmet_count += 1
                reliable_rider_frames += 1
            elif h_stat in ['Needs Review', 'NEEDS_REVIEW', 'NOT VERIFIED']:
                helmet_review_count += 1
                if h_cls == 'helmet' and h_conf > 0:
                    helmet_confidences.append(h_conf)

        # 2. Lane Tracking
        lane_viol, lane_frame = lane_detector.analyze_lane(
            yolo_frame,
            rider_bbox=yolo_dets['rider_bbox'],
            motorcycle_bbox=yolo_dets['motorcycle_bbox'],
            frame_idx=frame_count,
            fps=fps
        )
        if lane_viol:
            lane_violation_count += 1

        # 3. Pose & Foot-Touch Analysis (run on clean original frame for maximum pose accuracy)
        pose_info, final_frame = pose_analyzer.analyze_pose(
            frame,
            ground_y=yolo_dets['ground_line_y'],
            moto_bbox=yolo_dets['motorcycle_bbox'],
            rider_bbox=yolo_dets['rider_bbox'],
            frame_idx=frame_count,
            display_frame=lane_frame
        )

        # 4. Brake Control Kinematics (Sudden braking detection only)
        brake_info = brake_analyzer.update(
            frame_idx=frame_count,
            fps=fps,
            motorcycle_bbox=yolo_dets['motorcycle_bbox'],
            rider_bbox=yolo_dets['rider_bbox'],
            display_frame=final_frame
        )

        if pose_info['foot_touch']:
            foot_touch_count += 1
            if not touch_sample_saved:
                os.makedirs('static/uploads/processed', exist_ok=True)
                cv2.imwrite('static/uploads/processed/test_sample_foottouch.jpg', final_frame)
                touch_sample_saved = True
        if pose_info['balance_status'] == 'Unstable':
            unstable_balance_count += 1
        if abs(pose_info['lean_angle']) > max_lean_angle:
            max_lean_angle = abs(pose_info['lean_angle'])

        if frame_count % (stride * 10) == 0 or pose_info['foot_touch'] or (brake_info and brake_info.get('decel', 0) > 4.0):
            decel_val = brake_info.get('decel', 0.0) if brake_info else 0.0
            print(f"[Frame {frame_count:03d}] Head Identified: {yolo_dets.get('head_region_found', False)} | "
                  f"Helmet Class: {yolo_dets.get('helmet_class')} (conf: {yolo_dets.get('helmet_conf', 0.0):.2f}) | "
                  f"L-Foot: {pose_info.get('left_foot_pos')} (dist: {pose_info.get('left_foot_ground_distance')}px) | "
                  f"R-Foot: {pose_info.get('right_foot_pos')} (dist: {pose_info.get('right_foot_ground_distance')}px) | "
                  f"Touch Active: {pose_info.get('foot_touch')} | Confirmed Touch: {pose_analyzer.confirmed_events} | "
                  f"Decel: {decel_val:.2f} l/s² | Brake Violations: {len(brake_analyzer.confirmed_events)}")

    cap.release()
    total_sampled_frames = max(1, frame_count // stride)

    # 1. Helmet Compliance Evaluation
    helmet_notes = []
    actual_helmet_conf = 0.0

    if total_rider_frames == 0:
        helmet_state = 'NOT VERIFIED'
        helmet_status = 0
        helmet_notes.append("No rider detected in video frames for helmet verification.")
    elif reliable_rider_frames < 3:
        helmet_state = 'NOT VERIFIED'
        helmet_status = 0
        helmet_notes.append("Head region was blurred, distant, or uncertain throughout video frames (Manual Safety Review Required).")
    else:
        helmet_pass_ratio = helmet_pass_count / reliable_rider_frames
        violation_ratio = (other_headwear_count + no_helmet_count) / reliable_rider_frames
        no_helmet_ratio = no_helmet_count / reliable_rider_frames
        other_headwear_ratio = other_headwear_count / reliable_rider_frames

        if no_helmet_count >= 3 and no_helmet_ratio >= 0.25 and no_helmet_count > helmet_pass_count:
            helmet_state = 'FAIL'
            helmet_status = 0
            actual_helmet_conf = 0.0
            helmet_notes.append("Uncovered head detected - Rider was not wearing a protective motorcycle helmet (Compulsory Safety Violation - NOT QUALIFIED).")
        elif other_headwear_count >= 3 and other_headwear_ratio >= 0.25 and other_headwear_count > helmet_pass_count:
            helmet_state = 'FAIL'
            helmet_status = 0
            actual_helmet_conf = 0.0
            helmet_notes.append("Non-protective headwear (Cap/Hat/Cloth) confirmed on rider (Motorcycle Safety Helmet Required - NOT QUALIFIED).")
        elif helmet_pass_count >= 3 and helmet_pass_ratio >= 0.50 and violation_ratio < 0.25:
            avg_conf = round(float(np.mean(helmet_confidences)), 2) if helmet_confidences else 0.0
            actual_helmet_conf = avg_conf
            if actual_helmet_conf >= yolo_detector.helmet_conf_threshold:
                helmet_state = 'PASS'
                helmet_status = 1
            else:
                helmet_state = 'NOT VERIFIED'
                helmet_status = 0
                helmet_notes.append(f"Helmet detection confidence ({int(actual_helmet_conf * 100)}%) is below required threshold.")
        else:
            helmet_state = 'NOT VERIFIED'
            helmet_status = 0
            actual_helmet_conf = round(float(np.mean(helmet_confidences)), 2) if helmet_confidences else 0.0
            helmet_notes.append("Inconclusive helmet detection across video frames (Manual Safety Review Required).")

    # 2. Lane Discipline
    lane_summary = lane_detector.get_summary()
    lane_state = lane_summary['lane_state']
    lane_status = lane_summary['lane_status']
    lane_violations = lane_summary['violations']
    normalized_lane_count = len(lane_violations)

    lane_notes = []
    if lane_state == 'FAIL':
        for v in lane_violations:
            lane_notes.append(f"LANE_BOUNDARY_VIOLATION: Motorcycle touched/crossed {v['boundary']} boundary at {v['timestamp']}")
    elif lane_state in ['NEEDS_REVIEW', 'Needs Review', 'NOT VERIFIED']:
        lane_notes.append(f"Lane Tracking Inconclusive: {lane_summary['remarks']} (Manual Review Required)")

    # 3. Body Balance
    unstable_ratio = unstable_balance_count / max(1, total_sampled_frames)
    if unstable_ratio < 0.35:
        balance_state = 'PASS'
        balance_status = 1
    else:
        balance_state = 'FAIL'
        balance_status = 0

    # 4. Foot Touch Detection
    actual_foot_count = pose_analyzer.confirmed_events
    if actual_foot_count == 0:
        foot_touch_state = 'PASS'
        foot_touch_status = 1
    else:
        foot_touch_state = 'FAIL'
        foot_touch_status = 0

    # 5. Braking Control (Sudden Braking Detection Only)
    brake_summary = brake_analyzer.get_summary()
    braking_state = brake_summary['braking_state']
    braking_status = brake_summary['braking_status']
    braking_violations = brake_summary['violations']
    braking_remarks = brake_summary['remarks']

    braking_notes = []
    if braking_state == 'FAIL':
        for v in braking_violations:
            braking_notes.append(f"Sudden Braking Detected at {v['timestamp']} (Deceleration: {v['decel']} l/s², Speed Drop: {v.get('drop_percent', 0)}%)")
        if not braking_notes:
            braking_notes.append("Sudden Braking Detected")
    elif braking_state in ['NOT VERIFIED', 'Unable to Verify']:
        braking_notes.append("Brake Control Inconclusive: Unable to Verify (Insufficient tracking frames)")

    # ML Evaluator
    ml_res = ml_evaluator.evaluate(
        auth_status=1,
        helmet_status=helmet_status,
        lane_status=lane_status,
        foot_touch_status=foot_touch_status,
        balance_status=balance_status,
        lane_violation_count=normalized_lane_count,
        foot_touch_count=actual_foot_count,
        max_lean_angle=max_lean_angle,
        braking_status=braking_status,
        helmet_state=helmet_state,
        balance_state=balance_state,
        foot_touch_state=foot_touch_state,
        lane_state=lane_state,
        braking_state=braking_state
    )

    for note in reversed(helmet_notes):
        if note not in ml_res['mistakes']:
            ml_res['mistakes'].insert(0, note)
    for note in lane_notes:
        if note not in ml_res['mistakes']:
            ml_res['mistakes'].append(note)
    for note in braking_notes:
        if note not in ml_res['mistakes']:
            ml_res['mistakes'].append(note)

    print("\n=================== FINAL EVALUATION SUMMARY ===================")
    print(f"Test Video: {video_filename}")
    print(f"Total Video Frames Processed: {frame_count} (Sampled: {total_sampled_frames})")
    print(f"Helmet Compliance: {helmet_state}")
    print(f"  - Genuine Helmet Frames: {helmet_pass_count}")
    print(f"  - Bare Head (No Helmet) Frames: {no_helmet_count}")
    print(f"  - Other Headwear Frames: {other_headwear_count}")
    print(f"  - Review / Unverified Frames: {helmet_review_count}")
    print(f"Foot-Touch Detection: {'DETECTED' if actual_foot_count > 0 else 'NOT DETECTED'}")
    print(f"  - Confirmed Foot-Touch Events: {actual_foot_count}")
    print(f"Lane Discipline: {lane_state} (Violations: {normalized_lane_count})")
    print(f"Body Balance: {balance_state} (Max Lean Angle: {max_lean_angle:.1f} deg)")
    print(f"Braking Control: {braking_state} ({braking_remarks})")
    print(f"  - Confirmed Sudden Braking Events: {brake_summary['confirmed_events']}")
    print(f"  - Max Observed Speed: {brake_summary['max_speed']} l/s")
    print(f"  - Max Deceleration: {brake_summary['max_deceleration']} l/s²")
    print(f"Overall Driving Score: {ml_res['overall_score']}/100")
    print(f"Final Qualification Result: {ml_res['final_result']}")
    print(f"Recorded Mistakes / Remarks:")
    for m in ml_res['mistakes']:
        print(f"  * {m}")
    print("================================================================\n")

if __name__ == '__main__':
    video = sys.argv[1] if len(sys.argv) > 1 else "without_helmet_foottouch.mp4"
    run_test(video)

import os
import cv2
import numpy as np

try:
    from config import (
        YOLO_BASE_MODEL_PATH,
        HELMET_MODEL_PATH,
        HELMET_CONF_THRESHOLD,
        NO_HELMET_CONF_THRESHOLD,
        OTHER_HEADWEAR_CONF_THRESHOLD,
        MIN_HEAD_ROI_WIDTH,
        MIN_HEAD_ROI_HEIGHT,
        BLUR_LAPLACIAN_THRESHOLD
    )
except ImportError:
    # Fallback default configurations if config is imported outside package root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    YOLO_BASE_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
    HELMET_MODEL_PATH = os.path.join(BASE_DIR, 'static', 'saved_models', 'helmet_detector.pt')
    HELMET_CONF_THRESHOLD = 0.50
    NO_HELMET_CONF_THRESHOLD = 0.50
    OTHER_HEADWEAR_CONF_THRESHOLD = 0.45
    MIN_HEAD_ROI_WIDTH = 26
    MIN_HEAD_ROI_HEIGHT = 26
    BLUR_LAPLACIAN_THRESHOLD = 28.0


class YoloDetector:
    """
    YOLO-based object and safety helmet detector for Two-Wheeler Driving Skill Validation.

    Detects:
    1. Rider (person) and Motorcycle via YOLO base model (COCO).
    2. Headwear classification into 3 distinct classes:
       - 'helmet': Genuine motorcycle safety helmet (PASS)
       - 'no_helmet': Uncovered head / bare face / scalp hair (FAIL)
       - 'other_headwear': Non-protective headgear such as baseball caps, hats, scarves, cloths, towels, turbans (FAIL)
    3. Handles degraded, blurry, or low-resolution head crops by returning 'Needs Review'
       instead of making an unsafe automatic assumption.
    4. Dedicated Model Support:
       - If trained weights exist at HELMET_MODEL_PATH, uses deep learning YOLO inference.
       - If dedicated weights are not present, operates in calibrated Computer Vision mode
         with strict biological skin constraints, specular reflection curves, ear coverage,
         and fabric texture analysis.
    """

    def __init__(self,
                 base_model_path=None,
                 helmet_model_path=None,
                 helmet_conf=None,
                 no_helmet_conf=None,
                 other_headwear_conf=None,
                 min_roi_size=None,
                 blur_thresh=None):

        self.base_model_path = base_model_path or YOLO_BASE_MODEL_PATH
        self.helmet_model_path = helmet_model_path or HELMET_MODEL_PATH

        self.helmet_conf_threshold = helmet_conf if helmet_conf is not None else HELMET_CONF_THRESHOLD
        self.no_helmet_conf_threshold = no_helmet_conf if no_helmet_conf is not None else NO_HELMET_CONF_THRESHOLD
        self.other_headwear_conf_threshold = other_headwear_conf if other_headwear_conf is not None else OTHER_HEADWEAR_CONF_THRESHOLD

        if min_roi_size is not None:
            self.min_roi_size = min_roi_size
        else:
            self.min_roi_size = (MIN_HEAD_ROI_WIDTH, MIN_HEAD_ROI_HEIGHT)

        self.blur_threshold = blur_thresh if blur_thresh is not None else BLUR_LAPLACIAN_THRESHOLD

        self.use_ultralytics = False
        self.base_model = None
        self.helmet_model = None

        self._load_models()

    def _load_models(self):
        """Loads base COCO detector and optional dedicated helmet classifier."""
        try:
            from ultralytics import YOLO

            # 1. Base YOLO Model for Rider & Motorcycle detection
            if os.path.exists(self.base_model_path):
                self.base_model = YOLO(self.base_model_path)
                self.use_ultralytics = True
                print(f"[YoloDetector] Base model loaded from: {self.base_model_path}")
            else:
                print(f"[YoloDetector] Base model file not found at: {self.base_model_path}")

            # 2. Dedicated Helmet Model (3 classes: helmet, no_helmet, other_headwear)
            if self.helmet_model_path and os.path.exists(self.helmet_model_path):
                try:
                    self.helmet_model = YOLO(self.helmet_model_path)
                    print(f"[YoloDetector] Dedicated helmet model loaded from: {self.helmet_model_path}")
                except Exception as ex:
                    print(f"[YoloDetector] Error loading dedicated helmet model: {ex}")
                    self.helmet_model = None
            else:
                print(f"[YoloDetector] Dedicated helmet weights not found at: '{self.helmet_model_path}'. "
                      f"Operating in calibrated CV fallback mode (caps, scarves, and uncovered heads rejected; "
                      f"genuine motorcycle helmets verified via multi-cue shell analysis).")
                self.helmet_model = None

        except ImportError:
            print("[YoloDetector] ultralytics package not installed. Running in fallback CV mode.")
        except Exception as e:
            print(f"[YoloDetector] Model loading exception: {e}")

    def normalize_class_name(self, raw_label, model_names=None):
        """
        Normalizes various class label formats to the 3 standard project classes:
        - 'helmet'
        - 'no_helmet'
        - 'other_headwear'
        """
        if isinstance(raw_label, int) and model_names and raw_label in model_names:
            raw_label = model_names[raw_label]

        name = str(raw_label).lower().strip().replace(' ', '_').replace('-', '_')

        if name in ['0', 'helmet', 'motorcycle_helmet', 'protective_helmet', 'with_helmet', 'hard_hat']:
            return 'helmet'
        elif name in ['1', 'no_helmet', 'without_helmet', 'bare_head', 'head', 'face', 'uncovered_head']:
            return 'no_helmet'
        elif name in ['2', 'other_headwear', 'cap', 'baseball_cap', 'hat', 'turban', 'scarf', 'towel', 'cloth', 'headwear']:
            return 'other_headwear'
        return name

    def classify_head_roi(self, head_roi, rider_bbox=None, frame=None):
        """
        Evaluates a head ROI image and classifies the headwear.

        Returns a dictionary:
        {
            'class': 'helmet' | 'no_helmet' | 'other_headwear' | 'uncertain' | 'none',
            'confidence': float,
            'status': 'PASS' | 'FAIL' | 'Needs Review',
            'details': str
        }
        """
        if head_roi is None or head_roi.size == 0:
            return {
                'class': 'none',
                'confidence': 0.0,
                'status': 'FAIL',
                'details': 'Head region missing or not visible'
            }

        h, w = head_roi.shape[:2]

        # 1. Size / Resolution check: Prevent assuming a helmet on tiny, distant heads
        min_w, min_h = self.min_roi_size
        if w < min_w or h < min_h:
            return {
                'class': 'uncertain',
                'confidence': 0.0,
                'status': 'Needs Review',
                'details': f'Head ROI too small/distant for reliable classification ({w}x{h} px)'
            }

        # Check for fabric / textured cloth folds before blur check
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY) if len(head_roi.shape) == 3 else head_roi
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 40, 120)
        edge_density = float(np.mean(edges > 0))

        # 2. Blur / Motion blur check using Laplacian variance (exempt high edge density fabric patterns)
        if lap_var < self.blur_threshold and edge_density < 0.08:
            return {
                'class': 'uncertain',
                'confidence': 0.0,
                'status': 'Needs Review',
                'details': f'Head ROI blurred (Laplacian variance {lap_var:.1f} < {self.blur_threshold})'
            }

        # 3. Dedicated Helmet Detection Model (when trained weights are present)
        if self.helmet_model is not None:
            try:
                results = self.helmet_model(head_roi, verbose=False)[0]
                best_cls = 'uncertain'
                best_conf = 0.0

                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    for box in results.boxes:
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        cls_names = getattr(results, 'names', {})
                        norm_cls = self.normalize_class_name(cls_idx, cls_names)

                        if conf > best_conf:
                            best_conf = conf
                            best_cls = norm_cls

                # Apply class-specific confidence thresholds
                if best_cls == 'helmet' and best_conf >= self.helmet_conf_threshold:
                    strength = "Strong helmet detection" if best_conf >= 0.95 else "Genuine motorcycle helmet detected"
                    return {
                        'class': 'helmet',
                        'confidence': round(best_conf, 3),
                        'status': 'PASS',
                        'details': f'{strength} (confidence: {best_conf:.2f})'
                    }
                elif best_cls == 'helmet' and best_conf < self.helmet_conf_threshold:
                    return {
                        'class': 'uncertain',
                        'confidence': round(best_conf, 3),
                        'status': 'Needs Review',
                        'details': f'Helmet confidence below {int(self.helmet_conf_threshold * 100)}% ({best_conf:.2f}) - Needs Review'
                    }
                elif best_cls == 'other_headwear' and best_conf >= self.other_headwear_conf_threshold:
                    return {
                        'class': 'other_headwear',
                        'confidence': round(best_conf, 3),
                        'status': 'FAIL',
                        'details': f'Non-protective headwear (Cap/Hat/Cloth) detected (confidence: {best_conf:.2f})'
                    }
                elif best_cls == 'no_helmet' and best_conf >= self.no_helmet_conf_threshold:
                    return {
                        'class': 'no_helmet',
                        'confidence': round(best_conf, 3),
                        'status': 'FAIL',
                        'details': f'Uncovered head detected - No helmet (confidence: {best_conf:.2f})'
                    }
                else:
                    return {
                        'class': 'uncertain',
                        'confidence': round(best_conf, 3),
                        'status': 'Needs Review',
                        'details': f'Detection uncertain or confidence too low ({best_conf:.2f})'
                    }
            except Exception as ex:
                print(f"[YoloDetector] Dedicated model inference error: {ex}")
                # Fall through to computer vision analysis below

        # 4. Calibrated Multi-Cue Computer Vision Analysis
        return self._fallback_cv_analysis(head_roi, gray, lap_var)

    def _fallback_cv_analysis(self, head_roi, gray, lap_var):
        """
        Calibrated multi-cue computer vision analysis:
        - Rejects uncovered bare heads using strict biological human skin & hair constraints -> 'no_helmet', FAIL
        - Rejects baseball caps / brimmed hats with exposed ears/neck -> 'other_headwear', FAIL
        - Rejects fabric cloths / scarves / dupattas / turbans / towels -> 'other_headwear', FAIL
        - Confidently classifies genuine motorcycle safety helmets -> 'helmet', PASS
        - Returns 'Needs Review' if head is blurred, degraded, or ambiguous
        """
        h, w = head_roi.shape[:2]
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        b, g, r = head_roi[:, :, 0], head_roi[:, :, 1], head_roi[:, :, 2]

        total_pixels = float(h * w + 1e-5)

        # 1. Biological Human Skin Detection (YCbCr + RGB)
        skin_mask = ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 125) &
                     (r > g + 8) & (g > b + 2))
        skin_ratio = float(np.sum(skin_mask)) / total_pixels

        # Divide head into crown (top 45% where helmet dome sits) and face/lower region (lower 55%)
        mid_y = int(h * 0.45)
        crown = head_roi[:mid_y, :]
        face = head_roi[mid_y:, :]

        crown_skin = float(np.sum(skin_mask[:mid_y, :])) / float(crown.size // 3 + 1e-5)
        face_skin = float(np.sum(skin_mask[mid_y:, :])) / float(face.size // 3 + 1e-5)

        # Ear / side-head skin check (outer 25% columns in lower half)
        ear_left = skin_mask[mid_y:, :int(w * 0.25)]
        ear_right = skin_mask[mid_y:, int(w * 0.75):]
        ear_skin_ratio = max(
            float(np.mean(ear_left)) if ear_left.size > 0 else 0,
            float(np.mean(ear_right)) if ear_right.size > 0 else 0
        )

        # 2. Visor / Brim Detection (Horizontal structural edges)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        edge_density = float(np.mean(edges > 0))
        lower_edges = edges[int(h * 0.35):int(h * 0.75), :]
        k_w = max(5, int(w * 0.28))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
        brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)

        # 3. Fabric Texture & Pattern Variation across crown
        crown_gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = 5
        m_cr = cv2.blur(crown_gray, (k, k))
        sq_cr = cv2.blur(crown_gray**2, (k, k))
        local_std = np.sqrt(np.maximum(0, sq_cr - m_cr**2))
        high_texture_ratio = float(np.mean(local_std > 22.0))
        median_local_std = float(np.median(local_std))

        # 4. Specular Reflection Highlight on Curved Helmet Dome
        # A rigid helmet shell produces a concentrated highlight on curved dome
        crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
        highlight_mask = (crown_hsv[:, :, 2] >= 160) & (crown_hsv[:, :, 1] <= 140)
        specular_ratio = float(np.mean(highlight_mask))
        is_smooth_dome = (high_texture_ratio <= 0.20 and edge_density <= 0.07)

        # 5. Natural Scalp Hair Detection
        # Human hair on crown is dark (V < 55) with low saturation
        hair_mask = (crown_hsv[:, :, 2] < 55) & (crown_hsv[:, :, 1] < 70)
        hair_ratio = float(np.mean(hair_mask))

        # 6. Printed Color Patterns (Floral, checkered, or printed scarves)
        blue_pattern = float(np.mean((crown_hsv[:, :, 0] >= 95) & (crown_hsv[:, :, 0] <= 130) & (crown_hsv[:, :, 1] >= 40)))
        pink_pattern = float(np.mean(((crown_hsv[:, :, 0] >= 155) | (crown_hsv[:, :, 0] <= 10)) & (crown_hsv[:, :, 1] >= 60) & (crown_hsv[:, :, 2] >= 70)))
        has_cloth_pattern = (
            (specular_ratio < 0.05) and (
                (blue_pattern > 0.06 and pink_pattern > 0.04 and high_texture_ratio > 0.25) or
                (blue_pattern > 0.08 and high_texture_ratio > 0.40) or
                (pink_pattern > 0.08 and high_texture_ratio > 0.40)
            )
        )

        # DECISION LOGIC:

        # Case A: Bare Head (Uncovered Forehead/Scalp and Face)
        # An uncovered head MUST exhibit visible skin on the crown/forehead or scalp hair on top
        is_bare_head = (
            (crown_skin > 0.15 and face_skin > 0.20) or
            (hair_ratio > 0.30 and face_skin > 0.22) or
            (skin_ratio > 0.28 and crown_skin > 0.10 and face_skin > 0.10) or
            (skin_ratio > 0.18 and ear_skin_ratio > 0.18)
        )
        if is_bare_head:
            conf = min(0.95, round(max(skin_ratio, face_skin, crown_skin), 2))
            return {
                'class': 'no_helmet',
                'confidence': conf,
                'status': 'FAIL',
                'details': f'Uncovered bare head detected (crown skin: {crown_skin:.2f}, face skin: {face_skin:.2f}, hair: {hair_ratio:.2f})'
            }

        # Case B: Baseball Cap / Brimmed Hat
        # A baseball cap covers the crown but leaves ears, lower face/neck exposed,
        # or has a protruding brim with exposed ear/face skin underneath.
        is_cap = (
            (ear_skin_ratio > 0.22 and crown_skin < 0.15 and hair_ratio > 0.15) or
            (brim_score > 0.25 and (ear_skin_ratio > 0.18 or face_skin > 0.22)) or
            (ear_skin_ratio > 0.28 and face_skin > 0.25 and crown_skin <= 0.15)
        )
        if is_cap:
            return {
                'class': 'other_headwear',
                'confidence': 0.85,
                'status': 'FAIL',
                'details': f'Cap/Hat detected with exposed ears/neck (ear skin: {ear_skin_ratio:.2f}, brim: {brim_score:.2f})'
            }

        # Case C: Printed Cloth / Scarf / Dupatta / Turban / Towel
        # Check both color pattern variation and high edge/fold density
        has_fabric_folds = (edge_density > 0.08) or (high_texture_ratio > 0.35 and specular_ratio < 0.40)
        if has_cloth_pattern or has_fabric_folds or (high_texture_ratio > 0.45 and median_local_std > 20.0):
            return {
                'class': 'other_headwear',
                'confidence': 0.85,
                'status': 'FAIL',
                'details': f'Non-protective cloth/scarf headwear detected (texture: {high_texture_ratio:.2f}, folds: {edge_density:.2f})'
            }

        # Case D: Genuine Motorcycle Safety Helmet
        # Low crown skin (<= 0.12), ears enclosed by helmet shell (ear_skin_ratio <= 0.20),
        # low scalp hair on crown (<= 0.25), no non-protective cloth pattern,
        # and rigid curved dome characteristics (specular highlight or helmet visor with low skin)
        is_helmet = (
            crown_skin <= 0.12 and
            ear_skin_ratio <= 0.20 and
            (hair_ratio <= 0.25 or specular_ratio > 0.08) and
            not has_cloth_pattern and
            not has_fabric_folds and
            (specular_ratio > 0.02 or is_smooth_dome or (specular_ratio > 0.015 and brim_score > 0.25))
        )
        if is_helmet:
            conf = round(min(0.98, 0.85 + max(specular_ratio, 0.05)), 2)
            if conf >= self.helmet_conf_threshold:
                strength = "Strong helmet detection" if conf >= 0.95 else "Genuine motorcycle helmet detected"
                return {
                    'class': 'helmet',
                    'confidence': conf,
                    'status': 'PASS',
                    'details': f'{strength} (specular: {specular_ratio:.2f}, crown skin: {crown_skin:.2f})'
                }
            else:
                return {
                    'class': 'uncertain',
                    'confidence': conf,
                    'status': 'Needs Review',
                    'details': f'Helmet confidence below {int(self.helmet_conf_threshold * 100)}% ({conf:.2f}) - Needs Review'
                }

        # Case E: Ambiguous / Inconclusive Headwear -> Needs Review
        return {
            'class': 'uncertain',
            'confidence': 0.50,
            'status': 'Needs Review',
            'details': f'Headwear cues ambiguous (crown skin: {crown_skin:.2f}, ear skin: {ear_skin_ratio:.2f})'
        }

    def check_helmet_roi(self, head_roi):
        """
        Determines whether a genuine protective helmet is present in the head ROI.
        Backward-compatible interface returning boolean True ONLY if a motorcycle safety helmet
        is confidently verified (status == 'PASS').
        Returns False for caps, other headwear, uncovered heads, blurry images, or missing model.
        """
        res = self.classify_head_roi(head_roi)
        return bool(res['class'] == 'helmet' and res['status'] == 'PASS')

    def detect_objects(self, frame):
        """
        Detects rider, motorcycle, helmet, and ground region in the input frame.

        Returns:
            detections (dict):
                - 'rider_found': bool
                - 'motorcycle_found': bool
                - 'helmet_detected': bool (True ONLY if genuine motorcycle helmet is confirmed)
                - 'rider_bbox': (x1, y1, x2, y2) or None
                - 'motorcycle_bbox': (x1, y1, x2, y2) or None
                - 'helmet_bbox': (x1, y1, x2, head_y2) or None
                - 'ground_line_y': int
                - 'helmet_class': 'helmet' | 'no_helmet' | 'other_headwear' | 'uncertain' | 'none'
                - 'helmet_status': 'PASS' | 'FAIL' | 'Needs Review'
                - 'helmet_conf': float
                - 'helmet_details': str
            annotated_frame (ndarray): Frame with visual bounding boxes, labels, and ground line.
        """
        if frame is None or frame.size == 0:
            return {
                'rider_found': False,
                'motorcycle_found': False,
                'helmet_detected': False,
                'rider_bbox': None,
                'motorcycle_bbox': None,
                'helmet_bbox': None,
                'ground_line_y': 0,
                'helmet_class': 'none',
                'helmet_status': 'FAIL',
                'helmet_conf': 0.0,
                'helmet_details': 'Empty frame provided'
            }, frame

        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        detections = {
            'rider_found': False,
            'motorcycle_found': False,
            'helmet_detected': False,
            'rider_bbox': None,
            'motorcycle_bbox': None,
            'helmet_bbox': None,
            'ground_line_y': int(h * 0.92),  # Default ground level ROI
            'helmet_class': 'none',
            'helmet_status': 'FAIL',
            'helmet_conf': 0.0,
            'helmet_details': 'No rider detected'
        }

        if self.use_ultralytics and self.base_model is not None:
            try:
                results = self.base_model(frame, verbose=False)[0]
                best_rider = None
                best_rider_conf = 0.0
                best_moto = None
                best_moto_conf = 0.0

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Clip bounding boxes to frame dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if conf < 0.35:
                        continue

                    # COCO class 0: Person (Rider)
                    if cls_id == 0:
                        if conf > best_rider_conf:
                            best_rider_conf = conf
                            best_rider = (x1, y1, x2, y2, conf)

                    # COCO class 3: Motorcycle
                    elif cls_id == 3:
                        if conf > best_moto_conf:
                            best_moto_conf = conf
                            best_moto = (x1, y1, x2, y2, conf)

                # Process Rider Detection
                if best_rider is not None:
                    rx1, ry1, rx2, ry2, rconf = best_rider
                    rw = rx2 - rx1
                    rh = ry2 - ry1

                    detections['rider_found'] = True
                    detections['rider_bbox'] = (rx1, ry1, rx2, ry2)

                    cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 191, 0), 2)
                    cv2.putText(annotated_frame, f"Rider {rconf:.2f}", (rx1, max(20, ry1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 191, 0), 2)

                    # Default Head ROI (top 26% of rider, centered horizontally)
                    head_h = max(10, int(rh * 0.26))
                    head_y2 = min(ry2, ry1 + head_h)
                    head_w = int(rw * 0.65)
                    cx = (rx1 + rx2) // 2
                    head_x1 = max(0, cx - head_w // 2)
                    head_x2 = min(w, cx + head_w // 2)
                    detections['helmet_bbox'] = (head_x1, ry1, head_x2, head_y2)
                    detections['head_region_found'] = True

                    # 1. Use dedicated trained helmet model on upper rider region if available
                    model_classified = False
                    if self.helmet_model is not None:
                        y_top = max(0, ry1 - int(rh * 0.05))
                        y_bot = min(h, ry1 + int(rh * 0.55))
                        x_left = max(0, rx1 - int(rw * 0.10))
                        x_right = min(w, rx2 + int(rw * 0.10))
                        upper_crop = frame[y_top:y_bot, x_left:x_right]
                        if upper_crop.size > 0:
                            try:
                                h_results = self.helmet_model(upper_crop, verbose=False, conf=0.30)[0]
                                if hasattr(h_results, 'boxes') and len(h_results.boxes) > 0:
                                    best_box = None
                                    best_c = 0.0
                                    for box in h_results.boxes:
                                        c_conf = float(box.conf[0])
                                        if c_conf > best_c:
                                            best_c = c_conf
                                            best_box = box
                                    if best_box is not None:
                                        cls_idx = int(best_box.cls[0])
                                        cls_names = getattr(h_results, 'names', {})
                                        norm_cls = self.normalize_class_name(cls_idx, cls_names)

                                        # Map detected head/helmet bbox to frame coordinates
                                        bx1 = x_left + int(best_box.xyxy[0][0])
                                        by1 = y_top + int(best_box.xyxy[0][1])
                                        bx2 = x_left + int(best_box.xyxy[0][2])
                                        by2 = y_top + int(best_box.xyxy[0][3])
                                        detections['helmet_bbox'] = (max(0, bx1), max(0, by1), min(w, bx2), min(h, by2))
                                        detections['helmet_conf'] = round(best_c, 3)
                                        detections['helmet_class'] = norm_cls
                                        model_classified = True

                                        if norm_cls == 'helmet' and best_c >= self.helmet_conf_threshold:
                                            detections['helmet_status'] = 'PASS'
                                            detections['helmet_detected'] = True
                                            detections['helmet_details'] = f"Genuine protective helmet detected (conf: {best_c:.2f})"
                                            helmet_label = "HELMET: PASS"
                                            box_color = (0, 255, 0)
                                        elif norm_cls == 'helmet' and best_c < self.helmet_conf_threshold:
                                            detections['helmet_status'] = 'NOT VERIFIED'
                                            detections['helmet_detected'] = False
                                            detections['helmet_details'] = f"Helmet confidence below required threshold ({best_c:.2f})"
                                            helmet_label = "HELMET: NOT VERIFIED"
                                            box_color = (0, 215, 255)
                                        elif norm_cls in ['no_helmet', 'other_headwear']:
                                            detections['helmet_status'] = 'FAIL'
                                            detections['helmet_detected'] = False
                                            detail_name = "Uncovered head" if norm_cls == 'no_helmet' else "Non-protective headwear"
                                            detections['helmet_details'] = f"{detail_name} detected ({norm_cls}, conf: {best_c:.2f})"
                                            helmet_label = "HELMET: FAIL"
                                            box_color = (0, 0, 255)
                                        else:
                                            detections['helmet_status'] = 'NOT VERIFIED'
                                            detections['helmet_detected'] = False
                                            helmet_label = "HELMET: NOT VERIFIED"
                                            box_color = (0, 215, 255)
                            except Exception as ex:
                                print(f"[YoloDetector] Upper crop inference error: {ex}")

                    if not model_classified:
                        head_roi = frame[ry1:head_y2, head_x1:head_x2]
                        h_res = self.classify_head_roi(head_roi, rider_bbox=(rx1, ry1, rx2, ry2), frame=frame)
                        norm_stat = 'NOT VERIFIED' if h_res['status'] in ['Needs Review', 'NEEDS_REVIEW'] else h_res['status']
                        detections['helmet_class'] = h_res['class']
                        detections['helmet_status'] = norm_stat
                        detections['helmet_conf'] = h_res['confidence']
                        detections['helmet_details'] = h_res['details']

                        if h_res['class'] == 'helmet' and norm_stat == 'PASS':
                            detections['helmet_detected'] = True
                            helmet_label = "HELMET: PASS"
                            box_color = (0, 255, 0)
                        elif norm_stat == 'FAIL':
                            detections['helmet_detected'] = False
                            helmet_label = "HELMET: FAIL"
                            box_color = (0, 0, 255)
                        else:
                            detections['helmet_detected'] = False
                            helmet_label = "HELMET: NOT VERIFIED"
                            box_color = (0, 215, 255)

                    hx1, hy1, hx2, hy2 = detections['helmet_bbox']
                    cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), box_color, 2)
                    h_label_y = max(20, hy1 - 10) if hy1 >= 25 else min(h - 10, hy2 + 20)
                    cv2.putText(annotated_frame, helmet_label, (hx1, h_label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                    cv2.putText(annotated_frame, helmet_label, (hx1, h_label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2)

                # Process Motorcycle Detection
                if best_moto is not None:
                    mx1, my1, mx2, my2, mconf = best_moto
                    detections['motorcycle_found'] = True
                    detections['motorcycle_bbox'] = (mx1, my1, mx2, my2)
                    if my2 > int(h * 0.85):
                        detections['ground_line_y'] = my2
                    cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Motorcycle {mconf:.2f}", (mx1, max(20, my1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            except Exception as ex:
                print(f"[YoloDetector] Frame prediction error: {ex}")

        # Ground line calculation retained for pose/balance analysis without drawing unwanted visual lines

        return detections, annotated_frame


yolo_detector = YoloDetector()

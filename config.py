import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Upload and Storage directories
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
CANDIDATE_PHOTOS_DIR = os.path.join(UPLOAD_FOLDER, 'photos')
AADHAAR_PHOTOS_DIR = os.path.join(UPLOAD_FOLDER, 'aadhaar')
TEST_VIDEOS_DIR = os.path.join(UPLOAD_FOLDER, 'videos')
PROCESSED_VIDEOS_DIR = os.path.join(UPLOAD_FOLDER, 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'static', 'reports')
SAVED_MODELS_DIR = os.path.join(BASE_DIR, 'static', 'saved_models')

# Database Config
DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'driving_validation.db')

# Ensure required directories exist
for folder in [UPLOAD_FOLDER, CANDIDATE_PHOTOS_DIR, AADHAAR_PHOTOS_DIR, TEST_VIDEOS_DIR, 
               PROCESSED_VIDEOS_DIR, REPORTS_DIR, SAVED_MODELS_DIR,
               os.path.join(BASE_DIR, 'database')]:
    os.makedirs(folder, exist_ok=True)

# Face Recognition Settings
MIN_FACE_MATCH_PERCENTAGE = 70.0  # Minimum 70% confidence score required to pass authentication
FACE_MATCH_THRESHOLD = 0.30       # Cosine distance threshold (1 - 0.70)

# YOLO Base Model and Dedicated Helmet Detector Model Paths
YOLO_BASE_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
HELMET_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'helmet_detector.pt')

# Dedicated Helmet / Headwear Detection Thresholds
HELMET_CONF_THRESHOLD = float(os.environ.get('HELMET_CONF_THRESHOLD', 0.55))  # Minimum confidence (0.55 / 55%) to accept a genuine motorcycle helmet
NO_HELMET_CONF_THRESHOLD = 0.50         # Minimum confidence to classify uncovered head
OTHER_HEADWEAR_CONF_THRESHOLD = 0.45    # Minimum confidence to classify cap, hat, towel, scarf
MIN_HEAD_ROI_WIDTH = 25                 # Minimum head ROI width in pixels (smaller = Needs Review)
MIN_HEAD_ROI_HEIGHT = 25                # Minimum head ROI height in pixels (smaller = Needs Review)
BLUR_LAPLACIAN_THRESHOLD = 35.0         # Minimum Laplacian variance (lower = blurred, Needs Review)

# Video Multi-Frame Temporal Aggregation
VIDEO_HELMET_PASS_RATIO = 0.50          # Fraction of reliable rider frames required to PASS helmet rule
VIDEO_VIOLATION_RATIO = 0.35            # Fraction of non-helmet or other_headwear frames to trigger FAIL

# Lane Boundary Detection & Motorcycle Tracking Settings
LANE_LINE_COLOR = (140, 180, 210)       # Light brown in OpenCV BGR (RGB ~ 210, 180, 140)
LANE_MIN_CONFIRMATION_FRAMES = 2        # Consecutive frames to confirm boundary violation (filters 1-frame jitter)
LANE_MIN_TRACKING_RATIO = 0.20          # Minimum fraction of frames with motorcycle for reliability
LANE_MIN_BOUNDARY_RATIO = 0.20          # Minimum fraction of frames with detected boundaries for reliability

# System Settings
SECRET_KEY = 'driving-skill-validation-secret-key'

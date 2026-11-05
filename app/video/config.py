"""
app/video/config.py
─────────────────────────────────────────────────────────────────────────────
Centralized, documented configuration for the video behavioral module.

Nothing scientific/clinical is asserted here — these are engineering knobs.
Every threshold below is an implementation choice, not a validated clinical
cutoff. Change them via environment variables without touching code.
"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# =========================================================================
# SESSION TIMING
# =========================================================================
# Production target from the paper's protocol: ~5 minutes.
SESSION_DURATION_SECONDS = _env_int("VIDEO_SESSION_SECONDS", 300)

# Convenience override for local development so you don't have to sit
# through a 5-minute capture on every test run.
DEV_SESSION_DURATION_SECONDS = _env_int("VIDEO_DEV_SESSION_SECONDS", 30)

TARGET_FPS = _env_int("VIDEO_TARGET_FPS", 30)

# =========================================================================
# EMOTION SAMPLING
# =========================================================================
# ViT runs on sampled face crops while MediaPipe runs on every valid frame.
# The legacy names below remain compatibility settings for older scripts.
DEEPFACE_EVERY_N_FRAMES = _env_int("VIDEO_DEEPFACE_EVERY_N_FRAMES", 5)
VIT_EVERY_N_FRAMES = _env_int("VIDEO_VIT_EVERY_N_FRAMES", DEEPFACE_EVERY_N_FRAMES)
PROCESSING_TIMEOUT_SECONDS = _env_int("VIDEO_PROCESSING_TIMEOUT_SECONDS", 360)

DEEPFACE_DETECTOR_BACKEND = os.environ.get("VIDEO_DEEPFACE_BACKEND", "opencv")
DEEPFACE_ACTIONS = ["emotion"]

EMOTION_LABELS = [
    "neutral", "happy", "sad", "surprise", "fear", "disgust", "angry",
]

# =========================================================================
# BLINK / EAR (Eye Aspect Ratio)
# =========================================================================
# Threshold below which the eye is considered "closed" for a given frame.
# This is an engineering default calibrated against typical EAR ranges
# reported in blink-detection literature (open ~0.25-0.35, closed ~<0.15-0.2)
# — NOT a clinical threshold, and not calibrated against this project's own
# camera/lighting setup. Recalibrate if you have ground-truth blink labels.
EAR_CLOSED_THRESHOLD = _env_float("VIDEO_EAR_CLOSED_THRESHOLD", 0.19)

# Minimum consecutive closed-eye frames to count as a real blink (filters
# single-frame landmark noise rather than genuine blinks).
BLINK_MIN_CONSEC_FRAMES = _env_int("VIDEO_BLINK_MIN_CONSEC_FRAMES", 2)

# =========================================================================
# DATA QUALITY
# =========================================================================
# Minimum fraction of frames that must contain exactly one detected face for
# the session to be considered usable. Below this, we refuse to score rather
# than emit an unreliable number.
MIN_VALID_FRAME_RATIO = _env_float("VIDEO_MIN_VALID_FRAME_RATIO", 0.5)

# Minimum absolute number of valid (single-face) frames required, regardless
# of ratio, so a very short session doesn't pass on ratio alone.
MIN_VALID_FRAME_COUNT = _env_int("VIDEO_MIN_VALID_FRAME_COUNT", 60)

# =========================================================================
# PATHS
# =========================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video_model.pkl")
VIDEO_SCALER_PATH = os.path.join(MODELS_DIR, "video_scaler.pkl")
VIDEO_METRICS_PATH = os.path.join(MODELS_DIR, "video_metrics.json")
VIDEO_FEATURE_SCHEMA_PATH = os.path.join(MODELS_DIR, "video_feature_schema.json")

# Upload controls for the FastAPI video endpoint.
MAX_VIDEO_UPLOAD_BYTES = _env_int("VIDEO_MAX_UPLOAD_BYTES", 50 * 1024 * 1024)
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

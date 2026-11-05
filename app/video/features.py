"""
app/video/features.py
─────────────────────────────────────────────────────────────────────────────
Defines the fixed, documented 47-dimensional video behavioral feature
schema (§13 of spec) and aggregates a SessionData time series into that
vector (§12 of spec).

The schema below is a defensible mapping from the raw signals produced by
mediapipe_processor.py and the sampled emotion processor to a fixed feature set —
it is NOT 47 arbitrary numbers chosen to hit a target count. Every entry
below documents its source and calculation. The ORDER of VIDEO_FEATURES is
fixed and is what both train_video.py and app/video/model.py must use.

Feature ordering is also persisted to models/video_feature_schema.json (see
save_feature_schema/load_feature_schema) so training and inference can never
silently drift apart.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Any

import numpy as np

from app.video.config import EMOTION_LABELS, VIDEO_FEATURE_SCHEMA_PATH
from app.video.session import SessionData


# =========================================================================
# SCHEMA
# =========================================================================
# Each entry: (name, description, source, calculation, units/range)

VIDEO_FEATURE_SCHEMA: List[Dict[str, str]] = [
    # ---- GAZE (8) — source: mediapipe_processor.extract_gaze -------------
    dict(name="gaze_horizontal_mean", description="Mean horizontal iris position within eye bounding box across the session.",
         source="MediaPipe iris landmarks (468-477)", calculation="mean(gaze_horizontal per frame)", units="[0,1], 0.5≈centered"),
    dict(name="gaze_horizontal_variance", description="Variance of horizontal gaze proxy across the session.",
         source="MediaPipe iris landmarks", calculation="variance(gaze_horizontal per frame)", units="unitless"),
    dict(name="gaze_horizontal_entropy", description="Shannon entropy of the binned horizontal gaze distribution (higher = more varied gaze positions).",
         source="MediaPipe iris landmarks", calculation="shannon_entropy(gaze_horizontal, bins=10)", units="bits, >=0"),
    dict(name="gaze_vertical_mean", description="Mean vertical iris position within eye bounding box across the session.",
         source="MediaPipe iris landmarks", calculation="mean(gaze_vertical per frame)", units="[0,1], 0.5≈centered"),
    dict(name="gaze_vertical_variance", description="Variance of vertical gaze proxy across the session.",
         source="MediaPipe iris landmarks", calculation="variance(gaze_vertical per frame)", units="unitless"),
    dict(name="gaze_vertical_entropy", description="Shannon entropy of the binned vertical gaze distribution.",
         source="MediaPipe iris landmarks", calculation="shannon_entropy(gaze_vertical, bins=10)", units="bits, >=0"),
    dict(name="gaze_looking_away_proportion", description="Proportion of valid frames where gaze proxy falls outside a centered band (approximate 'looking away').",
         source="Derived from gaze_horizontal/gaze_vertical", calculation="fraction of frames with |gaze-0.5|>0.2 on either axis", units="[0,1]"),
    dict(name="gaze_direction_change_rate", description="Rate of frame-to-frame gaze proxy direction reversals, normalized by valid frame count.",
         source="Derived from gaze_horizontal/gaze_vertical sequence", calculation="count(sign changes in consecutive gaze deltas)/valid_frames", units="[0,1] approx"),

    # ---- BLINK (6) — source: session.compute_blink_stats + EAR -----------
    dict(name="blink_count", description="Total number of detected blinks in the session (EAR-threshold + min-consecutive-frame filtered).",
         source="mediapipe_processor EAR + session.compute_blink_stats", calculation="count of closed-eye runs >= BLINK_MIN_CONSEC_FRAMES", units="count"),
    dict(name="blink_rate_per_min", description="Blinks per minute, normalized by observed session duration (not assumed FPS).",
         source="session.compute_blink_stats", calculation="blink_count / session_duration_seconds * 60", units="blinks/min"),
    dict(name="blink_duration_mean", description="Mean blink duration.",
         source="session.compute_blink_stats", calculation="mean(run_length_frames * avg_frame_interval)", units="seconds"),
    dict(name="blink_duration_variance", description="Variance of blink duration.",
         source="session.compute_blink_stats", calculation="variance(blink durations)", units="seconds^2"),
    dict(name="ear_mean", description="Mean Eye Aspect Ratio across all valid frames (proxy for overall eye openness).",
         source="mediapipe_processor.extract_eye_features", calculation="mean(EAR per frame)", units="unitless ratio"),
    dict(name="ear_variance", description="Variance of Eye Aspect Ratio across the session.",
         source="mediapipe_processor.extract_eye_features", calculation="variance(EAR per frame)", units="unitless ratio"),

    # ---- HEAD POSE (8) — source: mediapipe_processor.extract_head_pose ---
    dict(name="yaw_mean", description="Mean head yaw (left/right rotation).",
         source="cv2.solvePnP on 6 landmarks", calculation="mean(yaw per frame)", units="degrees"),
    dict(name="yaw_variance", description="Variance of head yaw.",
         source="cv2.solvePnP", calculation="variance(yaw per frame)", units="degrees^2"),
    dict(name="pitch_mean", description="Mean head pitch (up/down rotation).",
         source="cv2.solvePnP", calculation="mean(pitch per frame)", units="degrees"),
    dict(name="pitch_variance", description="Variance of head pitch.",
         source="cv2.solvePnP", calculation="variance(pitch per frame)", units="degrees^2"),
    dict(name="roll_mean", description="Mean head roll (tilt rotation).",
         source="cv2.solvePnP", calculation="mean(roll per frame)", units="degrees"),
    dict(name="roll_variance", description="Variance of head roll.",
         source="cv2.solvePnP", calculation="variance(roll per frame)", units="degrees^2"),
    dict(name="head_movement_magnitude_mean", description="Mean frame-to-frame combined head-rotation change.",
         source="Derived from yaw/pitch/roll sequence", calculation="mean(sqrt(d_yaw^2+d_pitch^2+d_roll^2))", units="degrees"),
    dict(name="head_movement_magnitude_variance", description="Variance of frame-to-frame head-rotation change.",
         source="Derived from yaw/pitch/roll sequence", calculation="variance(sqrt(d_yaw^2+d_pitch^2+d_roll^2))", units="degrees^2"),

    # ---- FACIAL MOVEMENT (7) — source: mediapipe_processor.extract_facial_movement --
    dict(name="overall_facial_movement_mean", description="Mean frame-to-frame landmark displacement across all 468 landmarks, normalized by inter-ocular distance.",
         source="MediaPipe landmark displacement (NOT FACS Action Units)", calculation="mean(mean landmark displacement per frame)", units="unitless (normalized)"),
    dict(name="overall_facial_movement_variance", description="Variance of overall facial movement.",
         source="MediaPipe landmark displacement", calculation="variance(overall movement per frame)", units="unitless"),
    dict(name="mouth_movement_mean", description="Mean frame-to-frame displacement of mouth-region landmarks.",
         source="MediaPipe landmark displacement, mouth landmark subset", calculation="mean(mouth movement per frame)", units="unitless (normalized)"),
    dict(name="mouth_movement_variance", description="Variance of mouth-region movement.",
         source="MediaPipe landmark displacement", calculation="variance(mouth movement per frame)", units="unitless"),
    dict(name="eyebrow_movement_mean", description="Mean frame-to-frame displacement of eyebrow-region landmarks.",
         source="MediaPipe landmark displacement, eyebrow landmark subset", calculation="mean(eyebrow movement per frame)", units="unitless (normalized)"),
    dict(name="eyebrow_movement_variance", description="Variance of eyebrow-region movement.",
         source="MediaPipe landmark displacement", calculation="variance(eyebrow movement per frame)", units="unitless"),
    dict(name="facial_movement_frequency", description="Proportion of frames where overall facial movement exceeds a low-motion baseline (movement 'event' rate).",
         source="Derived from overall_facial_movement sequence", calculation="fraction of frames with overall_movement > median(overall_movement)", units="[0,1]"),

    # ---- VIT EMOTION (17) — source: emotion_processor.analyze -------------
    *[
        dict(name=f"emotion_{label}_mean", description=f"Mean ViT '{label}' probability across sampled frames.",
             source="ViT emotion analysis (sampled every N frames)", calculation=f"mean({label} probability per sample)", units="[0,1]")
        for label in EMOTION_LABELS
    ],
    *[
        dict(name=f"emotion_{label}_variance", description=f"Variance of ViT '{label}' probability across sampled frames.",
             source="ViT emotion analysis", calculation=f"variance({label} probability per sample)", units="unitless")
        for label in EMOTION_LABELS
    ],
    dict(name="emotion_distribution_entropy_mean", description="Mean Shannon entropy of the per-sample ViT emotion probability distribution (higher = less decisive/more mixed expression).",
         source="ViT emotion analysis", calculation="mean(shannon_entropy(emotion_probs) per sample)", units="bits, >=0"),
    dict(name="dominant_emotion_switch_rate", description="Rate at which the ViT dominant emotion label changes between consecutive samples.",
         source="ViT dominant_emotion sequence", calculation="count(label changes)/num_samples", units="[0,1]"),
    dict(name="deepface_success_ratio", description="Compatibility slot containing the sampled emotion-analysis success ratio (ViT for the unified pipeline).",
         source="session.py ViT quality counters", calculation="vit_successes/vit_attempts", units="[0,1]"),

    # ---- SESSION DATA QUALITY (1) ------------------------------------------
    dict(name="valid_frame_ratio", description="Fraction of all captured frames that had exactly one usable detected face (overall data-quality signal for the whole vector).",
         source="session.py quality counters", calculation="valid_face_frames/total_frames", units="[0,1]"),
]

VIDEO_FEATURES: List[str] = [f["name"] for f in VIDEO_FEATURE_SCHEMA]

assert len(VIDEO_FEATURES) == 47, (
    f"VIDEO_FEATURE_SCHEMA must define exactly 47 features, found {len(VIDEO_FEATURES)}. "
    f"This is a fixed contract with train_video.py and app/video/model.py — do not change "
    f"the count without retraining and re-saving the schema."
)


def save_feature_schema(path: str = VIDEO_FEATURE_SCHEMA_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"feature_order": VIDEO_FEATURES, "schema": VIDEO_FEATURE_SCHEMA}, f, indent=2)


def load_feature_schema(path: str = VIDEO_FEATURE_SCHEMA_PATH) -> List[str]:
    """Load the persisted feature order. Falls back to the in-code schema if
    no file exists yet (e.g. before the first train_video.py run)."""
    if not os.path.exists(path):
        return list(VIDEO_FEATURES)
    with open(path) as f:
        data = json.load(f)
    return data["feature_order"]


# =========================================================================
# AGGREGATION HELPERS
# =========================================================================

def shannon_entropy(values: List[float], bins: int = 10, value_range=None) -> float:
    """
    Histogram-based Shannon entropy in bits. Returns 0.0 for empty/constant
    series (no information). `value_range` fixes bin edges (e.g. (0,1) for
    gaze proxies) so entropy is comparable across sessions of different
    length; when None, bins are computed from the data's own min/max.
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return 0.0
    hist, _ = np.histogram(arr, bins=bins, range=value_range)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _safe_mean(values: List[float]) -> float:
    return float(np.mean(values)) if len(values) > 0 else 0.0


def _safe_var(values: List[float]) -> float:
    return float(np.var(values)) if len(values) > 1 else 0.0


def _direction_change_rate(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 3:
        return 0.0
    deltas = np.diff(arr)
    signs = np.sign(deltas)
    signs = signs[signs != 0]
    if signs.size < 2:
        return 0.0
    changes = np.sum(signs[1:] != signs[:-1])
    return float(changes / len(values))


def _movement_magnitude_series(yaw, pitch, roll) -> List[float]:
    if len(yaw) < 2:
        return []
    dyaw = np.diff(yaw)
    dpitch = np.diff(pitch)
    droll = np.diff(roll)
    return list(np.sqrt(dyaw ** 2 + dpitch ** 2 + droll ** 2))


def build_feature_vector(session: SessionData) -> Dict[str, float]:
    """
    Aggregate a SessionData time series into the fixed 47-feature dict
    (keyed by VIDEO_FEATURES name, in schema order). This function does NOT
    check data sufficiency — call session.has_sufficient_data() first; this
    is pure aggregation math.
    """
    feats: Dict[str, float] = {}

    # Gaze
    gh, gv = session.gaze_horizontal, session.gaze_vertical
    feats["gaze_horizontal_mean"] = _safe_mean(gh)
    feats["gaze_horizontal_variance"] = _safe_var(gh)
    feats["gaze_horizontal_entropy"] = shannon_entropy(gh, bins=10, value_range=(0, 1))
    feats["gaze_vertical_mean"] = _safe_mean(gv)
    feats["gaze_vertical_variance"] = _safe_var(gv)
    feats["gaze_vertical_entropy"] = shannon_entropy(gv, bins=10, value_range=(0, 1))
    if gh and gv:
        away = [1 for h, v in zip(gh, gv) if abs(h - 0.5) > 0.2 or abs(v - 0.5) > 0.2]
        feats["gaze_looking_away_proportion"] = len(away) / len(gh)
    else:
        feats["gaze_looking_away_proportion"] = 0.0
    feats["gaze_direction_change_rate"] = _direction_change_rate(gh)

    # Blink / eye openness
    blink_stats = session.compute_blink_stats()
    feats["blink_count"] = blink_stats["blink_count"]
    feats["blink_rate_per_min"] = blink_stats["blink_rate_per_min"]
    feats["blink_duration_mean"] = blink_stats["blink_duration_mean"]
    feats["blink_duration_variance"] = blink_stats["blink_duration_variance"]
    feats["ear_mean"] = _safe_mean(session.ear)
    feats["ear_variance"] = _safe_var(session.ear)

    # Head pose
    yaw, pitch, roll = session.yaw, session.pitch, session.roll
    feats["yaw_mean"] = _safe_mean(yaw)
    feats["yaw_variance"] = _safe_var(yaw)
    feats["pitch_mean"] = _safe_mean(pitch)
    feats["pitch_variance"] = _safe_var(pitch)
    feats["roll_mean"] = _safe_mean(roll)
    feats["roll_variance"] = _safe_var(roll)
    movement_mag = _movement_magnitude_series(yaw, pitch, roll)
    feats["head_movement_magnitude_mean"] = _safe_mean(movement_mag)
    feats["head_movement_magnitude_variance"] = _safe_var(movement_mag)

    # Facial movement
    om, mm, em = session.overall_movement, session.mouth_movement, session.eyebrow_movement
    feats["overall_facial_movement_mean"] = _safe_mean(om)
    feats["overall_facial_movement_variance"] = _safe_var(om)
    feats["mouth_movement_mean"] = _safe_mean(mm)
    feats["mouth_movement_variance"] = _safe_var(mm)
    feats["eyebrow_movement_mean"] = _safe_mean(em)
    feats["eyebrow_movement_variance"] = _safe_var(em)
    if om:
        median_mv = float(np.median(om))
        feats["facial_movement_frequency"] = sum(1 for v in om if v > median_mv) / len(om)
    else:
        feats["facial_movement_frequency"] = 0.0

    # ViT emotions
    emo_samples = session.emotion_scores
    for label in EMOTION_LABELS:
        series = [s.get(label, 0.0) for s in emo_samples]
        feats[f"emotion_{label}_mean"] = _safe_mean(series)
        feats[f"emotion_{label}_variance"] = _safe_var(series)

    if emo_samples:
        entropies = [shannon_entropy(list(s.values()), bins=7, value_range=(0, 1)) for s in emo_samples]
        feats["emotion_distribution_entropy_mean"] = _safe_mean(entropies)

        dominant_labels = [max(s, key=s.get) for s in emo_samples]
        switches = sum(1 for a, b in zip(dominant_labels, dominant_labels[1:]) if a != b)
        feats["dominant_emotion_switch_rate"] = switches / len(dominant_labels) if len(dominant_labels) > 1 else 0.0
    else:
        feats["emotion_distribution_entropy_mean"] = 0.0
        feats["dominant_emotion_switch_rate"] = 0.0

    quality = session.quality_metrics()
    feats["deepface_success_ratio"] = (
        quality["vit_success_ratio"]
        if quality["vit_attempts"]
        else quality["deepface_success_ratio"]
    )
    feats["valid_frame_ratio"] = quality["valid_frame_ratio"]

    return feats


def feature_dict_to_vector(feats: Dict[str, float], feature_order: List[str] = None) -> np.ndarray:
    """Convert the named feature dict into a fixed-order numpy array, using
    the persisted schema order so training/inference can never drift."""
    order = feature_order or load_feature_schema()
    return np.array([float(feats.get(name, 0.0)) for name in order], dtype=np.float64)

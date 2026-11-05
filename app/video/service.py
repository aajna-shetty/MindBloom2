"""
Framework-independent MediaPipe + ViT video analysis service.

The service owns capture, frame processing, emotion aggregation, feature-vector
validation, and optional video-model prediction. FastAPI only validates the
request and maps the returned result to its response schema.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2

from app.video.capture import CameraError, VideoCaptureSession
from app.video.config import (
    PROCESSING_TIMEOUT_SECONDS,
    SESSION_DURATION_SECONDS,
    VIT_EVERY_N_FRAMES,
)
from app.video.features import VIDEO_FEATURES, build_feature_vector
from app.video.mediapipe_processor import MediaPipeProcessor
from app.video.model import get_video_model
from app.video.session import SessionData

logger = logging.getLogger(__name__)


@dataclass
class VideoAnalysisResult:
    status: str
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    video_model_status: str = "video_model_not_available"
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    predicted_class: Optional[int] = None
    session_duration_seconds: Optional[float] = None
    valid_frame_ratio: Optional[float] = None
    feature_count: Optional[int] = None
    detail: Optional[str] = None
    errors: list[dict[str, str]] = field(default_factory=list)
    mediapipe: dict[str, Any] = field(default_factory=dict)
    vit: dict[str, Any] = field(default_factory=dict)
    combined_features: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    prediction: dict[str, Any] = field(default_factory=dict)


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def _emotion_statistics(session: SessionData) -> dict[str, Any]:
    if not session.emotion_scores:
        return {
            "mean": {},
            "variance": {},
            "fear_probability_mean": None,
            "distribution": {},
        }

    labels = sorted({label for scores in session.emotion_scores for label in scores})
    means = {
        label: round(sum(scores.get(label, 0.0) for scores in session.emotion_scores)
                    / len(session.emotion_scores), 4)
        for label in labels
    }
    variances = {
        label: round(
            sum((scores.get(label, 0.0) - means[label]) ** 2
                for scores in session.emotion_scores)
            / len(session.emotion_scores),
            4,
        )
        for label in labels
    }
    dominant = [max(scores, key=scores.get) for scores in session.emotion_scores]
    return {
        "mean": means,
        "variance": variances,
        "fear_probability_mean": means.get("fear"),
        "distribution": {
            label: dominant.count(label) for label in sorted(set(dominant))
        },
    }


def _build_result(
    session: SessionData,
    elapsed: float,
    features: Optional[dict[str, float]],
    status: str,
    detail: Optional[str] = None,
    errors: Optional[list[dict[str, str]]] = None,
) -> VideoAnalysisResult:
    quality = session.quality_metrics()
    blink = session.compute_blink_stats()
    emotion_stats = _emotion_statistics(session)
    mediapipe = {
        "total_frames": quality["total_frames"],
        "captured_frames": quality["captured_frames"],
        "mediapipe_processed_frames": quality["mediapipe_processed_frames"],
        "face_detection_ratio": quality["valid_frame_ratio"],
        "valid_frames": quality["valid_face_frames"],
        "no_face_frames": quality["no_face_frames"],
        "multiple_face_frames": quality["multi_face_frames"],
        "processing_errors": quality["processing_errors"],
        "gaze_features": {
            key: value for key, value in (features or {}).items() if key.startswith("gaze_")
        },
        "blink_features": blink,
        "head_pose_features": {
            key: value for key, value in (features or {}).items()
            if key.startswith(("yaw_", "pitch_", "roll_", "head_movement_"))
        },
        "facial_movement_features": {
            key: value for key, value in (features or {}).items()
            if key.startswith(("overall_facial_", "mouth_", "eyebrow_", "facial_movement_"))
        },
    }
    vit = {
        "frames_analyzed": session.vit_successes,
        "attempts": session.vit_attempts,
        "frames_submitted": quality["vit_submitted_frames"],
        "frames_failed": session.vit_failures,
        "emotion_distribution": emotion_stats["distribution"],
        "fear_probability_mean": emotion_stats["fear_probability_mean"],
        "emotion_statistics": emotion_stats,
    }
    vector_valid = (
        features is not None
        and len(features) == len(VIDEO_FEATURES)
        and all(math.isfinite(float(value)) for value in features.values())
    )
    return VideoAnalysisResult(
        status=status,
        video_model_status="video_model_not_available",
        session_duration_seconds=round(elapsed, 2),
        valid_frame_ratio=quality["valid_frame_ratio"],
        feature_count=len(features) if features is not None else None,
        detail=detail,
        errors=errors or [],
        mediapipe=mediapipe,
        vit=vit,
        combined_features={
            "feature_count": len(features) if features is not None else 0,
            "feature_schema_version": "1.0",
            "valid": vector_valid,
            "features": features or {},
        },
        performance={
            "average_mediapipe_latency_ms": _mean(session.mediapipe_latencies_ms),
            "average_vit_latency_ms": _mean(session.vit_latencies_ms),
        },
        prediction={
            "available": False,
            "predicted_class": None,
            "predicted_label": None,
            "probabilities": None,
        },
    )


def run_video_session(
    source: int | str = 0,
    duration_seconds: int = SESSION_DURATION_SECONDS,
    deepface_every_n_frames: int = VIT_EVERY_N_FRAMES,
    use_vit: bool = False,
) -> VideoAnalysisResult:
    """Run MediaPipe plus sampled ViT analysis for one capture source."""
    del deepface_every_n_frames  # retained for compatibility; DeepFace is not used here
    session = SessionData()
    started = time.monotonic()
    vit_processor = None
    errors: list[dict[str, str]] = []

    try:
        if use_vit:
            from app.video.emotion_processor import EmotionProcessor
            vit_processor = EmotionProcessor()

        with VideoCaptureSession(source=source, duration_seconds=duration_seconds) as cap:
            with MediaPipeProcessor() as mp_proc:
                for frame in cap.frames():
                    if time.monotonic() - started > PROCESSING_TIMEOUT_SECONDS:
                        return _build_result(
                            session, time.monotonic() - started, None,
                            "processing_timeout",
                            "Video processing exceeded the configured timeout.",
                            [{"code": "processing_timeout",
                              "message": "The video processing timeout was exceeded."}],
                        )
                    if not frame.valid or frame.rgb_frame is None:
                        session.record_capture_failure()
                        continue

                    mp_started = time.perf_counter()
                    landmark_result = mp_proc.process_frame(
                        frame.rgb_frame,
                        timestamp_ms=int(frame.timestamp * 1000),
                    )
                    session.mediapipe_latencies_ms.append(
                        (time.perf_counter() - mp_started) * 1000.0
                    )
                    session.record_landmark_result(frame.timestamp, landmark_result)

                    if (
                        vit_processor is not None
                        and frame.index % max(1, VIT_EVERY_N_FRAMES) == 0
                        and landmark_result.face_detected
                    ):
                        session.record_vit_submission()
                        crop = _crop_from_landmarks(
                            frame.rgb_frame,
                            landmark_result.landmarks,
                        )
                        if crop is not None:
                            vit_started = time.perf_counter()
                            vit_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                            vit_result = vit_processor.analyze(vit_crop)
                            session.record_vit_result(
                                vit_result["emotion_scores"] if vit_result["success"] else None,
                                (time.perf_counter() - vit_started) * 1000.0,
                            )

    except CameraError as exc:
        logger.error("Camera error: %s", exc)
        return _build_result(
            session, time.monotonic() - started, None, "error",
            str(exc), [{"code": "camera_unavailable", "message": str(exc)}],
        )
    except ImportError as exc:
        logger.error("Video dependency unavailable: %s", exc)
        return _build_result(
            session, time.monotonic() - started, None, "dependency_unavailable",
            str(exc), [{"code": "dependency_unavailable", "message": str(exc)}],
        )
    except Exception as exc:
        logger.exception("Unexpected video processing failure")
        return _build_result(
            session, time.monotonic() - started, None, "processing_error",
            str(exc), [{"code": "processing_error", "message": str(exc)}],
        )

    elapsed = time.monotonic() - started
    if not session.has_sufficient_data():
        return _build_result(
            session, elapsed, None, "insufficient_video_data",
            "Video did not contain enough valid face frames.",
            [{"code": "insufficient_video_data",
              "message": "Minimum valid-frame ratio and frame count were not met."}],
        )

    features = build_feature_vector(session)
    if len(features) != len(VIDEO_FEATURES) or not all(
        math.isfinite(float(value)) for value in features.values()
    ):
        return _build_result(
            session, elapsed, features, "invalid_feature_vector",
            "The generated feature vector is invalid.",
            [{"code": "invalid_feature_vector", "message": "Expected finite 47-feature vector."}],
        )

    result = _build_result(session, elapsed, features, "video_model_not_available")
    video_model = get_video_model()
    if not video_model.is_available():
        return result

    prediction = video_model.predict(features)
    if prediction is None:
        return result

    result.status = "success"
    result.video_model_status = "available"
    result.video_score = round(prediction.video_score, 4)
    result.video_confidence = round(prediction.video_confidence, 4)
    result.predicted_class = prediction.predicted_class
    result.prediction = {
        "available": True,
        "predicted_class": prediction.predicted_class,
        "predicted_label": "asd" if prediction.predicted_class == 1 else "non_asd",
        "probabilities": {
            "asd": result.video_score,
            "non_asd": round(1.0 - result.video_score, 4),
        },
    }
    return result


def _crop_from_landmarks(
    rgb_frame: Any,
    landmarks: Any,
) -> Optional[Any]:
    """Create a padded RGB crop from MediaPipe pixel landmarks."""
    if landmarks is None or len(landmarks) == 0:
        return None
    height, width = rgb_frame.shape[:2]
    x_min = max(0, int(landmarks[:, 0].min()))
    y_min = max(0, int(landmarks[:, 1].min()))
    x_max = min(width, int(landmarks[:, 0].max()))
    y_max = min(height, int(landmarks[:, 1].max()))
    pad_x = int((x_max - x_min) * 0.18)
    pad_y = int((y_max - y_min) * 0.22)
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(width, x_max + pad_x)
    y_max = min(height, y_max + pad_y)
    if x_max <= x_min or y_max <= y_min:
        return None
    return rgb_frame[y_min:y_max, x_min:x_max].copy()

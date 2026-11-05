"""
app/video/session.py
─────────────────────────────────────────────────────────────────────────────
accumulates per-frame measurements from MediaPipeProcessor and sampled ViT
inference across an entire session, applies §11 data-quality
handling, and does session-level blink counting (which needs visibility
across consecutive frames, unlike the other per-frame features).

No ASD prediction happens here — this is pure time-series bookkeeping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np

from app.video.config import (
    MIN_VALID_FRAME_RATIO, MIN_VALID_FRAME_COUNT, BLINK_MIN_CONSEC_FRAMES,
    EMOTION_LABELS,
)
from app.video.mediapipe_processor import LandmarkFrameFeatures

logger = logging.getLogger(__name__)


@dataclass
class SessionData:
    """Raw time series collected over the session. Mirrors §10 of the spec."""

    timestamps: List[float] = field(default_factory=list)

    gaze_horizontal: List[float] = field(default_factory=list)
    gaze_vertical: List[float] = field(default_factory=list)

    ear: List[float] = field(default_factory=list)
    eye_closed_frames: List[bool] = field(default_factory=list)  # per-frame, pre-blink-filter

    yaw: List[float] = field(default_factory=list)
    pitch: List[float] = field(default_factory=list)
    roll: List[float] = field(default_factory=list)

    overall_movement: List[float] = field(default_factory=list)
    mouth_movement: List[float] = field(default_factory=list)
    eyebrow_movement: List[float] = field(default_factory=list)

    # One dict per successful ViT sample (not per frame — sparse).
    emotion_scores: List[Dict[str, float]] = field(default_factory=list)

    # Quality counters (§11)
    total_frames: int = 0
    captured_frames: int = 0
    mediapipe_processed_frames: int = 0
    valid_face_frames: int = 0        # exactly-one-face-detected frames used for features
    invalid_frames: int = 0           # capture failures
    no_face_frames: int = 0
    multi_face_frames: int = 0
    deepface_attempts: int = 0
    deepface_successes: int = 0
    deepface_failures: int = 0
    vit_attempts: int = 0
    vit_submitted_frames: int = 0
    vit_successes: int = 0
    vit_failures: int = 0
    vit_latencies_ms: List[float] = field(default_factory=list)
    mediapipe_latencies_ms: List[float] = field(default_factory=list)
    processing_errors: int = 0

    def record_capture_failure(self) -> None:
        self.captured_frames += 1
        self.total_frames += 1
        self.invalid_frames += 1

    def record_deepface_result(self, result: Any) -> None:
        """Retain compatibility for the explicitly legacy test harness."""
        self.deepface_attempts += 1
        if result.success and result.emotion_scores:
            self.deepface_successes += 1
            self.emotion_scores.append(result.emotion_scores)
        else:
            self.deepface_failures += 1

    def record_landmark_result(self, timestamp: float, result: LandmarkFrameFeatures) -> None:
        self.captured_frames += 1
        self.mediapipe_processed_frames += 1
        self.total_frames += 1

        if not result.face_detected:
            self.no_face_frames += 1
            return
        if result.num_faces > 1:
            self.multi_face_frames += 1
            # We still use the primary/largest face's features (documented
            # in mediapipe_processor.py) rather than discarding the frame
            # outright, but we track how often this happened for QA.

        self.valid_face_frames += 1
        self.timestamps.append(timestamp)

        if result.gaze_horizontal is not None and result.gaze_vertical is not None:
            self.gaze_horizontal.append(result.gaze_horizontal)
            self.gaze_vertical.append(result.gaze_vertical)

        if result.ear is not None:
            self.ear.append(result.ear)
            self.eye_closed_frames.append(bool(result.eye_closed))

        if result.yaw is not None:
            self.yaw.append(result.yaw)
            self.pitch.append(result.pitch)
            self.roll.append(result.roll)

        if result.overall_movement is not None:
            self.overall_movement.append(result.overall_movement)
            self.mouth_movement.append(result.mouth_movement)
            self.eyebrow_movement.append(result.eyebrow_movement)

    def record_vit_result(
        self,
        emotion_scores: Optional[Dict[str, float]],
        latency_ms: float,
    ) -> None:
        self.vit_attempts += 1
        self.vit_latencies_ms.append(float(latency_ms))
        if emotion_scores:
            self.vit_successes += 1
            self.emotion_scores.append(dict(emotion_scores))
        else:
            self.vit_failures += 1

    def record_vit_submission(self) -> None:
        self.vit_submitted_frames += 1

    # -- blink counting (needs the full eye_closed_frames sequence) --------

    def compute_blink_stats(self) -> Dict[str, float]:
        """
        Counts a blink as a run of >= BLINK_MIN_CONSEC_FRAMES consecutive
        "eye closed" frames (per-frame EAR threshold from
        mediapipe_processor.py), then bounded by open frames on both sides
        (or by session start/end). Duration is measured in frames converted
        to seconds using the average inter-frame interval actually observed
        (not an assumed fixed FPS), which is a bit more robust to camera
        stutter than assuming a constant frame rate.

        This is an engineering blink detector, not a clinically validated
        instrument.
        """
        n = len(self.eye_closed_frames)
        if n == 0:
            return {
                "blink_count": 0.0, "blink_rate_per_min": 0.0,
                "blink_duration_mean": 0.0, "blink_duration_variance": 0.0,
            }

        if len(self.timestamps) >= 2:
            span = self.timestamps[-1] - self.timestamps[0]
            avg_interval = span / max(1, len(self.timestamps) - 1)
        else:
            avg_interval = 1.0 / 30.0  # fallback assumption, documented

        blink_durations = []
        run_len = 0
        for closed in self.eye_closed_frames:
            if closed:
                run_len += 1
            else:
                if run_len >= BLINK_MIN_CONSEC_FRAMES:
                    blink_durations.append(run_len * avg_interval)
                run_len = 0
        if run_len >= BLINK_MIN_CONSEC_FRAMES:
            blink_durations.append(run_len * avg_interval)

        blink_count = len(blink_durations)
        total_duration_s = n * avg_interval
        blink_rate_per_min = (blink_count / total_duration_s) * 60.0 if total_duration_s > 0 else 0.0

        if blink_durations:
            mean_dur = float(np.mean(blink_durations)) if blink_durations else 0.0
            var_dur = float(np.var(blink_durations)) if len(blink_durations) > 1 else 0.0
        else:
            mean_dur, var_dur = 0.0, 0.0

        return {
            "blink_count": float(blink_count),
            "blink_rate_per_min": float(blink_rate_per_min),
            "blink_duration_mean": mean_dur,
            "blink_duration_variance": var_dur,
        }

    # -- quality summary (§11) ---------------------------------------------

    def valid_frame_ratio(self) -> float:
        if self.total_frames == 0:
            return 0.0
        return self.valid_face_frames / self.total_frames

    def quality_metrics(self) -> Dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "captured_frames": self.captured_frames,
            "mediapipe_processed_frames": self.mediapipe_processed_frames,
            "valid_face_frames": self.valid_face_frames,
            "invalid_frames": self.invalid_frames,
            "no_face_frames": self.no_face_frames,
            "multi_face_frames": self.multi_face_frames,
            "valid_frame_ratio": round(self.valid_frame_ratio(), 4),
            "deepface_attempts": self.deepface_attempts,
            "deepface_successes": self.deepface_successes,
            "deepface_failures": self.deepface_failures,
            "deepface_success_ratio": round(
                self.deepface_successes / self.deepface_attempts, 4
            ) if self.deepface_attempts else 0.0,
            "vit_attempts": self.vit_attempts,
            "vit_submitted_frames": self.vit_submitted_frames,
            "vit_successes": self.vit_successes,
            "vit_failures": self.vit_failures,
            "vit_success_ratio": round(
                self.vit_successes / self.vit_attempts, 4
            ) if self.vit_attempts else 0.0,
            "processing_errors": self.processing_errors,
        }

    def has_sufficient_data(self) -> bool:
        return (
            self.valid_frame_ratio() >= MIN_VALID_FRAME_RATIO
            and self.valid_face_frames >= MIN_VALID_FRAME_COUNT
        )

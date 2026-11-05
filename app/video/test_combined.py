"""
Local MediaPipe + ViT webcam integration test.

Run with:

    python -m app.video.test_combined

Frames are processed in memory only. This harness does not invoke the
questionnaire pipeline, a video classifier, or any fusion logic.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.video.capture import CameraError, VideoCaptureSession
from app.video.config import BLINK_MIN_CONSEC_FRAMES
from app.video.emotion_processor import EmotionProcessor
from app.video.mediapipe_processor import LandmarkFrameFeatures, MediaPipeProcessor
from app.video.service import _crop_from_landmarks
from app.video.test_mediapipe import _draw_landmarks, _load_connections


WINDOW_NAME = "MindBloom - Combined MediaPipe + ViT Test"
TEXT_COLOR = (255, 255, 255)
FEATURE_COLOR = (255, 180, 0)
ERROR_COLOR = (0, 0, 255)


@dataclass(frozen=True)
class PredictionRecord:
    frame_number: int
    timestamp: float
    predicted_emotion: str
    probabilities: dict[str, float]
    inference_latency_ms: float


@dataclass
class WorkerStats:
    submitted: int = 0
    completed: int = 0
    dropped: int = 0
    failures: int = 0
    errors: list[str] = field(default_factory=list)


class VitWorker:
    """Keep the display loop responsive while ViT processes sampled frames."""

    def __init__(self, processor: EmotionProcessor, sample_interval: int):
        self.processor = processor
        self.sample_interval = max(1, sample_interval)
        self._queue: "queue.Queue[tuple[int, float, np.ndarray]]" = queue.Queue(maxsize=1)
        self._records: list[PredictionRecord] = []
        self._latest: Optional[PredictionRecord] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stats = WorkerStats()
        self._last_submitted_frame: Optional[int] = None
        self._last_completed_frame: Optional[int] = None

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame_number: int, timestamp: float, face_crop_bgr: np.ndarray) -> None:
        if frame_number % self.sample_interval:
            return
        self.stats.submitted += 1
        self._last_submitted_frame = frame_number
        try:
            self._queue.put_nowait((frame_number, timestamp, face_crop_bgr.copy()))
        except queue.Full:
            self.stats.dropped += 1

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                frame_number, timestamp, crop = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                result = self.processor.analyze(crop)
                latency_ms = (time.perf_counter() - started) * 1000.0
                if result["success"]:
                    record = PredictionRecord(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        predicted_emotion=result["dominant_emotion"],
                        probabilities=dict(result["emotion_scores"]),
                        inference_latency_ms=latency_ms,
                    )
                    with self._lock:
                        self._records.append(record)
                        self._latest = record
                    self.stats.completed += 1
                    self._last_completed_frame = frame_number
                else:
                    self.stats.failures += 1
                    message = result.get("error") or "inference_failed"
                    self.stats.errors.append(f"frame {frame_number}: {message}")
                    logging.warning("ViT failed on frame %s: %s", frame_number, message)
            except Exception as exc:
                self.stats.failures += 1
                self.stats.errors.append(f"frame {frame_number}: {exc}")
                logging.exception("ViT worker failed on frame %s", frame_number)
            finally:
                self._queue.task_done()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            self.stats.errors.append("worker_shutdown_timeout")

    def snapshot(self) -> Optional[PredictionRecord]:
        with self._lock:
            return self._latest

    @property
    def records(self) -> list[PredictionRecord]:
        with self._lock:
            return list(self._records)


def _text(frame: np.ndarray, value: str, y: int, color=TEXT_COLOR) -> None:
    cv2.putText(frame, value, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, value, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                color, 1, cv2.LINE_AA)


def _draw_overlay(
    frame: np.ndarray,
    result: LandmarkFrameFeatures,
    prediction: Optional[PredictionRecord],
    fps: float,
    mediapipe_ms: float,
    frame_number: int,
    blink_count: int,
) -> None:
    _text(frame, f"Frame: {frame_number}  FPS: {fps:.1f}", 25)
    status = "YES" if result.face_detected else "NO"
    _text(frame, f"Face: {status}  Faces: {result.num_faces}", 49,
          (0, 255, 0) if result.face_detected else ERROR_COLOR)
    if not result.face_detected:
        _text(frame, "Face not detected - live camera still active", 73, ERROR_COLOR)
    _text(frame, f"MediaPipe: {mediapipe_ms:.1f} ms", 97)
    _text(frame, f"Gaze H/V: {result.gaze_horizontal if result.gaze_horizontal is not None else 'N/A'} / "
          f"{result.gaze_vertical if result.gaze_vertical is not None else 'N/A'}", 121, FEATURE_COLOR)
    _text(frame, f"EAR: {result.ear if result.ear is not None else 'N/A'}  "
          f"Blink: {'YES' if result.eye_closed else 'NO' if result.eye_closed is not None else 'N/A'}  "
          f"Count: {blink_count}",
          145, FEATURE_COLOR)
    _text(frame, f"Pose P/Y/R: {result.pitch if result.pitch is not None else 'N/A'} / "
          f"{result.yaw if result.yaw is not None else 'N/A'} / "
          f"{result.roll if result.roll is not None else 'N/A'}", 169, FEATURE_COLOR)
    _text(frame, f"Movement O/M/E: "
          f"{result.overall_movement if result.overall_movement is not None else 'N/A'} / "
          f"{result.mouth_movement if result.mouth_movement is not None else 'N/A'} / "
          f"{result.eyebrow_movement if result.eyebrow_movement is not None else 'N/A'}",
          193, FEATURE_COLOR)

    if prediction is None:
        _text(frame, "ViT: waiting for prediction", 217, (0, 220, 255))
    else:
        confidence = prediction.probabilities[prediction.predicted_emotion]
        stale = frame_number - prediction.frame_number
        _text(frame, f"Emotion: {prediction.predicted_emotion.upper()} "
              f"{confidence * 100:.1f}%  ViT: {prediction.inference_latency_ms:.1f} ms",
              217, (0, 255, 0))
        _text(frame, f"ViT frame: {prediction.frame_number}  "
              f"{'STALE by ' + str(stale) + ' frames' if stale else 'CURRENT'}",
              241, (0, 220, 255) if stale else (0, 255, 0))
        _text(frame, f"Fear: {prediction.probabilities.get('fear', 0.0) * 100:.1f}%",
              265, (0, 165, 255))
        y = 289
        for label, probability in sorted(prediction.probabilities.items(),
                                         key=lambda item: item[1], reverse=True):
            _text(frame, f"{label:<8} {probability * 100:5.1f}%", y,
                  (0, 220, 255) if label == prediction.predicted_emotion else TEXT_COLOR)
            y += 22
    _text(frame, "Q / ESC: exit", frame.shape[0] - 18)


def _create_display_window(width: int, height: int) -> None:
    """Create a resizable window before the first camera frame is rendered."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if width > 0 and height > 0:
        cv2.resizeWindow(WINDOW_NAME, width, height)


def _print_summary(records: list[PredictionRecord], captured: int,
                   mp_processed: int, valid_faces: int, elapsed: float,
                   mp_latencies: list[float], worker: Optional[VitWorker],
                   processing_errors: list[str]) -> None:
    print("\n" + "=" * 64)
    print("COMBINED MEDIAPIPE + VIT WEBCAM TEST SUMMARY")
    print("=" * 64)
    print(f"Captured frames: {captured}")
    print(f"MediaPipe-processed frames: {mp_processed}")
    print(f"Valid face frames: {valid_faces}")
    print(f"Face detection ratio: {valid_faces / max(captured, 1):.4f}")
    print(f"ViT-analyzed frames: {len(records)}")
    worker_stats = worker.stats if worker is not None else WorkerStats()
    print(f"ViT submitted: {worker_stats.submitted}")
    print(f"ViT dropped/skipped: {worker_stats.dropped}")
    print(f"Processing errors: {len(processing_errors) + worker_stats.failures}")
    print(f"Approximate webcam FPS: {captured / max(elapsed, 1e-9):.2f}")
    if mp_latencies:
        print(f"Average MediaPipe latency: {np.mean(mp_latencies):.2f} ms")
    if records:
        latencies = [record.inference_latency_ms for record in records]
        counts = Counter(record.predicted_emotion for record in records)
        probabilities = {
            label: [record.probabilities.get(label, 0.0) for record in records]
            for label in EmotionProcessor.STANDARD_EMOTIONS
        }
        print(f"Average ViT latency: {np.mean(latencies):.2f} ms")
        print(f"Minimum ViT latency: {min(latencies):.2f} ms")
        print(f"Maximum ViT latency: {max(latencies):.2f} ms")
        print(f"Fear probability mean: {np.mean(probabilities['fear']):.4f}")
        print(f"Fear probability variance: {np.var(probabilities['fear']):.4f}")
        print("Emotion distribution:", dict(counts))
    else:
        print("No successful ViT predictions were produced.")
    if worker_stats.errors or processing_errors:
        print("Errors:")
        for error in processing_errors + worker_stats.errors:
            print(f"  {error}")
    print("No ASD prediction was produced; video classifier remains unavailable.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the combined webcam test.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=int, default=3600)
    parser.add_argument("--sample-interval", type=int, default=5)
    args = parser.parse_args()

    connections = _load_connections()
    captured = 0
    mp_processed = 0
    valid_faces = 0
    displayed_frames = 0
    blink_count = 0
    closed_run = 0
    processing_errors: list[str] = []
    mp_latencies: list[float] = []
    started = time.perf_counter()
    previous = started
    worker: Optional[VitWorker] = None

    print("Starting combined MediaPipe + ViT webcam test.")
    print("Press Q or ESC in the camera window to exit.")
    try:
        with VideoCaptureSession(source=args.camera, duration_seconds=args.duration) as camera:
            camera_width = getattr(camera, "width", 0)
            camera_height = getattr(camera, "height", 0)
            _create_display_window(camera_width, camera_height)
            with MediaPipeProcessor(max_num_faces=2, refine_landmarks=True) as mp_processor:
                worker = VitWorker(EmotionProcessor(), args.sample_interval)
                worker.start()
                for frame in camera.frames():
                    now = time.perf_counter()
                    captured += 1
                    if not frame.valid or frame.rgb_frame is None:
                        processing_errors.append(
                            f"frame {frame.index}: {frame.error or 'invalid_frame'}"
                        )
                        continue
                    mp_processed += 1
                    display = cv2.cvtColor(frame.rgb_frame, cv2.COLOR_RGB2BGR)
                    processing_started = time.perf_counter()
                    landmark_result = mp_processor.process_frame(
                        frame.rgb_frame, timestamp_ms=int(frame.timestamp * 1000)
                    )
                    mediapipe_ms = (time.perf_counter() - processing_started) * 1000.0
                    mp_latencies.append(mediapipe_ms)
                    if landmark_result.face_detected:
                        valid_faces += 1
                        if landmark_result.eye_closed:
                            closed_run += 1
                        elif closed_run >= BLINK_MIN_CONSEC_FRAMES:
                            blink_count += 1
                            closed_run = 0
                        else:
                            closed_run = 0
                        crop = _crop_from_landmarks(
                            frame.rgb_frame, landmark_result.landmarks
                        )
                        if crop is not None:
                            worker.submit(
                                frame.index, frame.timestamp,
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
                            )
                    _draw_landmarks(display, landmark_result.landmarks, connections)
                    fps = 1.0 / max(now - previous, 1e-9)
                    previous = now
                    _draw_overlay(display, landmark_result, worker.snapshot(),
                                  fps, mediapipe_ms, frame.index, blink_count)
                    if (
                        camera_width > 0
                        and camera_height > 0
                        and (
                            display.shape[1] != camera_width
                            or display.shape[0] != camera_height
                        )
                    ):
                        logging.warning(
                            "Camera frame dimensions changed from %sx%s to %sx%s",
                            camera_width,
                            camera_height,
                            display.shape[1],
                            display.shape[0],
                        )
                    cv2.imshow(WINDOW_NAME, display)
                    displayed_frames += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
    except CameraError as exc:
        print(f"Webcam access failed: {exc}")
        return 1
    except Exception as exc:
        logging.exception("Combined webcam test failed")
        print(f"Combined test failed: {exc}")
        return 1
    finally:
        if worker is not None:
            worker.stop()
        cv2.destroyAllWindows()

    if closed_run >= BLINK_MIN_CONSEC_FRAMES:
        blink_count += 1
    _print_summary(worker.records if worker is not None else [], captured,
                   mp_processed, valid_faces, time.perf_counter() - started,
                   mp_latencies, worker,
                   processing_errors)
    print(f"Frames displayed: {displayed_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

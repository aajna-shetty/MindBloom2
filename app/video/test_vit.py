"""
Independent local ViT emotion webcam test.

Run with:

    python -m app.video.test_vit

The harness uses the existing EmotionProcessor only. It does not import or
invoke MediaPipe, DeepFace, questionnaire scoring, or video-risk prediction.
Frames and prediction records remain in memory and are never written.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

# Do not permit a missing cached model to trigger a network download.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.video.capture import CameraError, VideoCaptureSession
from app.video.emotion_processor import EmotionProcessor


WINDOW_NAME = "MindBloom - Independent ViT Emotion Test"


@dataclass
class PredictionRecord:
    frame_number: int
    timestamp: float
    predicted_emotion: str
    probabilities: Dict[str, float]
    inference_latency_ms: float


class VitWorker:
    def __init__(self, processor: EmotionProcessor, sample_interval: int):
        self.processor = processor
        self.sample_interval = max(1, sample_interval)
        self.frames: "queue.Queue[tuple[int, float, np.ndarray]]" = queue.Queue(maxsize=1)
        self.records: list[PredictionRecord] = []
        self.latest: Optional[PredictionRecord] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame_number: int, timestamp: float, frame: np.ndarray) -> None:
        if frame_number % self.sample_interval != 0:
            return
        try:
            self.frames.put_nowait((frame_number, timestamp, frame.copy()))
        except queue.Full:
            pass

    def _run(self) -> None:
        while not self._stop.is_set() or not self.frames.empty():
            try:
                frame_number, timestamp, frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue

            started = time.perf_counter()
            result = self.processor.analyze(frame)
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
                    self.records.append(record)
                    self.latest = record
            else:
                logging.warning(
                    "ViT inference failed for frame %s: %s",
                    frame_number,
                    result.get("error"),
                )
            self.frames.task_done()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def snapshot(self) -> Optional[PredictionRecord]:
        with self._lock:
            return self.latest


def _draw_prediction(frame: np.ndarray, prediction: Optional[PredictionRecord]) -> None:
    if prediction is None:
        cv2.putText(
            frame,
            "ViT: waiting for prediction",
            (18, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return

    dominant = prediction.predicted_emotion
    confidence = prediction.probabilities.get(dominant, 0.0)
    ranked = sorted(
        prediction.probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    top_three = ", ".join(
        f"{emotion} {probability * 100:.1f}%"
        for emotion, probability in ranked[:3]
    )
    cv2.putText(
        frame,
        f"Emotion: {dominant.upper()}  Confidence: {confidence * 100:.1f}%",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ViT latency: {prediction.inference_latency_ms:.1f} ms  "
        f"Analyzed frame: {prediction.frame_number}",
        (18, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Fear probability: {prediction.probabilities.get('fear', 0.0) * 100:.1f}%",
        (18, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 165, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Top 3: {top_three}",
        (18, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    y = 140
    for emotion, probability in ranked:
        color = (0, 220, 255) if emotion == dominant else (235, 235, 235)
        text = f"{emotion:<8} {probability * 100:5.1f}%"
        cv2.putText(
            frame,
            text,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            2 if emotion == dominant else 1,
            cv2.LINE_AA,
        )
        y += 24

    cv2.putText(
        frame,
        "Q / ESC: exit",
        (18, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _detect_face_crop(
    frame_bgr: np.ndarray,
    detector: cv2.CascadeClassifier,
) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int, int]]]:
    """Return a padded BGR face crop using OpenCV only."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )
    if len(faces) == 0:
        return None, None

    x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
    padding_x = int(width * 0.18)
    padding_y = int(height * 0.22)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(frame_bgr.shape[1], x + width + padding_x)
    y2 = min(frame_bgr.shape[0], y + height + padding_y)
    return frame_bgr[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def _print_summary(
    worker: VitWorker,
    captured_frames: int,
    elapsed_seconds: float,
) -> None:
    records = worker.records
    latencies = [record.inference_latency_ms for record in records]
    counts = Counter(record.predicted_emotion for record in records)

    print("\n" + "=" * 60)
    print("INDEPENDENT VIT WEBCAM TEST SUMMARY")
    print("=" * 60)
    print(f"Captured frames: {captured_frames}")
    print(f"Analyzed frames: {len(records)}")
    print(f"Approximate webcam FPS: {captured_frames / max(elapsed_seconds, 1e-9):.2f}")
    if latencies:
        print(f"Average ViT inference latency: {np.mean(latencies):.2f} ms")
        print(f"Minimum ViT inference latency: {min(latencies):.2f} ms")
        print(f"Maximum ViT inference latency: {max(latencies):.2f} ms")
        print("Emotion prediction distribution:")
        for emotion, count in counts.most_common():
            print(f"  {emotion}: {count}")
        print("Sample predictions:")
        for record in records[:5]:
            print(
                f"  frame={record.frame_number} "
                f"time={record.timestamp:.2f}s "
                f"emotion={record.predicted_emotion} "
                f"confidence={record.probabilities[record.predicted_emotion]:.4f}"
            )
    else:
        print("No successful ViT predictions were produced.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an independent ViT webcam test.")
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=int(os.environ.get("VIT_SAMPLE_INTERVAL", "5")),
        help="Analyze every Nth captured frame (default: 5).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=int(os.environ.get("VIT_TEST_DURATION_SECONDS", "300")),
        help="Maximum session duration in seconds (default: 300).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Loading the existing Vision Transformer processor in offline mode.")
    print(f"Model: mo-thecreator/vit-Facial-Expression-Recognition")
    print(f"Sampling interval: every {max(1, args.sample_interval)} frame(s)")

    try:
        processor = EmotionProcessor()
    except Exception as exc:
        print(f"ViT model could not be loaded from the local cache: {exc}", file=sys.stderr)
        return 1

    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        print(f"OpenCV face detector could not be loaded: {cascade_path}", file=sys.stderr)
        return 1

    worker = VitWorker(processor, args.sample_interval)
    captured_frames = 0
    detected_face_frames = 0
    started_at = time.perf_counter()

    try:
        worker.start()
        with VideoCaptureSession(source=0, duration_seconds=max(1, args.duration)) as camera:
            frames = camera.frames()
            previous_frame_at = time.perf_counter()
            smoothed_fps = 0.0
            while True:
                captured = next(frames, None)
                if captured is None:
                    break
                if not captured.valid or captured.rgb_frame is None:
                    continue

                captured_frames += 1
                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_frame_at, 1e-9)
                smoothed_fps = instantaneous_fps if smoothed_fps == 0 else (
                    0.9 * smoothed_fps + 0.1 * instantaneous_fps
                )
                previous_frame_at = now

                frame_bgr = cv2.cvtColor(captured.rgb_frame, cv2.COLOR_RGB2BGR)
                face_crop, face_box = _detect_face_crop(frame_bgr, face_detector)
                if face_crop is not None:
                    detected_face_frames += 1
                    worker.submit(captured_frames, captured.timestamp, face_crop)

                display = cv2.flip(frame_bgr, 1)
                if face_box is not None:
                    x1, y1, x2, y2 = face_box
                    width = display.shape[1]
                    cv2.rectangle(
                        display,
                        (width - x2, y1),
                        (width - x1, y2),
                        (0, 255, 120),
                        2,
                    )
                    cv2.putText(
                        display,
                        "Face crop: YES",
                        (18, 184),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 120),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        display,
                        "Face crop: NO",
                        (18, 184),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                _draw_prediction(display, worker.snapshot())
                cv2.putText(
                    display,
                    f"FPS: {smoothed_fps:.1f}  Captured frame: {captured_frames}",
                    (18, display.shape[0] - 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(WINDOW_NAME, display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    finally:
        worker.stop()
        cv2.destroyAllWindows()

    _print_summary(worker, captured_frames, time.perf_counter() - started_at)
    print(f"Frames with OpenCV face detected: {detected_face_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import time
import sys
import inspect
from unittest.mock import patch

import numpy as np

from app.video.capture import FrameResult
from app.video.mediapipe_processor import LandmarkFrameFeatures
from app.video.service import _crop_from_landmarks
from app.video import test_combined
from app.video.test_combined import PredictionRecord, VitWorker


class _FakeEmotionProcessor:
    def analyze(self, image):
        assert image.shape == (20, 20, 3)
        return {
            "success": True,
            "dominant_emotion": "fear",
            "emotion_scores": {
                "angry": 0.01,
                "disgust": 0.01,
                "fear": 0.7,
                "happy": 0.05,
                "sad": 0.1,
                "surprise": 0.1,
                "neutral": 0.03,
            },
        }


def test_landmark_crop_is_nonempty_and_preserves_rgb_dimensions():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    landmarks = np.array([[30, 30], [50, 50], [70, 40]], dtype=np.float32)

    crop = _crop_from_landmarks(frame, landmarks)

    assert crop is not None
    assert crop.ndim == 3
    assert crop.shape[2] == 3
    assert crop.shape[0] > 0
    assert crop.shape[1] > 0


def test_vit_worker_preserves_frame_number_and_timestamp():
    worker = VitWorker(_FakeEmotionProcessor(), sample_interval=1)
    worker.start()
    worker.submit(17, 1.25, np.zeros((20, 20, 3), dtype=np.uint8))
    deadline = time.time() + 2
    while not worker.records and time.time() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert len(worker.records) == 1
    record = worker.records[0]
    assert isinstance(record, PredictionRecord)
    assert record.frame_number == 17
    assert record.timestamp == 1.25
    assert record.predicted_emotion == "fear"
    assert record.probabilities["fear"] == 0.7


def test_vit_worker_skips_unsampled_frames():
    worker = VitWorker(_FakeEmotionProcessor(), sample_interval=5)
    worker.start()
    worker.submit(3, 0.1, np.zeros((20, 20, 3), dtype=np.uint8))
    time.sleep(0.05)
    worker.stop()

    assert worker.records == []


def test_combined_pipeline_has_no_deepface_dependency():
    source = inspect.getsource(test_combined)

    assert "DeepFaceProcessor" not in source
    assert "deepface_processor" not in source
    assert "DeepFace" not in source


def test_vit_worker_survives_inference_exception_and_reports_it():
    class _FailingProcessor:
        def analyze(self, _image):
            raise RuntimeError("synthetic model failure")

    worker = VitWorker(_FailingProcessor(), sample_interval=1)
    worker.start()
    worker.submit(4, 0.25, np.zeros((20, 20, 3), dtype=np.uint8))
    deadline = time.time() + 2
    while not worker.stats.errors and time.time() < deadline:
        time.sleep(0.01)
    worker.stop()

    assert worker.stats.failures == 1
    assert "synthetic model failure" in worker.stats.errors[0]
    assert not worker._thread.is_alive()


def test_main_uses_one_capture_for_mediapipe_and_vit(monkeypatch):
    source_calls = []
    mediapipe_frames = []
    crop_frames = []
    submitted_frames = []
    rendered_frames = []

    class _Capture:
        def __init__(self, source, duration_seconds):
            source_calls.append(source)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def frames(self):
            for index in range(2):
                yield FrameResult(
                    index=index,
                    timestamp=index / 30.0,
                    rgb_frame=np.zeros((80, 80, 3), dtype=np.uint8),
                )

    class _MediaPipe:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def process_frame(self, frame, timestamp_ms):
            mediapipe_frames.append((id(frame), timestamp_ms))
            return LandmarkFrameFeatures(
                face_detected=True,
                num_faces=1,
                landmarks=np.array([[20, 20], [60, 60], [40, 40]], dtype=np.float32),
                gaze_horizontal=0.5,
                gaze_vertical=0.5,
                ear=0.3,
                eye_closed=False,
                yaw=1.0,
                pitch=2.0,
                roll=3.0,
                overall_movement=0.1,
                mouth_movement=0.2,
                eyebrow_movement=0.3,
            )

    class _Emotion:
        STANDARD_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

        def analyze(self, frame):
            submitted_frames.append(frame.shape)
            return {
                "success": True,
                "dominant_emotion": "happy",
                "emotion_scores": {label: (1.0 if label == "happy" else 0.0)
                                   for label in self.STANDARD_EMOTIONS},
            }

    monkeypatch.setattr(test_combined, "VideoCaptureSession", _Capture)
    monkeypatch.setattr(test_combined, "MediaPipeProcessor", _MediaPipe)
    monkeypatch.setattr(test_combined, "EmotionProcessor", _Emotion)
    monkeypatch.setattr(test_combined, "_load_connections", lambda: ())
    monkeypatch.setattr(test_combined, "_draw_landmarks", lambda *args: rendered_frames.append(args[1]))
    monkeypatch.setattr(test_combined.cv2, "namedWindow", lambda *_args: None)
    monkeypatch.setattr(test_combined.cv2, "resizeWindow", lambda *_args: None)
    monkeypatch.setattr(test_combined.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(test_combined.cv2, "waitKey", lambda *_args: ord("q"))
    monkeypatch.setattr(sys, "argv", ["test_combined", "--sample-interval", "1"])

    def crop_from_same_frame(frame, _landmarks):
        crop_frames.append(id(frame))
        return np.zeros((20, 20, 3), dtype=np.uint8)

    with patch.object(test_combined, "_crop_from_landmarks",
                      side_effect=crop_from_same_frame) as crop:
        result = test_combined.main()

    assert result == 0
    assert source_calls == [0]
    assert len(mediapipe_frames) == 1
    assert len(rendered_frames) == 1
    crop.assert_called_once()
    assert crop_frames == [mediapipe_frames[0][0]]
    assert submitted_frames

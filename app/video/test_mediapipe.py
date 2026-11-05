"""
Local MediaPipe webcam visual test.

Run with:

    python -m app.video.test_mediapipe

This harness intentionally exercises only OpenCV capture and the existing
MediaPipe Face Landmarker processor. Frames are processed in memory and are
never written or uploaded.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from app.video.capture import CameraError, VideoCaptureSession
from app.video.mediapipe_processor import LandmarkFrameFeatures, MediaPipeProcessor


WINDOW_NAME = "MindBloom - MediaPipe Webcam Test"
LANDMARK_COLOR = (0, 255, 0)
MESH_COLOR = (80, 180, 255)
FEATURE_COLOR = (255, 180, 0)
TEXT_COLOR = (255, 255, 255)
ERROR_COLOR = (0, 0, 255)


def _load_connections() -> Tuple[Tuple[int, int], ...]:
    """Load MediaPipe's standard topology when the legacy module is available."""
    try:
        from mediapipe.python.solutions.face_mesh_connections import (
            FACEMESH_CONTOURS,
            FACEMESH_IRISES,
            FACEMESH_TESSELATION,
        )
    except ImportError:
        return ()

    connections = set(FACEMESH_TESSELATION)
    connections.update(FACEMESH_CONTOURS)
    connections.update(FACEMESH_IRISES)
    return tuple(connections)


def _draw_delaunay_mesh(frame: np.ndarray, points: list[Tuple[int, int]]) -> None:
    """Draw a mesh from the detected landmarks without requiring legacy MediaPipe APIs."""
    if len(points) < 3:
        return

    height, width = frame.shape[:2]
    subdiv = cv2.Subdiv2D((0, 0, width, height))
    for point in points:
        subdiv.insert(point)

    triangles = subdiv.getTriangleList()
    for triangle in triangles:
        coords = [
            (int(round(triangle[0])), int(round(triangle[1]))),
            (int(round(triangle[2])), int(round(triangle[3]))),
            (int(round(triangle[4])), int(round(triangle[5]))),
        ]
        if all(0 <= x < width and 0 <= y < height for x, y in coords):
            cv2.line(frame, coords[0], coords[1], MESH_COLOR, 1, cv2.LINE_AA)
            cv2.line(frame, coords[1], coords[2], MESH_COLOR, 1, cv2.LINE_AA)
            cv2.line(frame, coords[2], coords[0], MESH_COLOR, 1, cv2.LINE_AA)


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    connections: Iterable[Tuple[int, int]],
) -> None:
    if landmarks is None:
        return

    height, width = frame.shape[:2]
    points = []
    for point in landmarks:
        x = int(np.clip(point[0], 0, width - 1))
        y = int(np.clip(point[1], 0, height - 1))
        points.append((x, y))

    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], MESH_COLOR, 1, cv2.LINE_AA)
    if not connections:
        _draw_delaunay_mesh(frame, points)
    for point in points:
        cv2.circle(frame, point, 1, LANDMARK_COLOR, -1, cv2.LINE_AA)


def _format_value(value: Optional[float], precision: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{precision}f}"


def _draw_overlay(
    frame: np.ndarray,
    result: LandmarkFrameFeatures,
    fps: float,
    processing_ms: float,
) -> None:
    face_status = "YES" if result.face_detected else "NO"
    face_color = LANDMARK_COLOR if result.face_detected else ERROR_COLOR
    lines = [
        (f"FPS: {fps:.1f}", TEXT_COLOR),
        (f"Face detected: {face_status}", face_color),
        (f"Faces: {result.num_faces}", TEXT_COLOR),
        (
            f"Landmarks: {len(result.landmarks) if result.landmarks is not None else 0}",
            TEXT_COLOR,
        ),
        (f"Processing: {processing_ms:.1f} ms", TEXT_COLOR),
        (f"Gaze H/V: {_format_value(result.gaze_horizontal, 2)} / "
         f"{_format_value(result.gaze_vertical, 2)}", FEATURE_COLOR),
        (f"EAR: {_format_value(result.ear)}", FEATURE_COLOR),
        (
            f"Blink: {'YES' if result.eye_closed else 'NO' if result.eye_closed is not None else 'N/A'}",
            FEATURE_COLOR,
        ),
        (
            f"Pose P/Y/R: {_format_value(result.pitch, 1)} / "
            f"{_format_value(result.yaw, 1)} / {_format_value(result.roll, 1)}",
            FEATURE_COLOR,
        ),
    ]

    y = 28
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 26

    if not result.face_detected:
        cv2.putText(
            frame,
            "No face detected - move into view",
            (15, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            ERROR_COLOR,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Q / ESC: exit",
        (15, frame.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    connections = _load_connections()
    print("Starting MediaPipe webcam test.")
    print("Press Q or ESC in the camera window to exit.")
    print(f"Mesh connections loaded: {len(connections)}")

    frame_count = 0
    detected_face_frames = 0
    max_faces = 0
    processing_total_ms = 0.0
    gaze_frames = 0
    ear_frames = 0
    pose_frames = 0
    started_at = time.perf_counter()
    previous_frame_at = started_at

    try:
        with VideoCaptureSession(source=0, duration_seconds=3600) as camera:
            with MediaPipeProcessor(max_num_faces=2, refine_landmarks=True) as processor:
                frames = camera.frames()
                while True:
                    frame = next(frames, None)
                    if frame is None:
                        print("Camera stream ended.")
                        break

                    now = time.perf_counter()
                    frame_count += 1
                    if not frame.valid or frame.rgb_frame is None:
                        display = np.zeros((480, 640, 3), dtype=np.uint8)
                        result = LandmarkFrameFeatures(
                            face_detected=False,
                            error=frame.error or "invalid_frame",
                        )
                        processing_ms = 0.0
                    else:
                        processing_started = time.perf_counter()
                        result = processor.process_frame(
                            frame.rgb_frame,
                            timestamp_ms=int(frame.timestamp * 1000),
                        )
                        processing_ms = (time.perf_counter() - processing_started) * 1000
                        display = cv2.cvtColor(frame.rgb_frame, cv2.COLOR_RGB2BGR)
                        processing_total_ms += processing_ms
                        if result.face_detected:
                            detected_face_frames += 1
                        if result.gaze_horizontal is not None and result.gaze_vertical is not None:
                            gaze_frames += 1
                        if result.ear is not None:
                            ear_frames += 1
                        if result.yaw is not None and result.pitch is not None and result.roll is not None:
                            pose_frames += 1
                        max_faces = max(max_faces, result.num_faces)

                    elapsed = max(now - started_at, 1e-9)
                    instantaneous_fps = 1.0 / max(now - previous_frame_at, 1e-9)
                    average_fps = frame_count / elapsed
                    fps = 0.5 * instantaneous_fps + 0.5 * average_fps
                    previous_frame_at = now

                    _draw_landmarks(display, result.landmarks, connections)
                    display = cv2.flip(display, 1)
                    _draw_overlay(display, result, fps, processing_ms)
                    cv2.imshow(WINDOW_NAME, display)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
    except CameraError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"MediaPipe dependency error: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started_at
    average_processing_ms = processing_total_ms / max(detected_face_frames, 1)
    print(f"MediaPipe webcam test ended after {elapsed:.2f}s ({frame_count} frames).")
    print(f"Frames with face: {detected_face_frames}")
    print(f"Maximum faces detected: {max_faces}")
    print(f"Frames with gaze values: {gaze_frames}")
    print(f"Frames with EAR values: {ear_frames}")
    print(f"Frames with head-pose values: {pose_frames}")
    print(f"Average processing time: {average_processing_ms:.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

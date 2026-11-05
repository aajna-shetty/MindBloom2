"""
app/video/mediapipe_processor.py

MediaPipe Face Landmarker based facial landmark processor.

Responsibilities:
    - Detect the primary face
    - Extract facial landmarks
    - Estimate gaze proxy
    - Calculate eye aspect ratio (EAR)
    - Estimate head pose
    - Calculate frame-to-frame facial movement

Uses the modern MediaPipe Tasks API:
    mp.tasks.vision.FaceLandmarker

Compatible with MediaPipe 1.x.
"""


from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:
    mp = None
    _import_error = exc

from app.video.config import EAR_CLOSED_THRESHOLD


logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/"
    "face_landmarker.task"
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


# ============================================================================
# LANDMARK INDICES
# ============================================================================

# EAR-style landmarks
LEFT_EYE_EAR_IDX = [
    362, 385, 387,
    263, 373, 380
]

RIGHT_EYE_EAR_IDX = [
    33, 160, 158,
    133, 153, 144
]


# Eye bounding boxes
LEFT_EYE_BOX_IDX = {
    "left": 362,
    "right": 263,
    "top": 386,
    "bottom": 374,
}

RIGHT_EYE_BOX_IDX = {
    "left": 33,
    "right": 133,
    "top": 159,
    "bottom": 145,
}


# Iris centers
LEFT_IRIS_CENTER_IDX = 473
RIGHT_IRIS_CENTER_IDX = 468


# Head pose landmarks
POSE_LANDMARK_IDX = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}


# Generic 3D face model
_MODEL_3D_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


# Facial movement regions
MOUTH_IDX = [
    61,
    291,
    13,
    14,
    78,
    308,
    82,
    312,
    87,
    317,
]

LEFT_EYEBROW_IDX = [
    70,
    63,
    105,
    66,
    107,
]

RIGHT_EYEBROW_IDX = [
    336,
    296,
    334,
    293,
    300,
]


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class LandmarkFrameFeatures:
    """
    Features extracted from one video frame.
    """
    @staticmethod
    def extract_face_bbox(landmarks, frame_shape, margin: float = 0.25):
        """
        Extracts an expanded bounding box from detected normalized facial landmarks.
        Includes extra top headroom for inner-eyebrow raise (AU1).
        """
        if not landmarks:
            return None
            
        h, w = frame_shape[:2]
        
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # 25% horizontal, 35% top margin for eyebrows/forehead, 20% bottom for chin
        x_min = max(0, int(x_min - box_w * margin))
        y_min = max(0, int(y_min - box_h * (margin + 0.10)))
        x_max = min(w, int(x_max + box_w * margin))
        y_max = min(h, int(y_max + box_h * margin))
        
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return None
            
        return (x_min, y_min, x_max, y_max)

    face_detected: bool

    num_faces: int = 0
    landmarks: Optional[np.ndarray] = None

    gaze_horizontal: Optional[float] = None
    gaze_vertical: Optional[float] = None

    eye_closed: Optional[bool] = None
    ear: Optional[float] = None

    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None

    overall_movement: Optional[float] = None
    mouth_movement: Optional[float] = None
    eyebrow_movement: Optional[float] = None

    error: Optional[str] = None


# ============================================================================
# HELPERS
# ============================================================================

def _euclidean(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return float(np.linalg.norm(a - b))


def _download_model() -> None:
    """
    Download the MediaPipe Face Landmarker model once.
    """

    if os.path.exists(MODEL_PATH):
        logger.info(
            "MediaPipe model found: %s",
            MODEL_PATH,
        )
        return

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    logger.info(
        "MediaPipe Face Landmarker model not found."
    )

    logger.info(
        "Downloading model..."
    )

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH,
        )

    except Exception as exc:

        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
            except OSError:
                pass

        raise RuntimeError(
            "Could not download MediaPipe Face Landmarker model.\n"
            "Please check your internet connection.\n"
            f"{exc}"
        ) from exc

    logger.info(
        "MediaPipe model downloaded successfully."
    )


# ============================================================================
# PROCESSOR
# ============================================================================

class MediaPipeProcessor:
    """
    MediaPipe Face Landmarker processor.

    Uses:

        mp.tasks.vision.FaceLandmarker

    instead of the old:

        mp.solutions.face_mesh

    API.
    """

    def __init__(
        self,
        max_num_faces: int = 2,
        refine_landmarks: bool = True,
    ):

        if mp is None:
            raise ImportError(
                "MediaPipe is not installed.\n"
                "Run:\n"
                "pip install mediapipe"
            ) from _import_error

        self.max_num_faces = max(
            1,
            max_num_faces,
        )

        self.refine_landmarks = refine_landmarks

        # Previous frame landmarks
        self._prev_landmarks: Optional[np.ndarray] = None

        # Blink tracking
        self._consec_closed_frames = 0

        # ------------------------------------------------------------
        # IMPORTANT:
        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        # ------------------------------------------------------------

        self._last_timestamp_ms = -1

        # Download model if necessary
        _download_model()

        # ------------------------------------------------------------
        # Create MediaPipe Face Landmarker
        # ------------------------------------------------------------

        try:

            BaseOptions = mp.tasks.BaseOptions

            FaceLandmarker = (
                mp.tasks.vision.FaceLandmarker
            )

            FaceLandmarkerOptions = (
                mp.tasks.vision.FaceLandmarkerOptions
            )

            RunningMode = (
                mp.tasks.vision.RunningMode
            )

            base_options = BaseOptions(
                model_asset_path=MODEL_PATH
            )

            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=RunningMode.VIDEO,
                num_faces=self.max_num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )

            self._face_landmarker = (
                FaceLandmarker.create_from_options(
                    options
                )
            )

            logger.info(
                "MediaPipe FaceLandmarker initialized successfully."
            )

        except Exception as exc:

            logger.exception(
                "Failed to initialize MediaPipe FaceLandmarker."
            )

            raise RuntimeError(
                "Could not initialize MediaPipe FaceLandmarker: "
                f"{exc}"
            ) from exc

    # ========================================================================
    # TIMESTAMP HANDLING
    # ========================================================================

    def _get_safe_timestamp(
        self,
        timestamp_ms: Optional[int],
    ) -> int:
        """
        Return a MediaPipe-safe timestamp.

        MediaPipe VIDEO mode requires:

            timestamp[n] > timestamp[n-1]

        Camera timestamps can occasionally be duplicated because they
        are converted to integer milliseconds.

        Example problematic sequence:

            10
            10
            11
            11

        We convert it into:

            10
            11
            12
            13

        This prevents:

            Input timestamp must be monotonically increasing.
        """

        if timestamp_ms is None:
            timestamp_ms = 0

        timestamp_ms = int(timestamp_ms)

        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp_ms

        return timestamp_ms

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(
        self,
    ) -> "MediaPipeProcessor":
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.close()
        return False

    def close(self) -> None:
        """
        Safely close MediaPipe.
        """

        try:

            if hasattr(
                self,
                "_face_landmarker",
            ):

                self._face_landmarker.close()

        except Exception as exc:

            logger.warning(
                "Error closing MediaPipe FaceLandmarker: %s",
                exc,
            )

    # ========================================================================
    # FRAME PROCESSING
    # ========================================================================

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> LandmarkFrameFeatures:
        """
        Process one RGB frame.

        Returns LandmarkFrameFeatures.

        Expected errors are returned inside the result instead of crashing
        the entire video pipeline.
        """

        try:

            # ------------------------------------------------------------
            # Validate frame
            # ------------------------------------------------------------

            if rgb_frame is None:

                return LandmarkFrameFeatures(
                    face_detected=False,
                    error="empty_frame",
                )

            if not isinstance(
                rgb_frame,
                np.ndarray,
            ):

                return LandmarkFrameFeatures(
                    face_detected=False,
                    error="invalid_frame_type",
                )

            if rgb_frame.ndim != 3:

                return LandmarkFrameFeatures(
                    face_detected=False,
                    error="invalid_frame_dimensions",
                )

            # ------------------------------------------------------------
            # Ensure contiguous RGB array
            # ------------------------------------------------------------

            rgb_frame = np.ascontiguousarray(
                rgb_frame
            )

            # ------------------------------------------------------------
            # Create MediaPipe Image
            # ------------------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            # ------------------------------------------------------------
            # FIX:
            # Generate strictly increasing timestamp.
            # ------------------------------------------------------------

            safe_timestamp = (
                self._get_safe_timestamp(
                    timestamp_ms
                )
            )

            # ------------------------------------------------------------
            # Run Face Landmarker
            # ------------------------------------------------------------

            results = (
                self._face_landmarker.detect_for_video(
                    mp_image,
                    safe_timestamp,
                )
            )

            faces = getattr(
                results,
                "face_landmarks",
                None,
            )

            # ------------------------------------------------------------
            # No face
            # ------------------------------------------------------------

            if not faces:

                self._consec_closed_frames = 0

                return LandmarkFrameFeatures(
                    face_detected=False,
                    num_faces=0,
                    error="no_face_detected",
                )

            num_faces = len(faces)

            # ------------------------------------------------------------
            # Select primary face
            # ------------------------------------------------------------

            face = self._select_primary_face(
                faces,
                rgb_frame.shape,
            )

            h, w = rgb_frame.shape[:2]

            pts = self._to_pixel_array(
                face,
                w,
                h,
            )

            # ------------------------------------------------------------
            # GAZE
            # ------------------------------------------------------------

            try:

                gaze_h, gaze_v = (
                    self.extract_gaze(pts)
                )

            except Exception as exc:

                logger.debug(
                    "Gaze extraction failed: %s",
                    exc,
                )

                gaze_h = None
                gaze_v = None

            # ------------------------------------------------------------
            # EYE / EAR
            # ------------------------------------------------------------

            try:

                ear, eye_closed = (
                    self.extract_eye_features(pts)
                )

            except Exception as exc:

                logger.debug(
                    "Eye extraction failed: %s",
                    exc,
                )

                ear = None
                eye_closed = None

            # ------------------------------------------------------------
            # HEAD POSE
            # ------------------------------------------------------------

            try:

                yaw, pitch, roll = (
                    self.extract_head_pose(
                        pts,
                        w,
                        h,
                    )
                )

            except Exception as exc:

                logger.debug(
                    "Head pose extraction failed: %s",
                    exc,
                )

                yaw = None
                pitch = None
                roll = None

            # ------------------------------------------------------------
            # FACIAL MOVEMENT
            # ------------------------------------------------------------

            try:

                (
                    overall_mv,
                    mouth_mv,
                    eyebrow_mv,
                ) = self.extract_facial_movement(
                    pts
                )

            except Exception as exc:

                logger.debug(
                    "Facial movement extraction failed: %s",
                    exc,
                )

                overall_mv = None
                mouth_mv = None
                eyebrow_mv = None

            # ------------------------------------------------------------
            # Save current landmarks
            # ------------------------------------------------------------

            self._prev_landmarks = pts.copy()

            # ------------------------------------------------------------
            # Return features
            # ------------------------------------------------------------

            return LandmarkFrameFeatures(
                face_detected=True,
                num_faces=num_faces,
                landmarks=pts,

                gaze_horizontal=gaze_h,
                gaze_vertical=gaze_v,

                eye_closed=eye_closed,
                ear=ear,

                yaw=yaw,
                pitch=pitch,
                roll=roll,

                overall_movement=overall_mv,
                mouth_movement=mouth_mv,
                eyebrow_movement=eyebrow_mv,
            )

        except Exception as exc:

            logger.warning(
                "MediaPipe frame processing failed: %s",
                exc,
            )

            return LandmarkFrameFeatures(
                face_detected=False,
                error=f"mediapipe_error:{exc}",
            )

    # ========================================================================
    # FACE SELECTION
    # ========================================================================

    @staticmethod
    def _select_primary_face(
        faces,
        frame_shape,
    ):
        
        """
        Select the largest detected face.
        """

        if len(faces) == 1:
            return faces[0]

        h, w = frame_shape[:2]

        def area(face):

            xs = [
                lm.x
                for lm in face
            ]

            ys = [
                lm.y
                for lm in face
            ]

            width = max(xs) - min(xs)
            height = max(ys) - min(ys)

            return (
                width
                * w
                * height
                * h
            )

        return max(
            faces,
            key=area,
        )
    @staticmethod
    def extract_face_bbox(landmarks, frame_shape, margin: float = 0.28):
        if not landmarks:
            return None
            
        h, w = frame_shape[:2]
        
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Expanded margin for forehead furrows and jaw tension
        x_min = max(0, int(x_min - box_w * margin))
        y_min = max(0, int(y_min - box_h * (margin + 0.08)))  # Extra 8% on top for eyebrows/forehead
        x_max = min(w, int(x_max + box_w * margin))
        y_max = min(h, int(y_max + box_h * margin))
        
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return None
            
        return (x_min, y_min, x_max, y_max)

    # ========================================================================
    # LANDMARK CONVERSION
    # ========================================================================

    @staticmethod
    def _to_pixel_array(
        face_landmarks,
        w: int,
        h: int,
    ) -> np.ndarray:
        """
        Convert normalized MediaPipe landmarks into:

            (x_pixel, y_pixel, z_pixel)
        """

        return np.array(
            [
                (
                    lm.x * w,
                    lm.y * h,
                    lm.z * w,
                )
                for lm in face_landmarks
            ],
            dtype=np.float64,
        )

    # ========================================================================
    # RAW LANDMARK EXTRACTION
    # ========================================================================

    def extract_landmarks(
        self,
        rgb_frame: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """
        Return pixel-space landmarks.

        Uses the same timestamp protection as process_frame().
        """

        try:

            if rgb_frame is None:
                return None

            rgb_frame = np.ascontiguousarray(
                rgb_frame
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            safe_timestamp = (
                self._get_safe_timestamp(
                    timestamp_ms
                )
            )

            results = (
                self._face_landmarker.detect_for_video(
                    mp_image,
                    safe_timestamp,
                )
            )

            faces = getattr(
                results,
                "face_landmarks",
                None,
            )

            if not faces:
                return None

            face = self._select_primary_face(
                faces,
                rgb_frame.shape,
            )

            h, w = rgb_frame.shape[:2]

            return self._to_pixel_array(
                face,
                w,
                h,
            )

        except Exception as exc:

            logger.debug(
                "Raw landmark extraction failed: %s",
                exc,
            )

            return None

    # ========================================================================
    # GAZE
    # ========================================================================

    def extract_gaze(
        self,
        pts: np.ndarray,
    ) -> Tuple[
        Optional[float],
        Optional[float],
    ]:
        """
        Estimate gaze proxy from iris position.

        Returns:

            (horizontal, vertical)

        Both are approximately in [0, 1].
        """
        refine_landmarks = getattr(
            self,
            "refine_landmarks",
            getattr(self, "_refine_landmarks", True),
        )
        if not refine_landmarks:
            return None, None

        required = max(
            LEFT_IRIS_CENTER_IDX,
            RIGHT_IRIS_CENTER_IDX,
        )

        if pts.shape[0] <= required:
            return None, None

        def eye_gaze(
            box_idx,
            iris_idx,
        ):

            left = pts[
                box_idx["left"]
            ][:2]

            right = pts[
                box_idx["right"]
            ][:2]

            top = pts[
                box_idx["top"]
            ][:2]

            bottom = pts[
                box_idx["bottom"]
            ][:2]

            iris = pts[
                iris_idx
            ][:2]

            width = right[0] - left[0]
            height = bottom[1] - top[1]

            if abs(width) < 1e-6:
                return None, None

            if abs(height) < 1e-6:
                return None, None

            gaze_h = (
                iris[0] - left[0]
            ) / width

            gaze_v = (
                iris[1] - top[1]
            ) / height

            return (
                float(
                    np.clip(
                        gaze_h,
                        0.0,
                        1.0,
                    )
                ),
                float(
                    np.clip(
                        gaze_v,
                        0.0,
                        1.0,
                    )
                ),
            )

        left_h, left_v = eye_gaze(
            LEFT_EYE_BOX_IDX,
            LEFT_IRIS_CENTER_IDX,
        )

        right_h, right_v = eye_gaze(
            RIGHT_EYE_BOX_IDX,
            RIGHT_IRIS_CENTER_IDX,
        )

        horizontal = [
            value
            for value in (
                left_h,
                right_h,
            )
            if value is not None
        ]

        vertical = [
            value
            for value in (
                left_v,
                right_v,
            )
            if value is not None
        ]

        if not horizontal or not vertical:
            return None, None

        return (
            float(np.mean(horizontal)),
            float(np.mean(vertical)),
        )

    # ========================================================================
    # EYE / EAR
    # ========================================================================

    def extract_eye_features(
        self,
        pts: np.ndarray,
    ) -> Tuple[
        Optional[float],
        Optional[bool],
    ]:
        """
        Calculate Eye Aspect Ratio (EAR).
        """

        def ear_for(indices):

            points = [
                pts[index][:2]
                for index in indices
            ]

            p1, p2, p3, p4, p5, p6 = points

            vertical = (
                _euclidean(p2, p6)
                + _euclidean(p3, p5)
            )

            horizontal = (
                2.0
                * _euclidean(p1, p4)
            )

            if horizontal < 1e-6:
                return None

            return (
                vertical
                / horizontal
            )

        left_ear = ear_for(
            LEFT_EYE_EAR_IDX
        )

        right_ear = ear_for(
            RIGHT_EYE_EAR_IDX
        )

        values = [
            value
            for value in (
                left_ear,
                right_ear,
            )
            if value is not None
        ]

        if not values:
            return None, None

        ear = float(
            np.mean(values)
        )

        closed = (
            ear
            < EAR_CLOSED_THRESHOLD
        )

        if closed:

            self._consec_closed_frames += 1

        else:

            self._consec_closed_frames = 0

        return ear, closed

    # ========================================================================
    # HEAD POSE
    # ========================================================================

    def extract_head_pose(
        self,
        pts: np.ndarray,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """
        Estimate yaw, pitch and roll using solvePnP.
        """

        image_points = np.array(
            [
                pts[index][:2]
                for index in (
                    POSE_LANDMARK_IDX.values()
                )
            ],
            dtype=np.float64,
        )

        focal_length = float(
            frame_w
        )

        center = (
            frame_w / 2.0,
            frame_h / 2.0,
        )

        camera_matrix = np.array(
            [
                [
                    focal_length,
                    0,
                    center[0],
                ],
                [
                    0,
                    focal_length,
                    center[1],
                ],
                [
                    0,
                    0,
                    1,
                ],
            ],
            dtype=np.float64,
        )

        dist_coeffs = np.zeros(
            (4, 1),
            dtype=np.float64,
        )

        success, rotation_vec, _translation_vec = (
            cv2.solvePnP(
                _MODEL_3D_POINTS,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        )

        if not success:
            return None, None, None

        rotation_mat, _ = cv2.Rodrigues(
            rotation_vec
        )

        pose_mat = np.hstack(
            (
                rotation_mat,
                np.zeros(
                    (3, 1)
                ),
            )
        )

        (
            _,
            _,
            _,
            _,
            _,
            _,
            euler_angles,
        ) = cv2.decomposeProjectionMatrix(
            pose_mat
        )

        pitch, yaw, roll = [
            float(angle[0])
            for angle in euler_angles
        ]

        return (
            yaw,
            pitch,
            roll,
        )

    # ========================================================================
    # FACIAL MOVEMENT
    # ========================================================================

    def extract_facial_movement(
        self,
        pts: np.ndarray,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        Optional[float],
    ]:
        """
        Calculate frame-to-frame landmark movement.

        Movement is normalized by inter-ocular distance.
        """

        if self._prev_landmarks is None:
            return None, None, None

        if (
            self._prev_landmarks.shape
            != pts.shape
        ):
            return None, None, None

        left_eye = pts[
            LEFT_EYE_BOX_IDX["left"]
        ][:2]

        right_eye = pts[
            RIGHT_EYE_BOX_IDX["right"]
        ][:2]

        inter_ocular = _euclidean(
            left_eye,
            right_eye,
        )

        if inter_ocular < 1e-6:
            return None, None, None

        diffs = np.linalg.norm(
            pts[:, :2]
            - self._prev_landmarks[:, :2],
            axis=1,
        )

        overall = (
            float(np.mean(diffs))
            / inter_ocular
        )

        mouth = (
            float(
                np.mean(
                    diffs[MOUTH_IDX]
                )
            )
            / inter_ocular
        )

        eyebrow_indices = (
            LEFT_EYEBROW_IDX
            + RIGHT_EYEBROW_IDX
        )

        eyebrow = (
            float(
                np.mean(
                    diffs[
                        eyebrow_indices
                    ]
                )
            )
            / inter_ocular
        )

        return (
            overall,
            mouth,
            eyebrow,
        )
"""
app/video/schemas.py
─────────────────────────────────────────────────────────────────────────────
Pydantic request/response models for the new POST /video/analyze endpoint.
Kept separate from app/main.py's existing Pydantic models so the video
module stays a self-contained addition (§20 of spec: don't unnecessarily
touch existing definitions).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class VideoAnalyzeRequest(BaseModel):
    """
    `source` selects a live webcam index on the machine running the API.
    Pre-recorded videos must be sent through the multipart `video` field;
    arbitrary server filesystem paths are not accepted.
    """
    child_id: int
    source: str = Field(
        default="0",
        description="Server webcam index, for example '0'. Uploaded files use the video form field.",
    )
    duration_seconds: Optional[int] = Field(
        default=None,
        description="Overrides config.SESSION_DURATION_SECONDS for this session (e.g. dev testing).",
    )


class VideoAnalyzeResponse(BaseModel):
    status: str
    session_id: str
    video_model_status: str
    mediapipe: Dict[str, Any] = Field(default_factory=dict)
    vit: Dict[str, Any] = Field(default_factory=dict)
    combined_features: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    prediction: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    # Legacy fields retained for existing clients.
    video_score: Optional[float] = None
    video_confidence: Optional[float] = None
    predicted_class: Optional[int] = None
    session_duration_seconds: Optional[float] = None
    valid_frame_ratio: Optional[float] = None
    feature_count: Optional[int] = None
    detail: Optional[str] = None


class AssessmentResponse(BaseModel):
    assessment_id: str
    status: str
    participant_id: str
    questionnaire: Dict[str, Any]
    video_assessment: Dict[str, Any]
    overall_summary: Dict[str, Any]
    timing: Dict[str, Optional[float]]
    errors: List[Dict[str, str]] = Field(default_factory=list)

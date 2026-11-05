"""
Research-paper hybrid fusion of questionnaire and combined-video ASD scores.

    c = 2 * max(p_v, 1 - p_v) - 1
    alpha = clip(0.90 - 0.60 * c, 0.30, 0.90)
    R_hybrid = alpha * R_q + (1 - alpha) * R_v

p_v is the combined video model's P(ASD). It is never an emotion probability
or a Random Forest max-class confidence.

This module is kept separate from app.utils.combine_scores, which remains
the untouched fixed-weight reference used by /predict.
"""

from __future__ import annotations

import math
from typing import Any, Optional

FUSION_FUNCTION = "app.video.fusion.hybrid_fusion"
FUSION_VERSION = "research-paper-1"

ALPHA_FLOOR = 0.30
ALPHA_CEILING = 0.90
ALPHA_INTERCEPT = 0.90
ALPHA_SLOPE = 0.60


def _as_unit_interval(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        return None
    return numeric


def _require_questionnaire_score(value: Any) -> float:
    if value is None:
        raise ValueError("Questionnaire risk score is required.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Questionnaire risk score must be a finite number in [0, 1]."
        ) from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError("Questionnaire risk score must be a finite number in [0, 1].")
    return numeric


def video_confidence_from_probability(p_v: float) -> float:
    """Paper video confidence: c = 2 * max(p_v, 1 - p_v) - 1."""
    return (2.0 * max(p_v, 1.0 - p_v)) - 1.0


def get_alpha(video_confidence: Optional[float]) -> float:
    """
    Map paper confidence c -> questionnaire weight alpha.

    video_confidence=None (video unavailable) -> alpha=1.0 so R_hybrid is R_q.
    Otherwise alpha = 0.90 - 0.60 * c, clamped to [0.30, 0.90].
    """
    if video_confidence is None:
        return 1.0
    try:
        c = float(video_confidence)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(c):
        return 1.0
    alpha = ALPHA_INTERCEPT - (ALPHA_SLOPE * c)
    return float(min(ALPHA_CEILING, max(ALPHA_FLOOR, alpha)))


def _questionnaire_only(rq: float, video_model_available: bool) -> dict:
    return {
        "questionnaire_risk_score": rq,
        "video_risk_score": None,
        "video_probability": None,
        "video_confidence": None,
        "alpha": 1.0,
        "hybrid_risk_score": rq,
        "scoring_method": "questionnaire-only",
        "video_model_available": video_model_available,
        "questionnaire_score": rq,
        "video_score": None,
    }


def hybrid_fusion(
    questionnaire_score: float,
    video_score: Optional[float] = None,
    video_probability: Optional[float] = None,
    *,
    available: bool = True,
    video_model_available: Optional[bool] = None,
) -> dict:
    """
    Compute R_hybrid = alpha * R_q + (1 - alpha) * R_v.

    If p_v is omitted, R_v is used as p_v. Missing or invalid video input
    yields questionnaire-only scoring; missing video is never treated as 0.
    Invalid questionnaire scores raise ValueError.
    """
    rq = _require_questionnaire_score(questionnaire_score)
    model_available = True if video_model_available is None else bool(video_model_available)

    if available is False:
        return _questionnaire_only(rq, False)

    rv = _as_unit_interval(video_score)
    pv = video_probability if video_probability is not None else video_score
    pv = _as_unit_interval(pv)

    if rv is None or pv is None:
        return _questionnaire_only(rq, False if video_model_available is None else model_available)

    c = video_confidence_from_probability(pv)
    alpha = get_alpha(c)
    r_hybrid = (alpha * rq) + ((1.0 - alpha) * rv)
    if not math.isfinite(r_hybrid):
        raise ValueError("Hybrid risk score is not finite.")
    r_hybrid = float(r_hybrid)

    return {
        "questionnaire_risk_score": rq,
        "video_risk_score": rv,
        "video_probability": pv,
        "video_confidence": c,
        "alpha": alpha,
        "hybrid_risk_score": r_hybrid,
        "scoring_method": "hybrid",
        "video_model_available": model_available,
        "questionnaire_score": rq,
        "video_score": rv,
    }

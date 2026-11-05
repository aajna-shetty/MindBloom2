"""Reusable, explicitly documented threshold evaluation.

Thresholds in this module are engineering configuration unless a model
artifact supplies a validated threshold. They must never be treated as a
clinical diagnosis rule without population-specific validation evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


HYBRID_THRESHOLD_SOURCE = "MindBloom research paper"
HYBRID_THRESHOLD_VERSION = "research-paper-1"
HYBRID_LOW_EXCLUSIVE_UPPER = 0.35
HYBRID_HIGH_EXCLUSIVE_LOWER = 0.65
HYBRID_THRESHOLD_CONFIGURATION = {
    "source": HYBRID_THRESHOLD_SOURCE,
    "version": HYBRID_THRESHOLD_VERSION,
    "low_exclusive_upper": HYBRID_LOW_EXCLUSIVE_UPPER,
    "moderate_inclusive": [HYBRID_LOW_EXCLUSIVE_UPPER, HYBRID_HIGH_EXCLUSIVE_LOWER],
    "high_exclusive_lower": HYBRID_HIGH_EXCLUSIVE_LOWER,
    "boundary_inclusivity": "Moderate includes 0.35 and 0.65",
}


@dataclass(frozen=True)
class ThresholdEvaluation:
    value: Optional[float]
    status: str
    classification: Optional[str]
    operator: Optional[str]
    threshold: Any
    source: str
    version: str
    reason: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "classification": self.classification,
            "operator": self.operator,
            "threshold": self.threshold,
            "source": self.source,
            "version": self.version,
            "reason": self.reason,
        }


class ThresholdEngine:
    """Evaluate configured thresholds while making missing data explicit."""

    def evaluate_probability(
        self,
        value: Optional[float],
        *,
        source: str,
        version: str,
        cutoff: Optional[float] = None,
        labels: tuple[str, str] = ("Below threshold", "Above screening threshold"),
    ) -> ThresholdEvaluation:
        if value is None:
            return ThresholdEvaluation(
                value=None,
                status="MISSING",
                classification=None,
                operator=">=",
                threshold=cutoff,
                source=source,
                version=version,
                reason="No probability was produced.",
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ThresholdEvaluation(
                value=None,
                status="INVALID",
                classification=None,
                operator=">=",
                threshold=cutoff,
                source=source,
                version=version,
                reason="Probability must be finite and between 0 and 1.",
            )
        if cutoff is None:
            return ThresholdEvaluation(
                value=numeric,
                status="UNAVAILABLE",
                classification=None,
                operator=None,
                threshold=None,
                source=source,
                version=version,
                reason="No validated cutoff is configured.",
            )
        return ThresholdEvaluation(
            value=numeric,
            status="EVALUATED",
            classification=labels[1] if numeric >= cutoff else labels[0],
            operator=">=",
            threshold=cutoff,
            source=source,
            version=version,
        )

    def evaluate_range(
        self,
        value: Optional[float],
        *,
        lower: float,
        upper: float,
        source: str,
        version: str,
    ) -> ThresholdEvaluation:
        if value is None:
            return ThresholdEvaluation(
                value=None,
                status="MISSING",
                classification=None,
                operator="between",
                threshold={"lower": lower, "upper": upper},
                source=source,
                version=version,
                reason="Measurement was not available.",
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric):
            return ThresholdEvaluation(
                value=None,
                status="INVALID",
                classification=None,
                operator="between",
                threshold={"lower": lower, "upper": upper},
                source=source,
                version=version,
                reason="Measurement is not finite.",
            )
        within = lower <= numeric <= upper
        return ThresholdEvaluation(
            value=numeric,
            status="EVALUATED",
            classification="Within reference interval" if within else "Outside reference interval",
            operator="between",
            threshold={"lower": lower, "upper": upper},
            source=source,
            version=version,
        )

    def evaluate_hybrid_risk(
        self,
        value: Optional[float],
        *,
        source: str = HYBRID_THRESHOLD_SOURCE,
        version: str = HYBRID_THRESHOLD_VERSION,
    ) -> ThresholdEvaluation:
        """Classify the final hybrid score using the research-paper bands."""
        threshold = dict(HYBRID_THRESHOLD_CONFIGURATION)
        if value is None:
            return ThresholdEvaluation(
                value=None,
                status="MISSING",
                classification=None,
                operator="paper-bands",
                threshold=threshold,
                source=source,
                version=version,
                reason="No hybrid risk score was produced.",
            )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = math.nan
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            return ThresholdEvaluation(
                value=None,
                status="INVALID",
                classification=None,
                operator="paper-bands",
                threshold=threshold,
                source=source,
                version=version,
                reason="Hybrid risk score must be finite and between 0 and 1.",
            )
        if numeric < HYBRID_LOW_EXCLUSIVE_UPPER:
            classification = "Low"
        elif numeric <= HYBRID_HIGH_EXCLUSIVE_LOWER:
            classification = "Moderate"
        else:
            classification = "High"
        return ThresholdEvaluation(
            value=numeric,
            status="EVALUATED",
            classification=classification,
            operator="paper-bands",
            threshold=threshold,
            source=source,
            version=version,
        )

"""Reusable services for the persisted end-to-end assessment workflow."""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Optional

from app.database import (
    Assessment,
    IntegratedAssessment,
    QuestionnaireResult,
    VideoAssessment,
)
from app.questionnaire import FEATURE_ORDER, build_feature_vector_from_answers
from app.thresholds import HYBRID_THRESHOLD_CONFIGURATION, ThresholdEngine
from app.utils import risk_level
from app.video.features import VIDEO_FEATURES
from app.video.fusion import FUSION_FUNCTION, FUSION_VERSION, hybrid_fusion
from app.video.service import VideoAnalysisResult, run_video_session


class QuestionnaireService:
    def __init__(self, model: Any, scaler: Any):
        self.model = model
        self.scaler = scaler

    def submit_responses(
        self,
        db,
        assessment_id: str,
        age: float,
        responses: dict[str, Any],
    ) -> QuestionnaireResult:
        missing = [f"A{i}_Score" for i in range(1, 11) if f"A{i}_Score" not in responses]
        if missing:
            raise ValueError(f"Missing questionnaire responses: {', '.join(missing)}")
        normalized = dict(responses)
        normalized["age"] = age
        for key in [f"A{i}_Score" for i in range(1, 11)]:
            value = float(normalized[key])
            if value not in (0.0, 1.0):
                raise ValueError(f"{key} must be 0 or 1")
            normalized[key] = value
        vector = build_feature_vector_from_answers(normalized)
        if len(vector) != len(FEATURE_ORDER):
            raise ValueError("Questionnaire feature vector has an unexpected length")
        score = float(self.model.predict_proba(self.scaler.transform([vector]))[0][1])
        threshold = ThresholdEngine().evaluate_probability(
            score,
            cutoff=0.70,
            source="app.utils.RISK_THRESHOLDS",
            version="repository-1",
            labels=("Below high-risk band", "High-risk band"),
        )
        result = QuestionnaireResult(
            assessment_id=assessment_id,
            instrument="MindBloom questionnaire model",
            responses_json=normalized,
            raw_score=score,
            maximum_score=1.0,
            normalized_score=score,
            scoring_method="trained questionnaire model predict_proba on the 32-feature questionnaire vector",
            threshold_metadata_json={
                **threshold.as_dict(),
                "validated_instrument": False,
            },
            interpretation_json={
                "risk_level": risk_level(score),
                "validated_instrument": False,
                "score_type": "model_probability",
            },
        )
        db.add(result)
        db.flush()
        return result


class CombinedVideoAssessmentService:
    def run_and_persist(
        self,
        db,
        assessment_id: str,
        source: int | str,
        duration_seconds: int,
        use_vit: bool = True,
    ) -> VideoAssessment:
        result: VideoAnalysisResult = run_video_session(
            source=source,
            duration_seconds=duration_seconds,
            use_vit=use_vit,
        )
        quality = result.mediapipe
        total_frames = quality.get("total_frames", 0)
        record = VideoAssessment(
            assessment_id=assessment_id,
            session_id=result.session_id,
            status=result.status,
            duration_seconds=result.session_duration_seconds,
            total_frames=total_frames,
            valid_frames=quality.get("valid_frames"),
            captured_frames=quality.get("captured_frames", total_frames),
            mediapipe_processed_frames=quality.get("mediapipe_processed_frames", 0),
            vit_submitted_frames=quality.get("vit_attempts", 0),
            vit_analyzed_frames=quality.get("vit_successes", 0),
            processing_errors=quality.get("processing_errors", 0),
            mediapipe_statistics_json=result.mediapipe,
            vit_statistics_json=result.vit,
            feature_vector_json=result.combined_features,
            feature_schema_json={
                "name": "MindBloom video behavioral features",
                "version": "47",
                "features": VIDEO_FEATURES,
            },
            model_metadata_json={
                "video_model_status": result.video_model_status,
                "validated_for_asd": False,
            },
            performance_json=result.performance,
            video_report_json={
                "status": result.status,
                "video_model_status": result.video_model_status,
                "prediction": result.prediction,
                "performance": result.performance,
                "errors": result.errors,
                "detail": result.detail,
            },
        )
        db.add(record)
        db.flush()
        return record


def _questionnaire_risk_score(questionnaire: QuestionnaireResult) -> float:
    if questionnaire.normalized_score is not None:
        return questionnaire.normalized_score
    return questionnaire.raw_score


def _combined_video_asd_probability(video: VideoAssessment) -> tuple[Optional[float], bool]:
    """Return (P(ASD), model_available) from the combined video model only."""
    metadata = video.model_metadata_json if isinstance(video.model_metadata_json, dict) else {}
    model_available = metadata.get("video_model_status") == "available"
    report = video.video_report_json if isinstance(video.video_report_json, dict) else {}
    prediction = report.get("prediction") if isinstance(report, dict) else None
    if not isinstance(prediction, dict):
        return None, model_available
    probabilities = prediction.get("probabilities")
    asd = probabilities.get("asd") if isinstance(probabilities, dict) else None
    if (
        video.status not in {"success", "video_model_not_available"}
        or prediction.get("available") is not True
        or not model_available
    ):
        return None, model_available
    return asd, model_available


class AssessmentIntegrationService:
    def create_integrated_assessment(self, db, assessment_id: str) -> IntegratedAssessment:
        assessment = db.get(Assessment, assessment_id)
        if assessment is None:
            raise ValueError("Assessment not found")
        questionnaire = assessment.questionnaire_result
        video = assessment.video_assessment
        if questionnaire is None or video is None:
            raise ValueError("Questionnaire and video results are both required")
        if video.status not in {"success", "video_model_not_available"}:
            raise ValueError(f"Video assessment is not complete: {video.status}")

        questionnaire_features = build_feature_vector_from_answers(
            questionnaire.responses_json
        )
        video_features = (
            video.feature_vector_json.get("features", {})
            if isinstance(video.feature_vector_json, dict)
            else {}
        )
        combined_features = {
            "schema": {
                "name": "MindBloom questionnaire + video observation representation",
                "version": "1",
                "questionnaire_feature_order": FEATURE_ORDER,
                "video_feature_order": VIDEO_FEATURES,
            },
            "questionnaire": {
                name: float(value)
                for name, value in zip(FEATURE_ORDER, questionnaire_features)
            },
            "video": video_features,
            "model_input_ready": False,
            "reason": "No validated multimodal ASD model or participant-level labeled dataset is available.",
        }
        video_probability, video_model_available = _combined_video_asd_probability(video)
        fusion = hybrid_fusion(
            _questionnaire_risk_score(questionnaire),
            video_probability,
            video_probability,
            available=video_probability is not None and video_model_available,
            video_model_available=video_model_available,
        )
        threshold_evaluation = ThresholdEngine().evaluate_hybrid_risk(
            fusion["hybrid_risk_score"]
        )
        hybrid_scoring = {
            **fusion,
            "risk_category": threshold_evaluation.classification,
            "thresholds": dict(HYBRID_THRESHOLD_CONFIGURATION),
        }
        model_metadata = {
            "name": FUSION_FUNCTION,
            "version": FUSION_VERSION,
            "available": fusion["scoring_method"] == "hybrid",
            "video_model_available": video_model_available,
            "validated_for_asd": False,
            "training_dataset": None,
            "validation_metrics": None,
        }
        threshold_metadata = {
            **threshold_evaluation.as_dict(),
            "configuration": dict(HYBRID_THRESHOLD_CONFIGURATION),
        }
        report = {
            "questionnaire_result": {
                "instrument": questionnaire.instrument,
                "score": questionnaire.raw_score,
                "interpretation": questionnaire.interpretation_json,
            },
            "video_result": video.video_report_json,
            "video_observations": {
                "mediapipe": video.mediapipe_statistics_json,
                "vit": video.vit_statistics_json,
                "features": video.feature_vector_json,
            },
            "combined_features": combined_features,
            "combined_model": model_metadata,
            "combined_threshold": threshold_metadata,
            "hybrid_scoring": hybrid_scoring,
            "consistent_findings": [],
            "different_findings": [
                "Questionnaire score and video observations remain separate modalities; only the combined video model ASD probability is fused with the questionnaire probability."
            ],
            "limitations": [
                "No clinically validated multimodal ASD model is available.",
                "Video measurements are observational and must not be interpreted as a diagnosis.",
                "Missing video ASD probability uses questionnaire-only scoring and is not treated as zero risk.",
            ],
            "overall_asd_prediction": {
                "hybrid_risk_score": fusion["hybrid_risk_score"],
                "risk_category": threshold_evaluation.classification,
                "scoring_method": fusion["scoring_method"],
            },
            "recommendation": "Professional assessment where indicated.",
        }
        existing = assessment.integrated_assessment
        if existing is not None:
            existing.status = "COMPLETED"
            existing.combined_features_json = combined_features
            existing.model_metadata_json = model_metadata
            existing.threshold_metadata_json = threshold_metadata
            existing.report_json = report
            return existing
        integrated = IntegratedAssessment(
            assessment_id=assessment_id,
            status="COMPLETED",
            combined_features_json=combined_features,
            model_metadata_json=model_metadata,
            threshold_metadata_json=threshold_metadata,
            report_json=report,
        )
        db.add(integrated)
        assessment.status = "COMPLETED"
        assessment.completed_at = datetime.datetime.utcnow()
        db.flush()
        return integrated


def create_assessment(db, participant_id: str) -> Assessment:
    assessment = Assessment(
        id=str(uuid.uuid4()),
        participant_id=participant_id,
        status="QUESTIONNAIRE_PENDING",
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)
    return assessment

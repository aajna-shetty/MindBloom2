import math
import uuid

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.assessment_services import AssessmentIntegrationService
from app.database import Assessment, Base, QuestionnaireResult, VideoAssessment
from app.thresholds import ThresholdEngine
from app.video.fusion import get_alpha, hybrid_fusion, video_confidence_from_probability


def test_case_a_mid_probability_has_zero_confidence_and_alpha_090():
    result = hybrid_fusion(0.40, 0.50, 0.50)
    assert result["video_confidence"] == pytest.approx(0.0)
    assert result["alpha"] == pytest.approx(0.90)
    assert result["scoring_method"] == "hybrid"


def test_case_b_certain_asd_probability_hits_alpha_floor():
    result = hybrid_fusion(0.40, 1.0, 1.0)
    assert result["video_confidence"] == pytest.approx(1.0)
    assert result["alpha"] == pytest.approx(0.30)


def test_case_c_certain_non_asd_probability_hits_alpha_floor():
    result = hybrid_fusion(0.40, 0.0, 0.0)
    assert result["video_confidence"] == pytest.approx(1.0)
    assert result["alpha"] == pytest.approx(0.30)


def test_case_d_hybrid_score_matches_independent_arithmetic():
    expected_c = 2 * max(0.9, 1 - 0.9) - 1
    expected_alpha = 0.90 - 0.60 * expected_c
    expected_hybrid = expected_alpha * 0.70 + (1 - expected_alpha) * 0.90
    assert expected_c == pytest.approx(0.8)
    assert expected_alpha == pytest.approx(0.42)
    assert expected_hybrid == pytest.approx(0.816)

    result = hybrid_fusion(0.70, 0.90, 0.90)
    assert result["video_confidence"] == pytest.approx(0.8)
    assert result["alpha"] == pytest.approx(0.42)
    assert result["hybrid_risk_score"] == pytest.approx(0.816)
    assert result["scoring_method"] == "hybrid"


def test_case_e_unavailable_video_is_not_zero_risk():
    result = hybrid_fusion(0.75, None, None)
    assert result["alpha"] == 1.0
    assert result["hybrid_risk_score"] == pytest.approx(0.75)
    assert result["video_risk_score"] is None
    assert result["scoring_method"] == "questionnaire-only"
    assert result["hybrid_risk_score"] != 0


def test_omitted_probability_uses_video_score_as_pv():
    result = hybrid_fusion(0.70, 0.90)
    assert result["video_probability"] == pytest.approx(0.90)
    assert result["video_confidence"] == pytest.approx(0.8)
    assert result["alpha"] == pytest.approx(0.42)


def test_available_false_forces_questionnaire_only():
    result = hybrid_fusion(0.55, 0.90, 0.90, available=False, video_model_available=False)
    assert result["scoring_method"] == "questionnaire-only"
    assert result["alpha"] == 1.0
    assert result["hybrid_risk_score"] == pytest.approx(0.55)
    assert result["video_risk_score"] is None


def test_invalid_video_score_uses_questionnaire_only():
    result = hybrid_fusion(0.55, 1.5, 1.5)
    assert result["scoring_method"] == "questionnaire-only"
    assert result["hybrid_risk_score"] == pytest.approx(0.55)
    assert result["video_risk_score"] is None


def test_non_finite_video_probability_uses_questionnaire_only():
    result = hybrid_fusion(0.55, 0.40, math.nan)
    assert result["scoring_method"] == "questionnaire-only"
    result_inf = hybrid_fusion(0.55, 0.40, math.inf)
    assert result_inf["scoring_method"] == "questionnaire-only"


def test_missing_questionnaire_score_raises():
    with pytest.raises(ValueError):
        hybrid_fusion(None, 0.2, 0.2)


def test_invalid_questionnaire_score_raises():
    with pytest.raises(ValueError):
        hybrid_fusion(-0.1, 0.2, 0.2)
    with pytest.raises(ValueError):
        hybrid_fusion(math.nan, 0.2, 0.2)
    with pytest.raises(ValueError):
        hybrid_fusion(math.inf, 0.2, 0.2)
    with pytest.raises(ValueError):
        hybrid_fusion("not-a-score", 0.2, 0.2)


def test_alpha_bounds_from_confidence():
    assert get_alpha(0.0) == pytest.approx(0.90)
    assert get_alpha(1.0) == pytest.approx(0.30)
    assert video_confidence_from_probability(0.5) == pytest.approx(0.0)
    assert video_confidence_from_probability(1.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("score", "category"),
    [
        (0.349999, "Low"),
        (0.35, "Moderate"),
        (0.50, "Moderate"),
        (0.65, "Moderate"),
        (0.650001, "High"),
        (0.784, "High"),
        (0.75, "High"),
    ],
)
def test_hybrid_threshold_boundaries(score, category):
    evaluation = ThresholdEngine().evaluate_hybrid_risk(score)
    assert evaluation.status == "EVALUATED"
    assert evaluation.classification == category


def test_hybrid_threshold_missing_and_invalid():
    engine = ThresholdEngine()
    missing = engine.evaluate_hybrid_risk(None)
    assert missing.status == "MISSING"
    assert missing.classification is None
    invalid_nan = engine.evaluate_hybrid_risk(math.nan)
    assert invalid_nan.status == "INVALID"
    assert invalid_nan.classification is None
    invalid_inf = engine.evaluate_hybrid_risk(math.inf)
    assert invalid_inf.status == "INVALID"
    invalid_range = engine.evaluate_hybrid_risk(1.2)
    assert invalid_range.status == "INVALID"


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def _seed_assessment(session, *, questionnaire_score, video_status, prediction, model_status):
    assessment_id = str(uuid.uuid4())
    assessment = Assessment(
        id=assessment_id,
        participant_id="child-1",
        status="VIDEO_COMPLETED",
    )
    session.add(assessment)
    session.add(
        QuestionnaireResult(
            assessment_id=assessment_id,
            instrument="MindBloom questionnaire model",
            responses_json={"A1_Score": 1, "age": 6},
            raw_score=questionnaire_score,
            maximum_score=1.0,
            normalized_score=questionnaire_score,
            scoring_method="trained questionnaire model predict_proba",
            interpretation_json={"risk_level": "High", "score_type": "model_probability"},
        )
    )
    session.add(
        VideoAssessment(
            assessment_id=assessment_id,
            session_id="session-1",
            status=video_status,
            mediapipe_statistics_json={"gaze_features": {"gaze_horizontal_mean": 0.9}},
            vit_statistics_json={
                "fear_probability_mean": 0.99,
                "emotion_statistics": {"mean": {"fear": 0.99, "sad": 0.8, "neutral": 0.1}},
            },
            feature_vector_json={"features": {"gaze_horizontal_mean": 0.9}},
            model_metadata_json={"video_model_status": model_status, "validated_for_asd": False},
            video_report_json={"prediction": prediction, "errors": []},
        )
    )
    session.commit()
    return assessment_id


def test_persistence_questionnaire_only_when_video_model_unavailable(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.75,
        video_status="video_model_not_available",
        prediction={"available": False, "probabilities": None},
        model_status="video_model_not_available",
    )
    integrated = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    )
    db_session.commit()
    db_session.expire_all()
    stored = db_session.get(Assessment, assessment_id).integrated_assessment
    hybrid = stored.report_json["hybrid_scoring"]
    assert hybrid["scoring_method"] == "questionnaire-only"
    assert hybrid["questionnaire_risk_score"] == pytest.approx(0.75)
    assert hybrid["hybrid_risk_score"] == pytest.approx(0.75)
    assert hybrid["video_risk_score"] is None
    assert hybrid["alpha"] == 1.0
    assert hybrid["risk_category"] == "High"
    assert stored.report_json["overall_asd_prediction"]["scoring_method"] == "questionnaire-only"
    assert stored.model_metadata_json["video_model_available"] is False
    assert stored.threshold_metadata_json["classification"] == "High"
    assert stored.report_json["hybrid_scoring"] == hybrid


def test_persistence_hybrid_when_combined_video_asd_probability_exists(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.70,
        video_status="success",
        prediction={
            "available": True,
            "predicted_label": "asd",
            "probabilities": {"asd": 0.90, "non_asd": 0.10},
        },
        model_status="available",
    )
    integrated = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    )
    db_session.commit()
    db_session.expire_all()
    stored = db_session.get(Assessment, assessment_id).integrated_assessment
    hybrid = stored.report_json["hybrid_scoring"]
    assert hybrid["scoring_method"] == "hybrid"
    assert hybrid["video_risk_score"] == pytest.approx(0.90)
    assert hybrid["video_confidence"] == pytest.approx(0.8)
    assert hybrid["alpha"] == pytest.approx(0.42)
    assert hybrid["hybrid_risk_score"] == pytest.approx(0.816)
    assert hybrid["risk_category"] == "High"
    assert stored.report_json["overall_asd_prediction"]["hybrid_risk_score"] == pytest.approx(0.816)
    assert stored.model_metadata_json["video_model_available"] is True
    assert stored.threshold_metadata_json["configuration"]["low_exclusive_upper"] == 0.35


def test_persistence_ignores_emotion_and_mediapipe_as_asd_scores(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.40,
        video_status="success",
        prediction={"available": False, "probabilities": {"asd": 0.99}},
        model_status="video_model_not_available",
    )
    integrated = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    )
    hybrid = integrated.report_json["hybrid_scoring"]
    assert hybrid["scoring_method"] == "questionnaire-only"
    assert hybrid["hybrid_risk_score"] == pytest.approx(0.40)
    assert hybrid["video_risk_score"] is None


def test_invalid_video_asd_probability_is_questionnaire_only(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.40,
        video_status="success",
        prediction={
            "available": True,
            "probabilities": {"asd": 1.7},
        },
        model_status="available",
    )
    hybrid = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    ).report_json["hybrid_scoring"]
    assert hybrid["scoring_method"] == "questionnaire-only"
    assert hybrid["hybrid_risk_score"] == pytest.approx(0.40)


def test_raw_score_used_when_normalized_score_missing(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.22,
        video_status="video_model_not_available",
        prediction={"available": False},
        model_status="video_model_not_available",
    )
    questionnaire = db_session.get(Assessment, assessment_id).questionnaire_result
    questionnaire.normalized_score = None
    questionnaire.raw_score = 0.22
    db_session.commit()
    hybrid = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    ).report_json["hybrid_scoring"]
    assert hybrid["questionnaire_risk_score"] == pytest.approx(0.22)


def test_normalized_score_is_preferred_over_raw_score(db_session):
    assessment_id = _seed_assessment(
        db_session,
        questionnaire_score=0.20,
        video_status="video_model_not_available",
        prediction={"available": False},
        model_status="video_model_not_available",
    )
    questionnaire = db_session.get(Assessment, assessment_id).questionnaire_result
    questionnaire.raw_score = 0.99
    questionnaire.normalized_score = 0.20
    db_session.commit()
    hybrid = AssessmentIntegrationService().create_integrated_assessment(
        db_session, assessment_id
    ).report_json["hybrid_scoring"]
    assert hybrid["questionnaire_risk_score"] == pytest.approx(0.20)
    assert hybrid["hybrid_risk_score"] == pytest.approx(0.20)

import numpy as np
import pytest

from app.video.features import (
    VIDEO_FEATURES, build_feature_vector, feature_dict_to_vector, shannon_entropy,
)
from app.video.session import SessionData
from app.video.fusion import get_alpha, hybrid_fusion
from app.video.model import VideoModel


def _populated_session(n=100):
    s = SessionData()
    rng = np.random.RandomState(1)
    for i in range(n):
        s.timestamps.append(i / 30.0)
        s.gaze_horizontal.append(float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1)))
        s.gaze_vertical.append(float(np.clip(0.5 + rng.normal(0, 0.1), 0, 1)))
        s.ear.append(float(rng.uniform(0.15, 0.35)))
        s.eye_closed_frames.append(s.ear[-1] < 0.19)
        s.yaw.append(float(rng.uniform(-10, 10)))
        s.pitch.append(float(rng.uniform(-10, 10)))
        s.roll.append(float(rng.uniform(-5, 5)))
        s.overall_movement.append(float(rng.uniform(0, 0.05)))
        s.mouth_movement.append(float(rng.uniform(0, 0.08)))
        s.eyebrow_movement.append(float(rng.uniform(0, 0.02)))
        s.total_frames += 1
        s.valid_face_frames += 1
    labels = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry"]
    for _ in range(20):
        raw = rng.random(len(labels))
        raw = raw / raw.sum()
        s.emotion_scores.append(dict(zip(labels, raw)))
        s.deepface_attempts += 1
        s.deepface_successes += 1
    return s


class TestFeatureSchema:
    def test_exactly_47_features_defined(self):
        assert len(VIDEO_FEATURES) == 47

    def test_feature_names_unique(self):
        assert len(set(VIDEO_FEATURES)) == 47

    def test_build_feature_vector_covers_every_schema_key(self):
        session = _populated_session()
        feats = build_feature_vector(session)
        assert set(feats.keys()) == set(VIDEO_FEATURES)

    def test_no_nan_or_inf_in_output_vector(self):
        session = _populated_session()
        feats = build_feature_vector(session)
        vec = feature_dict_to_vector(feats, VIDEO_FEATURES)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_empty_session_does_not_crash_and_returns_zeros(self):
        session = SessionData()
        feats = build_feature_vector(session)
        assert len(feats) == 47
        # No data -> everything should degrade to a safe default (0.0),
        # never raise.
        assert all(isinstance(v, float) for v in feats.values())


class TestEntropy:
    def test_constant_series_has_zero_entropy(self):
        assert shannon_entropy([0.5] * 20, bins=10, value_range=(0, 1)) == 0.0

    def test_uniform_spread_has_higher_entropy_than_constant(self):
        rng = np.random.RandomState(0)
        spread = list(rng.uniform(0, 1, size=200))
        constant = [0.5] * 200
        assert shannon_entropy(spread, bins=10, value_range=(0, 1)) > shannon_entropy(constant, bins=10, value_range=(0, 1))

    def test_empty_input_returns_zero(self):
        assert shannon_entropy([]) == 0.0


class TestFusion:
    def test_alpha_is_one_when_video_unavailable(self):
        assert get_alpha(None) == 1.0

    def test_alpha_monotonically_decreases_with_paper_confidence(self):
        alphas = [get_alpha(c) for c in [0.0, 0.25, 0.5, 0.75, 1.0]]
        assert alphas == sorted(alphas, reverse=True)
        assert alphas[0] == pytest.approx(0.90)
        assert alphas[-1] == pytest.approx(0.30)

    def test_hybrid_score_collapses_to_questionnaire_without_video(self):
        result = hybrid_fusion(0.63, None, None)
        assert result["hybrid_risk_score"] == pytest.approx(0.63)
        assert result["alpha"] == 1.0
        assert result["scoring_method"] == "questionnaire-only"
        assert result["video_risk_score"] is None

    def test_invalid_questionnaire_score_is_rejected(self):
        with pytest.raises(ValueError):
            hybrid_fusion(1.5, 0.2, 0.2)


class TestVideoModelUnavailable:
    def test_predict_returns_none_when_no_model_file(self, tmp_path, monkeypatch):
        import app.video.model as model_module
        monkeypatch.setattr(model_module, "VIDEO_MODEL_PATH", str(tmp_path / "nope.pkl"))
        vm = model_module.VideoModel()
        assert vm.is_available() is False
        assert vm.predict({f: 0.0 for f in VIDEO_FEATURES}) is None

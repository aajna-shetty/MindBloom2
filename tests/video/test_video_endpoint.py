import asyncio
import json
from types import SimpleNamespace

import httpx

from app import main
from app.database import get_db
from app.video.service import VideoAnalysisResult


class _FakeQuery:
    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return SimpleNamespace(id=1)


class _FakeDb:
    def query(self, *_args, **_kwargs):
        return _FakeQuery()

    def add(self, _value):
        pass

    def commit(self):
        pass

    def flush(self):
        pass

    def refresh(self, _value):
        pass

    def rollback(self):
        pass


def _request(method, url, **kwargs):
    async def send():
        main.app.dependency_overrides[get_db] = lambda: _FakeDb()
        try:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.request(method, url, **kwargs)
        finally:
            main.app.dependency_overrides.pop(get_db, None)

    return asyncio.run(send())


def test_rejects_non_numeric_webcam_source(monkeypatch):
    response = _request(
        "POST",
        "/video/analyze",
        json={"child_id": 1, "source": "C:\\private\\video.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_video_source"


def test_rejects_unsupported_upload_extension():
    response = _request(
        "POST",
        "/video/analyze",
        data={"child_id": "1"},
        files={"video": ("capture.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "unsupported_video_format"


def test_rejects_oversized_upload(monkeypatch):
    monkeypatch.setattr(main, "MAX_VIDEO_UPLOAD_BYTES", 4)

    response = _request(
        "POST",
        "/video/analyze",
        data={"child_id": "1"},
        files={"video": ("capture.mp4", b"12345", "video/mp4")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "video_upload_too_large"


def test_rejects_unreadable_upload(monkeypatch):
    class _Probe:
        def isOpened(self):
            return False

        def release(self):
            pass

    monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: _Probe())

    response = _request(
        "POST",
        "/video/analyze",
        data={"child_id": "1"},
        files={"video": ("capture.mp4", b"not a video", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_video"


def test_processes_upload_and_removes_temporary_file(monkeypatch, tmp_path):
    created_paths = []

    real_named_temporary_file = main.tempfile.NamedTemporaryFile

    def named_temporary_file(*args, **kwargs):
        result = real_named_temporary_file(*args, dir=tmp_path, **kwargs)
        created_paths.append(result.name)
        return result

    class _Probe:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            pass

    monkeypatch.setattr(main.tempfile, "NamedTemporaryFile", named_temporary_file)
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: _Probe())
    monkeypatch.setattr(
        main,
        "run_video_session",
        lambda **_kwargs: VideoAnalysisResult(status="success", feature_count=47),
    )

    response = _request(
        "POST",
        "/video/analyze",
        data={"child_id": "1"},
        files={"video": ("capture.mp4", b"valid enough for mocked probe", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["combined_features"] == {}
    assert created_paths
    assert not tmp_path.joinpath(created_paths[0].split("\\")[-1]).exists()


def _questionnaire_responses():
    return {f"A{i}_Score": 0 for i in range(1, 11)}


def test_end_to_end_assessment_returns_separate_questionnaire_and_video_sections(monkeypatch):
    class _Scaler:
        def transform(self, values):
            assert len(values[0]) == 32
            return values

    class _Model:
        def predict_proba(self, _values):
            return [[0.25, 0.75]]

    monkeypatch.setitem(main.app_state, "model", _Model())
    monkeypatch.setitem(main.app_state, "scaler", _Scaler())
    monkeypatch.setattr(
        main.AssessmentIntegrationService,
        "create_integrated_assessment",
        lambda *_args: SimpleNamespace(status="COMPLETED", report_json={}),
    )
    monkeypatch.setattr(
        main.CombinedVideoAssessmentService,
        "run_and_persist",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="video_model_not_available",
            session_id="session-1",
            duration_seconds=1.0,
            total_frames=10,
            valid_frames=9,
            model_metadata_json={"video_model_status": "video_model_not_available"},
            mediapipe_statistics_json={"total_frames": 10, "valid_frames": 9, "no_face_frames": 1},
            vit_statistics_json={"frames_analyzed": 2},
            feature_vector_json={"feature_count": 47, "valid": True},
            performance_json={"average_mediapipe_latency_ms": 10.0},
            video_report_json={"prediction": {}, "errors": [], "detail": None},
        ),
    )

    response = _request(
        "POST",
        "/api/v1/mindbloom/assessments",
        json={
            "participant_id": "session-abc",
            "age": 6,
            "questionnaire_responses": _questionnaire_responses(),
            "source": "0",
            "duration_seconds": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["questionnaire"]["score"] == 0.75
    assert body["video_assessment"]["model_validated_for_asd"] is False
    assert body["video_assessment"]["unified_video_score"] is None
    assert body["overall_summary"]["clinical_interpretation"] is None
    assert body["timing"]["feature_fusion_ms"] is None


def test_end_to_end_assessment_accepts_multipart_questionnaire_and_video(monkeypatch, tmp_path):
    class _Scaler:
        def transform(self, values):
            return values

    class _Model:
        def predict_proba(self, _values):
            return [[0.5, 0.5]]

    class _Probe:
        def isOpened(self):
            return True

        def read(self):
            return True, object()

        def release(self):
            pass

    monkeypatch.setitem(main.app_state, "model", _Model())
    monkeypatch.setitem(main.app_state, "scaler", _Scaler())
    monkeypatch.setattr(
        main.AssessmentIntegrationService,
        "create_integrated_assessment",
        lambda *_args: SimpleNamespace(status="COMPLETED", report_json={}),
    )
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda _path: _Probe())
    monkeypatch.setattr(
        main.CombinedVideoAssessmentService,
        "run_and_persist",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="video_model_not_available",
            session_id="session-upload",
            duration_seconds=1.0,
            total_frames=1,
            valid_frames=1,
            model_metadata_json={"video_model_status": "video_model_not_available"},
            mediapipe_statistics_json={"total_frames": 1, "valid_frames": 1, "no_face_frames": 0},
            vit_statistics_json={},
            feature_vector_json={"feature_count": 47},
            performance_json={},
            video_report_json={"prediction": {}, "errors": [], "detail": None},
        ),
    )

    response = _request(
        "POST",
        "/api/v1/mindbloom/assessments",
        data={
            "participant_id": "session-upload",
            "age": "5",
            "questionnaire_responses": json.dumps(_questionnaire_responses()),
            "duration_seconds": "1",
        },
        files={"video": ("capture.mp4", b"video-bytes", "video/mp4")},
    )

    assert response.status_code == 200
    assert response.json()["participant_id"] == "session-upload"


def test_end_to_end_assessment_rejects_incomplete_questionnaire(monkeypatch):
    monkeypatch.setitem(main.app_state, "model", object())
    monkeypatch.setitem(main.app_state, "scaler", object())

    response = _request(
        "POST",
        "/api/v1/mindbloom/assessments",
        json={
            "participant_id": "session-invalid",
            "age": 6,
            "questionnaire_responses": {"A1_Score": 1},
            "source": "0",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "missing_questionnaire_responses"

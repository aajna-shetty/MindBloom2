import os
import sys
import logging
import tempfile
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List

import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import desc


# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core Machine Learning Pipelines
from app.model import load_best_model, load_metrics, load_feature_importance
from app.preprocess import load_scaler, get_feature_list
from app.utils import risk_level, top_features, combine_scores
from app.questionnaire import build_feature_vector_from_answers, QUESTIONS

# Relational Storage Engine
from app.database import (
    engine,
    Base,
    get_db,
    Assessment,
    ChildProfile,
    CognitiveGameLog,
    IntegratedAssessment,
    ensure_sqlite_schema,
    QuestionnaireResult,
    ScreeningTransaction,
    VideoAssessment,
)
from app.assessment_services import (
    AssessmentIntegrationService,
    CombinedVideoAssessmentService,
    QuestionnaireService,
    create_assessment as create_persisted_assessment,
)
from app.thresholds import HYBRID_THRESHOLD_CONFIGURATION


from app.video.service import VideoAnalysisResult, run_video_session
from app.video.schemas import (
    AssessmentResponse,
    VideoAnalyzeRequest,
    VideoAnalyzeResponse,
)
from app.video.config import ALLOWED_VIDEO_EXTENSIONS, MAX_VIDEO_UPLOAD_BYTES

# Initialize relational schemas seamlessly in the terminal on boot
try:
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_schema()
    print("\n[Database] Success: SQLite structural tables synchronized.\n")
except Exception as db_err:
    print(f"\n[Database] Critical connection error: {str(db_err)}\n")

# LOGGING CONFIGURATION
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app_state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artefacts ...")
    try:
        app_state["model"] = load_best_model()
        app_state["scaler"] = load_scaler()
        app_state["feature_list"] = get_feature_list()
        app_state["metrics"] = load_metrics()
        app_state["feature_import"] = load_feature_importance()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed loading artifacts: {str(e)}")
        app_state["model"] = None
    yield
    logger.info("Shutting down...")

app = FastAPI(
    title="Autism Early Risk Screening API",
    description="Multimodal autism screening using questionnaire + ML with active DB memory",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================================
# PYDANTIC REQ / RESP COMMUNICATORS
# ========================================================

class ProfileCreateInput(BaseModel):
    name: str
    age_months: int
    sex: str
    family_history_asd: bool = False
    jaundice_history: bool = False

class PredictRequest(BaseModel):
    child_id: int  # Linked directly to profile tracking tables
    features: Dict[str, Any]
    video_score: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)

class FeatureItem(BaseModel):
    feature: str
    importance: float

class PredictResponse(BaseModel):
    questionnaire_score: float
    final_score: float
    risk_level: str
    top_features: List[FeatureItem]
    explanation: str

class GameLogInput(BaseModel):
    game_name: str
    accuracy: float
    completion_time_sec: float
    reaction_time_ms: float

# ========================================================
# FUNCTIONAL SERVER GATEWAYS
# ========================================================

@app.get("/health")
def health():
    return {
        "status": "ok" if app_state.get("model") else "model_not_loaded",
        "model_ready": app_state.get("model") is not None,
    }


def _score_assessment_questionnaire(
    age: float,
    responses: Dict[str, Any],
) -> Dict[str, Any]:
    required_items = [f"A{i}_Score" for i in range(1, 11)]
    missing = [key for key in required_items if key not in responses]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "missing_questionnaire_responses",
                "message": "All A1_Score through A10_Score responses are required.",
                "fields": missing,
            },
        )

    normalized = dict(responses)
    normalized["age"] = age
    for key in required_items:
        try:
            value = float(normalized[key])
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_questionnaire_response",
                    "message": f"{key} must be 0 or 1.",
                },
            ) from exc
        if value not in (0.0, 1.0):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_questionnaire_response",
                    "message": f"{key} must be 0 or 1.",
                },
            )
        normalized[key] = value

    if app_state.get("model") is None or app_state.get("scaler") is None:
        raise HTTPException(status_code=503, detail="Questionnaire model not loaded.")

    try:
        vector = build_feature_vector_from_answers(normalized)
        scaled = app_state["scaler"].transform([vector])
        score = float(app_state["model"].predict_proba(scaled)[0][1])
        interpretation = risk_level(score)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "questionnaire_scoring_failed",
                "message": str(exc),
            },
        ) from exc

    return {
        "instrument": "MindBloom questionnaire model",
        "score": round(score, 4),
        "interpretation": interpretation,
        "screening_outcome": interpretation,
        "validated_instrument": False,
        "limitations": [
            "The repository does not identify this instrument as a clinically validated questionnaire."
        ],
    }

@app.get("/questions")
def get_questions():
    return {"total": len(QUESTIONS), "questions": QUESTIONS}

@app.post("/profiles/", status_code=status.HTTP_201_CREATED)
def register_child_file(profile: ProfileCreateInput, db: Session = Depends(get_db)):
    db_profile = ChildProfile(**profile.model_dump())
    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)
    logger.info(f"Terminal Event: Registered profile successfully for ID {db_profile.id}")
    return {"status": "Success", "child_id": db_profile.id, "name": db_profile.name}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    if app_state.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    child = db.query(ChildProfile).filter(ChildProfile.id == request.child_id).first()
    if not child:
        raise HTTPException(status_code=444, detail="Child profile execution requires active ID log.")

    model = app_state["model"]
    scaler = app_state["scaler"]
    feature_import = app_state.get("feature_import", {})

    try:
        X_raw = build_feature_vector_from_answers(request.features)
        X_scaled = scaler.transform([X_raw])
        q_score = float(model.predict_proba(X_scaled)[0][1])
        final_score = combine_scores(q_score, request.video_score)
        risk = risk_level(final_score)

        top_fi = []
        if feature_import:
            for f in top_features(feature_import, n=3):
                top_fi.append(FeatureItem(**f))

        explanation = (
            f"Risk level is {risk} with probability {final_score:.2f}. "
            f"Questionnaire score contributed {q_score:.2f} and video score {request.video_score:.2f}. "
        )

        # Permanent transactional archival commit
        screening_record = ScreeningTransaction(
            child_id=request.child_id,
            questionnaire_score=round(q_score, 4),
            video_score=round(request.video_score, 4),
            hybrid_risk_score=round(final_score, 4),
            risk_tier=risk
        )
        db.add(screening_record)
        db.commit()

        logger.info(f"Terminal Event: Screening saved for ID {request.child_id} -> Result: {risk}")

        return PredictResponse(
            questionnaire_score=round(q_score, 4),
            final_score=round(final_score, 4),
            risk_level=risk,
            top_features=top_fi,
            explanation=explanation
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.post(
    "/video/analyze",
    response_model=VideoAnalyzeResponse,
    responses={
        400: {"description": "Invalid webcam source."},
        404: {"description": "Child profile not found."},
        415: {"description": "Unsupported video format."},
        422: {"description": "Invalid video or insufficient analysis data."},
        413: {"description": "Video upload exceeds the configured limit."},
        503: {"description": "Camera or processing dependency unavailable."},
        504: {"description": "Video processing exceeded its timeout."},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/VideoAnalyzeRequest"},
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["child_id"],
                        "properties": {
                            "child_id": {"type": "integer"},
                            "source": {"type": "string", "default": "0"},
                            "duration_seconds": {"type": "integer"},
                            "video": {
                                "type": "string",
                                "format": "binary",
                                "description": "Temporary video upload.",
                            },
                        },
                    },
                },
            },
        }
    },
)
async def analyze_video(request: Request, db: Session = Depends(get_db)):
    """
    Run the complete video analysis pipeline.

    Webcam/upload
        ↓
    OpenCV
        ↓
    MediaPipe
        ↓
    ViT emotion analysis
        ↓
    47-dimensional feature vector
        ↓
    Video classifier availability check
    """

    form_upload = None
    temp_path = None
    try:
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            raw_child_id = form.get("child_id")
            raw_source = form.get("source", "0")
            raw_duration = form.get("duration_seconds")
            form_upload = form.get("video")
            if raw_child_id is None:
                raise HTTPException(status_code=422, detail={
                    "code": "missing_child_id",
                    "message": "child_id is required.",
                })
            try:
                request_data = VideoAnalyzeRequest(
                    child_id=int(raw_child_id),
                    source=str(raw_source or "0"),
                    duration_seconds=(
                        int(raw_duration) if raw_duration not in (None, "") else None
                    ),
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=422, detail={
                    "code": "invalid_request",
                    "message": str(exc),
                }) from exc
        else:
            try:
                request_data = VideoAnalyzeRequest.model_validate(await request.json())
            except Exception as exc:
                raise HTTPException(status_code=422, detail={
                    "code": "invalid_request",
                    "message": "Expected JSON with child_id and a webcam source.",
                }) from exc

        if request_data.duration_seconds is not None and not 1 <= request_data.duration_seconds <= 3600:
            raise HTTPException(status_code=422, detail={
                "code": "invalid_duration",
                "message": "duration_seconds must be between 1 and 3600.",
            })

        if form_upload is not None:
            filename = getattr(form_upload, "filename", "") or ""
            extension = os.path.splitext(filename)[1].lower()
            if extension not in ALLOWED_VIDEO_EXTENSIONS:
                raise HTTPException(status_code=415, detail={
                    "code": "unsupported_video_format",
                    "message": f"Supported formats: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
                })
            temp_file = tempfile.NamedTemporaryFile(
                mode="wb", suffix=extension, prefix="mindbloom_video_", delete=False
            )
            temp_path = temp_file.name
            total_bytes = 0
            try:
                while True:
                    chunk = await form_upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail={
                            "code": "video_upload_too_large",
                            "message": f"Maximum upload size is {MAX_VIDEO_UPLOAD_BYTES} bytes.",
                        })
                    temp_file.write(chunk)
            finally:
                temp_file.close()
                await form_upload.close()

            probe = cv2.VideoCapture(temp_path)
            opened = probe.isOpened()
            has_frame, _ = probe.read() if opened else (False, None)
            probe.release()
            if not opened or not has_frame:
                raise HTTPException(status_code=422, detail={
                    "code": "invalid_video",
                    "message": "The uploaded file is not a readable video.",
                })
            source = temp_path
        else:
            source = request_data.source.strip()
            if not source.isdigit():
                raise HTTPException(status_code=400, detail={
                    "code": "invalid_video_source",
                    "message": "Only a server webcam index is allowed without an upload.",
                })
            source = int(source)

        child = (
            db.query(ChildProfile)
            .filter(ChildProfile.id == request_data.child_id)
            .first()
        )

        if not child:
            raise HTTPException(
                status_code=404,
                detail=f"Child profile {request_data.child_id} not found."
            )

        logger.info(
            "Starting video analysis for child ID %s",
            request_data.child_id
        )

        result = run_video_session(
            source=source,
            duration_seconds=request_data.duration_seconds or 300,
            use_vit=True,
        )
        status_by_result = {
            "camera_unavailable": 503,
            "error": 503,
            "dependency_unavailable": 503,
            "processing_timeout": 504,
            "insufficient_video_data": 422,
            "invalid_feature_vector": 500,
            "processing_error": 500,
        }
        if result.status in status_by_result:
            raise HTTPException(
                status_code=status_by_result[result.status],
                detail={
                    "code": result.status,
                    "message": result.detail or result.status,
                    "session_id": result.session_id,
                    "errors": result.errors,
                },
            )

        return VideoAnalyzeResponse(
            status=result.status,
            session_id=result.session_id,
            video_model_status=result.video_model_status,
            mediapipe=result.mediapipe,
            vit=result.vit,
            combined_features=result.combined_features,
            performance=result.performance,
            prediction=result.prediction,
            errors=result.errors,
            video_score=result.video_score,
            video_confidence=result.video_confidence,
            predicted_class=result.predicted_class,
            session_duration_seconds=result.session_duration_seconds,
            valid_frame_ratio=result.valid_frame_ratio,
            feature_count=result.feature_count,
            detail=result.detail,
        )
    finally:
        if form_upload is not None:
            await form_upload.close()
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


@app.post(
    "/api/v1/mindbloom/assessments",
    response_model=AssessmentResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"description": "Invalid assessment metadata or questionnaire responses."},
        413: {"description": "Video upload exceeds the configured limit."},
        415: {"description": "Unsupported video format."},
        503: {"description": "Required model or video dependency is unavailable."},
    },
)
async def create_assessment(request: Request, db: Session = Depends(get_db)):
    """Run questionnaire scoring and the complete MediaPipe + ViT workflow."""
    assessment_id = str(uuid.uuid4())
    started = time.perf_counter()
    temp_path = None
    upload_started = time.perf_counter()
    form_upload = None

    try:
        content_type = request.headers.get("content-type", "").lower()
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            participant_id = str(form.get("participant_id", "")).strip()
            raw_age = form.get("age")
            raw_responses = form.get("questionnaire_responses")
            raw_source = str(form.get("source", "0") or "0").strip()
            raw_duration = form.get("duration_seconds")
            form_upload = form.get("video")
            metadata = form.get("metadata", "{}")
        else:
            payload = await request.json()
            participant_id = str(payload.get("participant_id", "")).strip()
            raw_age = payload.get("age")
            raw_responses = payload.get("questionnaire_responses")
            raw_source = str(payload.get("source", "0") or "0").strip()
            raw_duration = payload.get("duration_seconds")
            metadata = payload.get("metadata", {})

        if not participant_id:
            raise HTTPException(
                status_code=422,
                detail={"code": "missing_participant_id", "message": "participant_id is required."},
            )
        try:
            age = float(raw_age)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_age", "message": "age must be numeric."},
            ) from exc
        if not 0 < age <= 120:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_age", "message": "age must be between 0 and 120."},
            )

        if isinstance(raw_responses, str):
            try:
                questionnaire_responses = json.loads(raw_responses)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_questionnaire_json",
                        "message": "questionnaire_responses must contain valid JSON.",
                    },
                ) from exc
        else:
            questionnaire_responses = raw_responses
        if not isinstance(questionnaire_responses, dict):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "invalid_questionnaire_responses",
                    "message": "questionnaire_responses must be an object.",
                },
            )

        try:
            duration_seconds = int(raw_duration) if raw_duration not in (None, "") else 300
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_duration", "message": "duration_seconds must be an integer."},
            ) from exc
        if not 1 <= duration_seconds <= 3600:
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_duration", "message": "duration_seconds must be between 1 and 3600."},
            )

        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError as exc:
                raise HTTPException(
                    status_code=422,
                    detail={"code": "invalid_metadata", "message": "metadata must contain valid JSON."},
                ) from exc
        if not isinstance(metadata, dict):
            raise HTTPException(
                status_code=422,
                detail={"code": "invalid_metadata", "message": "metadata must be an object."},
            )

        assessment = create_persisted_assessment(db, participant_id)
        questionnaire_started = time.perf_counter()
        if app_state.get("model") is None or app_state.get("scaler") is None:
            raise HTTPException(status_code=503, detail="Questionnaire model not loaded.")
        try:
            questionnaire_result = QuestionnaireService(
                app_state["model"], app_state["scaler"]
            ).submit_responses(
                db, assessment.id, age, questionnaire_responses
            )
        except (TypeError, ValueError) as exc:
            code = (
                "missing_questionnaire_responses"
                if str(exc).startswith("Missing questionnaire responses")
                else "invalid_questionnaire_responses"
            )
            raise HTTPException(
                status_code=422,
                detail={"code": code, "message": str(exc)},
            ) from exc
        assessment.status = "QUESTIONNAIRE_COMPLETED"
        questionnaire = {
            "instrument": questionnaire_result.instrument,
            "score": questionnaire_result.raw_score,
            "interpretation": questionnaire_result.interpretation_json,
            "maximum_score": questionnaire_result.maximum_score,
            "normalized_score": questionnaire_result.normalized_score,
            "scoring_method": questionnaire_result.scoring_method,
            "threshold": questionnaire_result.threshold_metadata_json,
        }
        questionnaire_ms = (time.perf_counter() - questionnaire_started) * 1000.0
        db.commit()
        upload_ms = (time.perf_counter() - upload_started) * 1000.0

        if form_upload is not None:
            filename = getattr(form_upload, "filename", "") or ""
            extension = os.path.splitext(filename)[1].lower()
            if extension not in ALLOWED_VIDEO_EXTENSIONS:
                raise HTTPException(
                    status_code=415,
                    detail={
                        "code": "unsupported_video_format",
                        "message": f"Supported formats: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
                    },
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=extension, prefix="mindbloom_assessment_", delete=False
            ) as temp_file:
                temp_path = temp_file.name
                total_bytes = 0
                while True:
                    chunk = await form_upload.read(1024 * 1024)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_VIDEO_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail={
                                "code": "video_upload_too_large",
                                "message": f"Maximum upload size is {MAX_VIDEO_UPLOAD_BYTES} bytes.",
                            },
                        )
                    temp_file.write(chunk)
            source = temp_path
        else:
            if not raw_source.isdigit():
                raise HTTPException(
                    status_code=422,
                    detail={
                        "code": "invalid_video_source",
                        "message": "source must be a webcam index when video is not uploaded.",
                    },
                )
            source = int(raw_source)

        video_started = time.perf_counter()
        assessment.status = "VIDEO_PROCESSING"
        db.commit()
        video_record = CombinedVideoAssessmentService().run_and_persist(
            db, assessment.id, source, duration_seconds, use_vit=True
        )
        video_result = VideoAnalysisResult(
            status=video_record.status,
            session_id=video_record.session_id,
            video_model_status=video_record.model_metadata_json.get(
                "video_model_status", "video_model_not_available"
            ) if video_record.model_metadata_json else "video_model_not_available",
            session_duration_seconds=video_record.duration_seconds,
            valid_frame_ratio=(
                video_record.valid_frames / video_record.total_frames
                if video_record.total_frames else 0.0
            ),
            feature_count=(
                video_record.feature_vector_json.get("feature_count")
                if video_record.feature_vector_json else None
            ),
            mediapipe=video_record.mediapipe_statistics_json,
            vit=video_record.vit_statistics_json,
            combined_features=video_record.feature_vector_json or {},
            performance=video_record.performance_json or {},
            video_score=(
                video_record.video_report_json.get("prediction", {}).get("probabilities", {}).get("asd")
                if video_record.video_report_json else None
            ),
            prediction=video_record.video_report_json.get("prediction", {})
            if video_record.video_report_json else {},
            errors=video_record.video_report_json.get("errors", [])
            if video_record.video_report_json else [],
            detail=video_record.video_report_json.get("detail")
            if video_record.video_report_json else None,
        )
        video_ms = (time.perf_counter() - video_started) * 1000.0
        db.commit()
        video_status = "COMPLETED" if video_result.status in (
            "success",
            "video_model_not_available",
        ) else "PARTIAL"
        limitations = [
            "No validated video ASD classifier is available; video observations are not an ASD risk score.",
        ]
        if video_result.video_model_status == "video_model_not_available":
            limitations.append("Unified 47-feature video representation was generated but not clinically validated for ASD.")

        report_started = time.perf_counter()
        video_assessment = {
            "status": video_status,
            "method": "MediaPipe + ViT feature extraction",
            "model_validated_for_asd": video_result.video_model_status == "available",
            "unified_video_score": video_result.video_score,
            "score_type": "OBSERVATIONAL_INDEX" if video_result.video_score is None else "MODEL_OUTPUT",
            "data_quality": {
                "total_frames": video_result.mediapipe.get("total_frames", 0),
                "valid_frames": video_result.mediapipe.get("valid_frames", 0),
                "missing_frame_percentage": None,
                "valid_frame_ratio": video_result.valid_frame_ratio,
            },
            "observations": {
                "mediapipe": video_result.mediapipe,
                "vit": video_result.vit,
                "combined_features": video_result.combined_features,
                "prediction": video_result.prediction,
                "performance": video_result.performance,
            },
            "limitations": limitations,
        }
        errors = list(video_result.errors)
        if video_result.detail:
            errors.append({"code": video_result.status, "message": video_result.detail})
        report_ms = (time.perf_counter() - report_started) * 1000.0
        overall_status = "COMPLETED" if video_status == "COMPLETED" else "PARTIAL"
        if overall_status == "COMPLETED":
            assessment.status = "VIDEO_COMPLETED"
            integrated = AssessmentIntegrationService().create_integrated_assessment(
                db, assessment.id
            )
            db.commit()
        else:
            assessment.status = "FAILED"
            db.commit()
            integrated = None
        report = getattr(integrated, "report_json", None) or {}
        hybrid = report.get("hybrid_scoring") if isinstance(report, dict) else None
        if not isinstance(hybrid, dict):
            hybrid = {}
        overall_prediction = (
            report.get("overall_asd_prediction") if isinstance(report, dict) else None
        )
        if not isinstance(overall_prediction, dict):
            overall_prediction = {}
        return AssessmentResponse(
            assessment_id=assessment.id,
            status=overall_status,
            participant_id=participant_id,
            questionnaire=questionnaire,
            video_assessment=video_assessment,
            overall_summary={
                "questionnaire_result": questionnaire,
                "video_result": {
                    "status": video_status,
                    "model_validated_for_asd": video_assessment["model_validated_for_asd"],
                },
                "clinical_interpretation": overall_prediction.get("risk_category"),
                "questionnaire_risk_score": hybrid.get("questionnaire_risk_score"),
                "video_risk_score": hybrid.get("video_risk_score"),
                "video_confidence": hybrid.get("video_confidence"),
                "alpha": hybrid.get("alpha"),
                "hybrid_risk_score": hybrid.get("hybrid_risk_score"),
                "risk_category": hybrid.get("risk_category"),
                "scoring_method": hybrid.get("scoring_method"),
                "video_model_available": hybrid.get("video_model_available"),
                "thresholds": hybrid.get("thresholds") or dict(HYBRID_THRESHOLD_CONFIGURATION),
                "recommendation": "Professional assessment where indicated.",
                "metadata": metadata,
            },
            timing={
                "video_upload_ms": round(upload_ms, 2),
                "questionnaire_scoring_ms": round(questionnaire_ms, 2),
                "video_processing_ms": round(video_ms, 2),
                "feature_fusion_ms": None,
                "model_inference_ms": None,
                "report_generation_ms": round(report_ms, 2),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 2),
            },
            errors=errors,
        )
    finally:
        if form_upload is not None:
            await form_upload.close()
        if temp_path:
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass


@app.post("/api/v1/assessments", status_code=status.HTTP_201_CREATED)
def create_assessment_record(
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
):
    participant_id = str(payload.get("participant_id", "")).strip()
    if not participant_id:
        raise HTTPException(
            status_code=422,
            detail={"code": "missing_participant_id", "message": "participant_id is required."},
        )
    assessment = create_persisted_assessment(db, participant_id)
    return {
        "assessment_id": assessment.id,
        "participant_id": assessment.participant_id,
        "status": assessment.status,
        "created_at": assessment.created_at,
    }


@app.get("/api/v1/assessments/{assessment_id}/questionnaire")
def get_assessment_questions(assessment_id: str, db: Session = Depends(get_db)):
    if db.get(Assessment, assessment_id) is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    return {"assessment_id": assessment_id, "total": len(QUESTIONS), "questions": QUESTIONS}


@app.post(
    "/api/v1/mindbloom/assessments/{assessment_id}/questionnaire",
    response_model=None,
)
@app.post("/api/v1/assessments/{assessment_id}/questionnaire")
def submit_assessment_questionnaire(
    assessment_id: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    if app_state.get("model") is None or app_state.get("scaler") is None:
        raise HTTPException(status_code=503, detail="Questionnaire model not loaded.")
    try:
        age = float(payload["age"])
        if assessment.status != "QUESTIONNAIRE_PENDING":
            raise HTTPException(
                status_code=409,
                detail=f"Questionnaire cannot be submitted while assessment is {assessment.status}.",
            )
        responses = payload["questionnaire_responses"]
        result = QuestionnaireService(
            app_state["model"], app_state["scaler"]
        ).submit_responses(db, assessment_id, age, responses)
        assessment.status = "QUESTIONNAIRE_COMPLETED"
        db.commit()
        return {
            "assessment_id": assessment_id,
            "status": assessment.status,
            "questionnaire": {
                "instrument": result.instrument,
                "score": result.raw_score,
                "interpretation": result.interpretation_json,
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        db.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/mindbloom/assessments/{assessment_id}/video/start")
@app.post("/api/v1/assessments/{assessment_id}/video/start")
def start_assessment_video(
    assessment_id: str,
    payload: Dict[str, Any],
    db: Session = Depends(get_db),
):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    if assessment.questionnaire_result is None:
        raise HTTPException(
            status_code=409,
            detail="Questionnaire must be completed before video processing.",
        )
    if assessment.status == "VIDEO_PROCESSING":
        raise HTTPException(status_code=409, detail="A video session is already active.")
    if assessment.video_assessment is not None:
        raise HTTPException(
            status_code=409,
            detail="This assessment already has a video result; create a new assessment to retry.",
        )
    source = payload.get("source", 0)
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    duration = int(payload.get("duration_seconds", 15))
    assessment.status = "VIDEO_PROCESSING"
    db.commit()
    try:
        result = CombinedVideoAssessmentService().run_and_persist(
            db, assessment_id, source, duration, use_vit=True
        )
        assessment.status = (
            "VIDEO_COMPLETED"
            if result.status in {"success", "video_model_not_available"}
            else "FAILED"
        )
        db.commit()
    except Exception as exc:
        assessment.status = "FAILED"
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "assessment_id": assessment_id,
        "status": assessment.status,
        "session_id": result.session_id,
        "video_assessment": result.video_report_json,
        "processing_core": "app.video.service.run_video_session",
    }


@app.get("/api/v1/mindbloom/assessments/{assessment_id}/video/status")
@app.get("/api/v1/assessments/{assessment_id}/video/status")
def get_assessment_video_status(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    video = assessment.video_assessment
    return {
        "assessment_id": assessment_id,
        "status": assessment.status,
        "video": None if video is None else {
            "session_id": video.session_id,
            "status": video.status,
            "total_frames": video.total_frames,
            "valid_frames": video.valid_frames,
            "vit": video.vit_statistics_json,
        },
    }


@app.get("/api/v1/assessments/{assessment_id}/video/result")
def get_assessment_video_result(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    video = assessment.video_assessment
    if video is None:
        raise HTTPException(status_code=404, detail="Video assessment not found.")
    return {
        "assessment_id": assessment_id,
        "session_id": video.session_id,
        "status": video.status,
        "total_frames": video.total_frames,
        "valid_frames": video.valid_frames,
        "mediapipe": video.mediapipe_statistics_json,
        "vit": video.vit_statistics_json,
        "feature_vector": video.feature_vector_json,
        "report": video.video_report_json,
    }


@app.post("/api/v1/mindbloom/assessments/{assessment_id}/integrate")
@app.post("/api/v1/assessments/{assessment_id}/integrate")
def integrate_assessment(assessment_id: str, db: Session = Depends(get_db)):
    try:
        result = AssessmentIntegrationService().create_integrated_assessment(
            db, assessment_id
        )
        db.commit()
        return {
            "assessment_id": assessment_id,
            "status": result.status,
            "integrated_assessment": result.report_json,
        }
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/v1/mindbloom/assessments/{assessment_id}")
@app.get("/api/v1/assessments/{assessment_id}")
def get_assessment(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    questionnaire = assessment.questionnaire_result
    video = assessment.video_assessment
    integrated = assessment.integrated_assessment
    return {
        "assessment_id": assessment.id,
        "participant_id": assessment.participant_id,
        "status": assessment.status,
        "questionnaire": None if questionnaire is None else {
            "instrument": questionnaire.instrument,
            "score": questionnaire.raw_score,
            "interpretation": questionnaire.interpretation_json,
            "responses_saved": True,
        },
        "video_assessment": None if video is None else {
            "session_id": video.session_id,
            "status": video.status,
            "total_frames": video.total_frames,
            "valid_frames": video.valid_frames,
            "feature_vector": video.feature_vector_json,
            "report": video.video_report_json,
        },
        "integrated_assessment": None if integrated is None else {
            "status": integrated.status,
            "report": integrated.report_json,
        },
    }


@app.get("/api/v1/assessments/{assessment_id}/questionnaire/result")
def get_assessment_questionnaire_result(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    result = assessment.questionnaire_result
    if result is None:
        raise HTTPException(status_code=404, detail="Questionnaire result not found.")
    return {
        "assessment_id": assessment_id,
        "instrument": result.instrument,
        "responses": result.responses_json,
        "score": result.raw_score,
        "interpretation": result.interpretation_json,
    }


@app.get("/api/v1/assessments/{assessment_id}/integrated-result")
@app.get("/api/v1/assessments/{assessment_id}/report")
def get_integrated_assessment_report(assessment_id: str, db: Session = Depends(get_db)):
    assessment = db.get(Assessment, assessment_id)
    if assessment is None:
        raise HTTPException(status_code=404, detail="Assessment not found.")
    result = assessment.integrated_assessment
    if result is None:
        raise HTTPException(status_code=404, detail="Integrated assessment not found.")
    return {
        "assessment_id": assessment_id,
        "status": result.status,
        "report": result.report_json,
    }


@app.post("/profiles/{child_id}/game-session/")
def log_cognitive_game_metrics(child_id: int, current_log: GameLogInput, db: Session = Depends(get_db)):
    child = db.query(ChildProfile).filter(ChildProfile.id == child_id).first()
    if not child:
        raise HTTPException(status_code=404, detail="Requested patient record missing.")

    # Query most recent performance delta to evaluate longitudinal progress tracker
    last_session = db.query(CognitiveGameLog)\
        .filter(CognitiveGameLog.child_id == child_id, CognitiveGameLog.game_name == current_log.game_name)\
        .order_by(desc(CognitiveGameLog.session_date))\
        .first()
    
    analysis = {"status": "Baseline data point logged", "accuracy_delta": 0.0, "speed_delta_seconds": 0.0}
    
    if last_session:
        acc_diff = current_log.accuracy - last_session.accuracy_score
        time_diff = last_session.completion_time_sec - current_log.completion_time_sec
        analysis["accuracy_delta"] = round(acc_diff, 2)
        analysis["speed_delta_seconds"] = round(time_diff, 2)
        analysis["status"] = "Cognitive Progress Verified" if (time_diff > 0 or acc_diff > 0) else "Stable Level Maintenance"

    # Save active transaction to memory tables
    new_log = CognitiveGameLog(
        child_id=child_id,
        game_name=current_log.game_name,
        accuracy_score=current_log.accuracy,
        completion_time_sec=current_log.completion_time_sec,
        reaction_time_ms=current_log.reaction_time_ms
    )
    db.add(new_log)
    db.commit()

    logger.info(f"Terminal Event: Game score saved for ID {child_id} -> {current_log.game_name}: {analysis['status']}")
    return {"status": "Success", "analysis": analysis}

@app.get("/metrics")
def metrics():
    return app_state.get("metrics", {})

@app.get("/feature-importance")
def feature_importance(top_n: int = 10):
    fi = app_state.get("feature_import", {})
    if not fi:
        return {"message": "Not available"}
    return {"top_features": top_features(fi, n=top_n)}
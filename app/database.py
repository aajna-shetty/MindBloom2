import datetime
import os
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = Path(
    os.getenv("MINDBLOOM_SQLITE_PATH", str(PROJECT_ROOT / "mindbloom.db"))
)
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ChildProfile(Base):
    __tablename__ = "child_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    age_months = Column(Integer, nullable=False)
    sex = Column(String, nullable=False)
    family_history_asd = Column(Boolean, default=False)
    jaundice_history = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    screenings = relationship(
        "ScreeningTransaction", back_populates="child", cascade="all, delete-orphan"
    )
    game_logs = relationship(
        "CognitiveGameLog", back_populates="child", cascade="all, delete-orphan"
    )


class ScreeningTransaction(Base):
    __tablename__ = "screening_transactions"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"))
    questionnaire_score = Column(Float, nullable=False)
    video_score = Column(Float, nullable=False)
    hybrid_risk_score = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="screenings")


class CognitiveGameLog(Base):
    __tablename__ = "cognitive_game_logs"

    id = Column(Integer, primary_key=True, index=True)
    child_id = Column(Integer, ForeignKey("child_profiles.id", ondelete="CASCADE"))
    game_name = Column(String, nullable=False)
    accuracy_score = Column(Float, nullable=False)
    completion_time_sec = Column(Float, nullable=False)
    reaction_time_ms = Column(Float, nullable=False)
    session_date = Column(DateTime, default=datetime.datetime.utcnow)

    child = relationship("ChildProfile", back_populates="game_logs")


class Assessment(Base):
    __tablename__ = "assessments"

    id = Column(String, primary_key=True)
    participant_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, default="QUESTIONNAIRE_PENDING")
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    questionnaire_result = relationship(
        "QuestionnaireResult",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    video_assessment = relationship(
        "VideoAssessment",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    integrated_assessment = relationship(
        "IntegratedAssessment",
        back_populates="assessment",
        uselist=False,
        cascade="all, delete-orphan",
    )


class QuestionnaireResult(Base):
    __tablename__ = "questionnaire_results"

    id = Column(Integer, primary_key=True)
    assessment_id = Column(
        String, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    instrument = Column(String, nullable=False)
    responses_json = Column(JSON, nullable=False)
    raw_score = Column(Float, nullable=False)
    maximum_score = Column(Float, nullable=True)
    normalized_score = Column(Float, nullable=True)
    scoring_method = Column(String, nullable=True)
    threshold_metadata_json = Column(JSON, nullable=True)
    interpretation_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    assessment = relationship("Assessment", back_populates="questionnaire_result")


class VideoAssessment(Base):
    __tablename__ = "video_assessments"

    id = Column(Integer, primary_key=True)
    assessment_id = Column(
        String, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    session_id = Column(String, nullable=False)
    status = Column(String, nullable=False)
    duration_seconds = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    valid_frames = Column(Integer, nullable=True)
    captured_frames = Column(Integer, nullable=True)
    mediapipe_processed_frames = Column(Integer, nullable=True)
    vit_submitted_frames = Column(Integer, nullable=True)
    vit_analyzed_frames = Column(Integer, nullable=True)
    processing_errors = Column(Integer, nullable=True)
    mediapipe_statistics_json = Column(JSON, nullable=False, default=dict)
    vit_statistics_json = Column(JSON, nullable=False, default=dict)
    feature_vector_json = Column(JSON, nullable=True)
    feature_schema_json = Column(JSON, nullable=True)
    model_metadata_json = Column(JSON, nullable=True)
    performance_json = Column(JSON, nullable=True)
    video_report_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    assessment = relationship("Assessment", back_populates="video_assessment")


class IntegratedAssessment(Base):
    __tablename__ = "integrated_assessments"

    id = Column(Integer, primary_key=True)
    assessment_id = Column(
        String, ForeignKey("assessments.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    status = Column(String, nullable=False)
    combined_features_json = Column(JSON, nullable=True)
    model_metadata_json = Column(JSON, nullable=True)
    threshold_metadata_json = Column(JSON, nullable=True)
    report_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)

    assessment = relationship("Assessment", back_populates="integrated_assessment")


class ThresholdConfiguration(Base):
    __tablename__ = "threshold_configurations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    source = Column(String, nullable=False)
    version = Column(String, nullable=False)
    configuration_json = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    modality = Column(String, nullable=False)
    artifact_path = Column(String, nullable=True)
    validation_metrics_json = Column(JSON, nullable=True)
    threshold_metadata_json = Column(JSON, nullable=True)
    available = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


def ensure_sqlite_schema() -> None:
    """Apply additive SQLite migrations for databases created by older builds."""
    inspector = inspect(engine)
    additions = {
        "questionnaire_results": {
            "maximum_score": "FLOAT",
            "normalized_score": "FLOAT",
            "scoring_method": "VARCHAR",
            "threshold_metadata_json": "JSON",
        },
        "video_assessments": {
            "captured_frames": "INTEGER",
            "mediapipe_processed_frames": "INTEGER",
            "vit_submitted_frames": "INTEGER",
            "vit_analyzed_frames": "INTEGER",
            "processing_errors": "INTEGER",
            "feature_schema_json": "JSON",
            "model_metadata_json": "JSON",
            "performance_json": "JSON",
        },
        "integrated_assessments": {
            "combined_features_json": "JSON",
            "model_metadata_json": "JSON",
            "threshold_metadata_json": "JSON",
        },
    }
    with engine.begin() as connection:
        for table, columns in additions.items():
            if table not in inspector.get_table_names():
                continue
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, sql_type in columns.items():
                if name not in existing:
                    connection.execute(
                        text(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sql_type}')
                    )

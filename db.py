import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    Index,
    select,
    func,
    desc,
    asc,
)

load_dotenv()


def _get_db_url():
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "formini")
    user = os.getenv("DB_USER", "admin")
    password = os.getenv("DB_PASSWORD", "")
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"


def get_engine():
    return create_engine(_get_db_url(), pool_pre_ping=True, future=True)


metadata = MetaData()

batch_uploads = Table(
    "batch_uploads",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("original_filename", String(255), nullable=False),
    Column("total_rows", Integer, nullable=False),
    Column("created_at", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("batch_id", Integer, nullable=True),
    Column("quiz_score_avg", Float, nullable=False),
    Column("progress_percentage", Float, nullable=False),
    Column("time_spent_hours", Float, nullable=False),
    Column("average_session_duration_min", Float, nullable=False),
    Column("login_frequency", Float, nullable=False),
    Column("video_completion_rate", Float, nullable=False),
    Column("assignments_submitted", Float, nullable=False),
    Column("discussion_participation", Float, nullable=False),
    Column("cluster_id", Integer, nullable=False),
    Column("cluster_name", String(100), nullable=False),
    Column("cluster_description", Text, nullable=False),
    Column("created_at", DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)),
)

idx_predictions_created_at = Index("idx_predictions_created_at", predictions.c.created_at)
idx_predictions_cluster_id = Index("idx_predictions_cluster_id", predictions.c.cluster_id)
idx_predictions_batch_id = Index("idx_predictions_batch_id", predictions.c.batch_id)
idx_batch_uploads_created_at = Index("idx_batch_uploads_created_at", batch_uploads.c.created_at)


def init_db(engine):
    metadata.create_all(engine)


def ensure_indexes(engine):
    indexes = [
        idx_predictions_created_at,
        idx_predictions_cluster_id,
        idx_predictions_batch_id,
        idx_batch_uploads_created_at,
    ]
    for idx in indexes:
        idx.create(bind=engine, checkfirst=True)


def save_prediction(engine, data_dict, result, batch_id=None):
    row = {
        "batch_id": batch_id,
        "quiz_score_avg": float(data_dict["Quiz_Score_Avg"]),
        "progress_percentage": float(data_dict["Progress_Percentage"]),
        "time_spent_hours": float(data_dict["Time_Spent_Hours"]),
        "average_session_duration_min": float(data_dict["Average_Session_Duration_Min"]),
        "login_frequency": float(data_dict["Login_Frequency"]),
        "video_completion_rate": float(data_dict["Video_Completion_Rate"]),
        "assignments_submitted": float(data_dict["Assignments_Submitted"]),
        "discussion_participation": float(data_dict["Discussion_Participation"]),
        "cluster_id": int(result["cluster_id"]),
        "cluster_name": result["cluster_name"],
        "cluster_description": result["description"],
        "created_at": datetime.now(timezone.utc),
    }
    with engine.begin() as conn:
        conn.execute(predictions.insert(), row)


def create_batch_upload(engine, filename, total_rows):
    with engine.begin() as conn:
        res = conn.execute(
            batch_uploads.insert().values(
                original_filename=filename,
                total_rows=int(total_rows),
                created_at=datetime.now(timezone.utc),
            )
        )
        return int(res.lastrowid)


def insert_prediction_rows(engine, rows):
    if not rows:
        return
    with engine.begin() as conn:
        conn.execute(predictions.insert(), rows)


def fetch_recent_predictions(engine, limit=200):
    stmt = (
        select(predictions)
        .order_by(desc(predictions.c.created_at))
        .limit(int(limit))
    )
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]


def fetch_predictions_filtered(
    engine,
    cluster_name=None,
    batch_id=None,
    start_dt=None,
    end_dt=None,
    limit=200,
):
    stmt = select(predictions)
    if cluster_name:
        stmt = stmt.where(predictions.c.cluster_name == cluster_name)
    if batch_id is not None:
        stmt = stmt.where(predictions.c.batch_id == int(batch_id))
    if start_dt is not None:
        stmt = stmt.where(predictions.c.created_at >= start_dt)
    if end_dt is not None:
        stmt = stmt.where(predictions.c.created_at < end_dt)

    stmt = stmt.order_by(desc(predictions.c.created_at)).limit(int(limit))
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]


def fetch_cluster_counts(engine, batch_id=None):
    stmt = select(
        predictions.c.cluster_name,
        func.count().label("count"),
    )
    if batch_id is not None:
        stmt = stmt.where(predictions.c.batch_id == int(batch_id))
    stmt = stmt.group_by(predictions.c.cluster_name).order_by(desc("count"))
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt).fetchall()]


def fetch_last_batch(engine):
    stmt = select(batch_uploads).order_by(desc(batch_uploads.c.created_at)).limit(1)
    with engine.connect() as conn:
        row = conn.execute(stmt).first()
        return dict(row._mapping) if row else None


def fetch_totals(engine):
    with engine.connect() as conn:
        total_predictions = conn.execute(select(func.count()).select_from(predictions)).scalar()
        total_batches = conn.execute(select(func.count()).select_from(batch_uploads)).scalar()
    return {
        "predictions": int(total_predictions or 0),
        "batches": int(total_batches or 0),
    }


def fetch_daily_counts(engine, start_dt, end_dt=None, batch_id=None):
    stmt = select(
        func.date(predictions.c.created_at).label("day"),
        func.count().label("count"),
    )
    if start_dt is not None:
        stmt = stmt.where(predictions.c.created_at >= start_dt)
    if end_dt is not None:
        stmt = stmt.where(predictions.c.created_at < end_dt)
    if batch_id is not None:
        stmt = stmt.where(predictions.c.batch_id == int(batch_id))

    stmt = stmt.group_by("day").order_by(asc("day"))
    with engine.connect() as conn:
        rows = conn.execute(stmt).fetchall()
        return [dict(row._mapping) for row in rows]

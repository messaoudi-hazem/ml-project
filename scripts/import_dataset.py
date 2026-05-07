import argparse
import os
import sys
from datetime import datetime, timezone
import joblib
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import get_engine, init_db, create_batch_upload, insert_prediction_rows


def parse_args():
    parser = argparse.ArgumentParser(description="Import CSV and store predictions in MySQL")
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument(
        "--model",
        default="kmeans_segmentation_apprenants.joblib",
        help="Path to the joblib model artifact",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = joblib.load(args.model)
    scaler = artifact["scaler"]
    kmeans = artifact["kmeans"]
    features = artifact["features"]
    cluster_names = artifact["cluster_names"]
    cluster_descriptions = artifact["cluster_descriptions"]

    df = pd.read_csv(args.csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    X = df[features].copy()
    X_scaled = scaler.transform(X)
    preds = kmeans.predict(X_scaled)

    engine = get_engine()
    init_db(engine)
    batch_id = create_batch_upload(engine, args.csv, len(df))

    rows = []
    for i, row in X.iterrows():
        cluster_id = int(preds[i])
        rows.append({
            "batch_id": batch_id,
            "quiz_score_avg": float(row["Quiz_Score_Avg"]),
            "progress_percentage": float(row["Progress_Percentage"]),
            "time_spent_hours": float(row["Time_Spent_Hours"]),
            "average_session_duration_min": float(row["Average_Session_Duration_Min"]),
            "login_frequency": float(row["Login_Frequency"]),
            "video_completion_rate": float(row["Video_Completion_Rate"]),
            "assignments_submitted": float(row["Assignments_Submitted"]),
            "discussion_participation": float(row["Discussion_Participation"]),
            "cluster_id": cluster_id,
            "cluster_name": cluster_names[cluster_id],
            "cluster_description": cluster_descriptions[cluster_id],
            "created_at": datetime.now(timezone.utc),
        })

    insert_prediction_rows(engine, rows)
    print(f"Imported {len(rows)} rows into MySQL (batch_id={batch_id}).")


if __name__ == "__main__":
    main()

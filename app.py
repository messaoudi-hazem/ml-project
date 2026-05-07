"""
🔵 Segmentation des Apprenants — Application Flask
Projet Formini · Plateforme E-Learning Intelligente

Lance localement avec :
    python app.py

L'app sera accessible sur http://localhost:5000
"""
import os
import io
import json
import csv
from datetime import datetime, timezone, timedelta
import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify,
    send_file, redirect, url_for, flash, make_response
)
from db import (
    get_engine,
    init_db,
    ensure_indexes,
    save_prediction,
    create_batch_upload,
    insert_prediction_rows,
    fetch_recent_predictions,
    fetch_predictions_filtered,
    fetch_cluster_counts,
    fetch_last_batch,
    fetch_totals,
    fetch_daily_counts,
)

# ============================================================
# CONFIGURATION
# ============================================================
app = Flask(__name__)
app.config["SECRET_KEY"] = "formini-secret-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max upload

MODEL_PATH = "kmeans_segmentation_apprenants.joblib"

# ============================================================
# CHARGEMENT DU MODÈLE (une seule fois au démarrage)
# ============================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"❌ Fichier {MODEL_PATH} introuvable. "
        f"Exécute d'abord le notebook pour le générer."
    )

print(f"📦 Chargement du modèle depuis {MODEL_PATH}...")
artifact = joblib.load(MODEL_PATH)
SCALER = artifact["scaler"]
KMEANS = artifact["kmeans"]
FEATURES = artifact["features"]
CLUSTER_NAMES = artifact["cluster_names"]
CLUSTER_DESCRIPTIONS = artifact["cluster_descriptions"]
CLUSTER_COLORS = artifact["cluster_colors"]
METRICS = artifact["metrics"]
print(f"✅ Modèle chargé · {len(FEATURES)} features · {len(CLUSTER_NAMES)} clusters")

# ============================================================
# BASE DE DONNEES (MySQL)
# ============================================================
DB_ENGINE = get_engine()
init_db(DB_ENGINE)
ensure_indexes(DB_ENGINE)

# ============================================================
# RECOMMANDATIONS PÉDAGOGIQUES PAR PROFIL
# ============================================================
RECOMMENDATIONS = {
    "🟢 Apprenants Avancés": [
        "✅ Proposer des modules avancés et certifications premium",
        "🏆 Inviter à devenir mentor pour les autres apprenants",
        "🎯 Suggérer des projets capstone complexes",
        "📈 Suivre l'inscription aux modules avancés (KPI principal)",
    ],
    "🔵 Apprenants Intermédiaires": [
        "🎯 Activer les quiz adaptatifs pour pousser le niveau",
        "📚 Recommandations de cours suivants ciblées (Content-Based Filtering)",
        "💡 Proposer des challenges hebdomadaires pour stimuler",
        "📊 Suivre la progression vers le profil Avancé",
    ],
    "🟠 Débutants Motivés": [
        "👨‍🏫 Programmer des sessions de tutoring ou Q&A live",
        "📖 Fournir des fiches méthode et résumés visuels",
        "🎬 Recommander des micro-cours fondamentaux",
        "📈 KPI : score moyen aux quiz à T+1 mois",
    ],
    "🔴 Apprenants Passifs": [
        "📧 Lancer une campagne de réengagement (email + notifications)",
        "✂️ Proposer des modules courts (15 min) pour relancer la dynamique",
        "🎁 Offrir une réduction ou un essai sur un nouveau cours",
        "📈 KPI : taux de reconnexion à 7 / 14 / 30 jours",
    ],
}

# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================
def predict_cluster(data_dict):
    """Prend un dict {feature: valeur} et retourne le cluster + métadonnées."""
    df = pd.DataFrame([data_dict])[FEATURES]
    X_scaled = SCALER.transform(df)
    cluster_id = int(KMEANS.predict(X_scaled)[0])

    # Distance aux centroïdes → proximité « soft »
    distances = KMEANS.transform(X_scaled)[0]
    proximities = 1 / (1 + distances)
    proximities = (proximities / proximities.sum() * 100).round(1)

    cluster_label = CLUSTER_NAMES[cluster_id]
    return {
        "cluster_id": cluster_id,
        "cluster_name": cluster_label,
        "description": CLUSTER_DESCRIPTIONS[cluster_id],
        "color": CLUSTER_COLORS[cluster_id],
        "recommendations": RECOMMENDATIONS.get(cluster_label, []),
        "proximities": [
            {
                "name": CLUSTER_NAMES[i],
                "value": float(proximities[i]),
                "color": CLUSTER_COLORS[i],
            }
            for i in range(len(CLUSTER_NAMES))
        ],
    }


# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    """Page d'accueil avec le formulaire de prédiction."""
    return render_template(
        "index.html",
        features=FEATURES,
        metrics=METRICS,
        cluster_names=list(CLUSTER_NAMES.values()),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint API qui reçoit le formulaire et renvoie la prédiction (JSON)."""
    try:
        data = {
            "Quiz_Score_Avg":               float(request.form["quiz_score"]),
            "Progress_Percentage":          float(request.form["progress"]),
            "Time_Spent_Hours":             float(request.form["time_spent"]),
            "Average_Session_Duration_Min": float(request.form["session_dur"]),
            "Login_Frequency":              float(request.form["login_freq"]),
            "Video_Completion_Rate":        float(request.form["video"]),
            "Assignments_Submitted":        float(request.form["assignments"]),
            "Discussion_Participation":     float(request.form["discussion"]),
        }
        result = predict_cluster(data)
        try:
            save_prediction(DB_ENGINE, data, result)
        except Exception as e:
            print(f"⚠️ DB insert failed (single): {e}")
        return jsonify({"ok": True, "result": result})
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": f"Données invalides : {e}"}), 400


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """Page de prédiction par lot (upload CSV)."""
    if request.method == "GET":
        return render_template("batch.html", features=FEATURES)

    # POST : on reçoit un fichier CSV
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("❌ Aucun fichier sélectionné", "error")
        return redirect(url_for("batch"))

    try:
        df_in = pd.read_csv(file)
    except Exception as e:
        flash(f"❌ Erreur de lecture du CSV : {e}", "error")
        return redirect(url_for("batch"))

    missing = [c for c in FEATURES if c not in df_in.columns]
    if missing:
        flash(f"❌ Colonnes manquantes dans le CSV : {missing}", "error")
        return redirect(url_for("batch"))

    # Pipeline batch
    X = df_in[FEATURES].copy()
    X_scaled = SCALER.transform(X)
    preds = KMEANS.predict(X_scaled)
    df_in["Cluster"] = preds
    df_in["Profil"] = [CLUSTER_NAMES[c] for c in preds]

    # DB: create batch + insert rows
    try:
        batch_id = create_batch_upload(DB_ENGINE, file.filename, len(df_in))
        rows = []
        for i, row in df_in[FEATURES].iterrows():
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
                "cluster_name": CLUSTER_NAMES[cluster_id],
                "cluster_description": CLUSTER_DESCRIPTIONS[cluster_id],
                "created_at": datetime.now(timezone.utc),
            })
        insert_prediction_rows(DB_ENGINE, rows)
    except Exception as e:
        print(f"⚠️ DB insert failed (batch): {e}")

    # Stats
    counts = df_in["Profil"].value_counts().to_dict()
    total = len(df_in)
    distribution = [
        {
            "name": name,
            "count": int(counts.get(name, 0)),
            "pct": round(counts.get(name, 0) / total * 100, 1),
            "color": CLUSTER_COLORS[cid],
        }
        for cid, name in CLUSTER_NAMES.items()
    ]

    # Sauvegarde temporaire pour le téléchargement
    output_path = "/tmp/apprenants_segmentes.csv"
    df_in.to_csv(output_path, index=False)

    # Aperçu (50 premières lignes)
    preview = df_in[["Profil"] + FEATURES].head(50).to_html(
        classes="table table-striped table-hover", index=False, border=0
    )

    return render_template(
        "batch_result.html",
        total=total,
        distribution=distribution,
        preview=preview,
    )


@app.route("/download")
def download():
    """Télécharge le CSV résultat de la dernière segmentation batch."""
    path = "/tmp/apprenants_segmentes.csv"
    if not os.path.exists(path):
        flash("❌ Aucun résultat à télécharger", "error")
        return redirect(url_for("batch"))
    return send_file(
        path,
        as_attachment=True,
        download_name="apprenants_segmentes.csv",
        mimetype="text/csv",
    )


@app.route("/about")
def about():
    """Page « À propos » du modèle."""
    return render_template(
        "about.html",
        features=FEATURES,
        cluster_names=CLUSTER_NAMES,
        cluster_descriptions=CLUSTER_DESCRIPTIONS,
        cluster_colors=CLUSTER_COLORS,
        metrics=METRICS,
    )


@app.route("/history")
def history():
    """Historique des prédictions (dernières lignes)."""
    filters = _parse_history_filters(max_limit=500)
    rows = fetch_predictions_filtered(
        DB_ENGINE,
        cluster_name=filters["cluster"],
        batch_id=filters["batch_id"],
        start_dt=filters["start_dt"],
        end_dt=filters["end_dt"],
        limit=filters["limit"],
    )
    cluster_color_by_name = {
        CLUSTER_NAMES[cid]: CLUSTER_COLORS[cid] for cid in CLUSTER_NAMES
    }
    return render_template(
        "history.html",
        rows=rows,
        limit=filters["limit"],
        filter_cluster=filters["cluster"],
        filter_batch_id=filters["batch_id"],
        filter_start=filters["start"],
        filter_end=filters["end"],
        cluster_options=list(CLUSTER_NAMES.values()),
        cluster_color_by_name=cluster_color_by_name,
    )


@app.route("/history/export")
def history_export():
    """Export CSV de l'historique filtré."""
    filters = _parse_history_filters(max_limit=5000)
    rows = fetch_predictions_filtered(
        DB_ENGINE,
        cluster_name=filters["cluster"],
        batch_id=filters["batch_id"],
        start_dt=filters["start_dt"],
        end_dt=filters["end_dt"],
        limit=filters["limit"],
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "created_at",
        "cluster_name",
        "batch_id",
        "quiz_score_avg",
        "progress_percentage",
        "time_spent_hours",
        "average_session_duration_min",
        "login_frequency",
        "video_completion_rate",
        "assignments_submitted",
        "discussion_participation",
    ])
    for row in rows:
        writer.writerow([
            row.get("created_at"),
            row.get("cluster_name"),
            row.get("batch_id"),
            row.get("quiz_score_avg"),
            row.get("progress_percentage"),
            row.get("time_spent_hours"),
            row.get("average_session_duration_min"),
            row.get("login_frequency"),
            row.get("video_completion_rate"),
            row.get("assignments_submitted"),
            row.get("discussion_participation"),
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Type"] = "text/csv"
    response.headers["Content-Disposition"] = "attachment; filename=history_export.csv"
    return response


@app.route("/admin")
def admin():
    """Tableau de bord simple pour les stats DB."""
    totals = fetch_totals(DB_ENGINE)
    overall_counts = fetch_cluster_counts(DB_ENGINE)
    last_batch = fetch_last_batch(DB_ENGINE)
    last_batch_counts = []
    if last_batch:
        last_batch_counts = fetch_cluster_counts(DB_ENGINE, batch_id=last_batch["id"])

    cluster_color_by_name = {
        CLUSTER_NAMES[cid]: CLUSTER_COLORS[cid] for cid in CLUSTER_NAMES
    }

    end_dt = datetime.now(timezone.utc) + timedelta(seconds=1)
    start_dt = end_dt - timedelta(days=29)
    daily_rows = fetch_daily_counts(DB_ENGINE, start_dt=start_dt, end_dt=end_dt)
    daily_map = {str(r["day"]): int(r["count"]) for r in daily_rows}

    trend_labels = []
    trend_values = []
    cursor = start_dt.date()
    end_date = end_dt.date()
    while cursor <= end_date:
        label = cursor.isoformat()
        trend_labels.append(label)
        trend_values.append(daily_map.get(label, 0))
        cursor = cursor + timedelta(days=1)

    return render_template(
        "admin.html",
        totals=totals,
        overall_counts=overall_counts,
        last_batch=last_batch,
        last_batch_counts=last_batch_counts,
        cluster_color_by_name=cluster_color_by_name,
        trend_labels=trend_labels,
        trend_values=trend_values,
    )


def _parse_history_filters(max_limit=500):
    limit = request.args.get("limit", "200")
    cluster = request.args.get("cluster", "").strip() or None
    batch_id = request.args.get("batch_id", "").strip() or None
    start = request.args.get("start", "").strip()
    end = request.args.get("end", "").strip()

    try:
        limit_value = min(max(int(limit), 10), int(max_limit))
    except ValueError:
        limit_value = 200

    start_dt = None
    end_dt = None
    if start:
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            start = ""
    if end:
        try:
            end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end_dt = end_dt + timedelta(days=1)
        except ValueError:
            end = ""

    return {
        "limit": limit_value,
        "cluster": cluster,
        "batch_id": batch_id,
        "start": start,
        "end": end,
        "start_dt": start_dt,
        "end_dt": end_dt,
    }


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API JSON pure (pour intégration externe).

    POST JSON : {"Quiz_Score_Avg": 75, "Progress_Percentage": 60, ...}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "JSON invalide"}), 400

    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(
            {"ok": False, "error": f"Features manquantes : {missing}"}
        ), 400

    try:
        payload = {f: float(data[f]) for f in FEATURES}
        result = predict_cluster(payload)
        try:
            save_prediction(DB_ENGINE, payload, result)
        except Exception as e:
            print(f"⚠️ DB insert failed (api): {e}")
        return jsonify({"ok": True, "result": result})
    except (TypeError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ============================================================
# LANCEMENT
# ============================================================
if __name__ == "__main__":
    # debug=True en dev seulement, à passer à False en production
    app.run(host="0.0.0.0", port=5000, debug=True)

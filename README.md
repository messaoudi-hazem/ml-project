# 🔵 Segmentation des Apprenants — K-Means + Flask

> **Tâche 3a** du projet ML SAE2 · Plateforme E-Learning **Formini** · BinaryLords 2025-2026

Segmentation comportementale de **100 000 apprenants** en 4 profils business (Avancés / Intermédiaires / Débutants Motivés / Passifs) via K-Means, avec une **application web Flask** pour les conseillers pédagogiques.

---

## 📂 Contenu du dossier

```
flask_app/
├── app.py                                  ← Backend Flask
├── requirements.txt                        ← Dépendances Python
├── kmeans_segmentation_apprenants.joblib   ← Modèle entraîné (généré par le notebook)
├── templates/
│   ├── base.html                           ← Layout commun (navbar + footer)
│   ├── index.html                          ← Page accueil + formulaire de prédiction
│   ├── batch.html                          ← Page upload CSV
│   ├── batch_result.html                   ← Page résultats segmentation lot
│   └── about.html                          ← Page « À propos »
├── static/
│   └── style.css                           ← CSS custom
└── README.md                               ← Ce fichier
```

---

## 🚀 Partie 1 — Générer le modèle (notebook Colab)

1. Ouvre **`Segmentation_Apprenants_KMeans.ipynb`** sur Google Colab
2. Upload `Course_Completion_Prediction.csv` dans le panneau Files
3. **Runtime → Run all** (ou `Ctrl+F9`)
4. Le notebook produit `kmeans_segmentation_apprenants.joblib`
5. **Télécharge** ce fichier et place-le **à côté de `app.py`**

---

## 🌐 Partie 2 — Lancer l'app Flask

### ⭐ Option A — Lancement local (dev / démo en présentation)

```bash
# 1. Va dans le dossier
cd flask_app

# 2. (Recommandé) Crée un environnement virtuel
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Installe les dépendances
pip install -r requirements.txt

# 4. Lance l'app
python app.py
```

L'app est accessible sur **`http://localhost:5000`** 🎉

> 💡 Pour la démo en classe, ce mode est parfait. Le serveur de développement Flask suffit pour une démo locale.

---

### Option B — Production locale avec Gunicorn (Linux/Mac)

Le serveur de dev Flask **n'est pas fait pour la production**. Pour servir l'app proprement :

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

- `-w 4` : 4 workers (4 requêtes en parallèle)
- `-b 0.0.0.0:5000` : accessible depuis le réseau

> ⚠️ Sur Windows, Gunicorn ne fonctionne pas. Utilise `waitress` à la place : `pip install waitress` puis `waitress-serve --port=5000 app:app`.

---

### ⭐ Option C — Déploiement gratuit sur Render.com (recommandé)

[Render.com](https://render.com) héberge gratuitement les apps Flask à partir d'un repo GitHub.

#### Étape 1 — Préparer le repo GitHub

```bash
# Crée un nouveau repo sur GitHub (ex: formini-segmentation)
git clone https://github.com/<TON_USER>/formini-segmentation.git
cd formini-segmentation

# Copie tous les fichiers du dossier flask_app/ dedans
# (app.py, requirements.txt, .joblib, templates/, static/)

git add .
git commit -m "Initial commit - Segmentation Apprenants Flask"
git push origin main
```

> Le `.joblib` fait ~400 KB → pas besoin de Git LFS.

#### Étape 2 — Déployer sur Render

1. Va sur **[render.com](https://render.com)** → Sign in with GitHub
2. **New + → Web Service**
3. Sélectionne ton repo `formini-segmentation`
4. Configure :
   - **Name** : `formini-segmentation`
   - **Region** : `Frankfurt` (le plus proche)
   - **Branch** : `main`
   - **Runtime** : `Python 3`
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `gunicorn app:app`
   - **Instance Type** : `Free`
5. Clique **Create Web Service** ⏱️ Attends 3-5 minutes

🎉 Ton app est **live** à une URL du type `https://formini-segmentation.onrender.com`

> ⚠️ Le tier gratuit Render endort l'app après 15 min d'inactivité — le premier accès prend ~30s pour la réveiller. Largement suffisant pour la soutenance.

---

### Option D — Déploiement sur PythonAnywhere (alternative gratuite)

1. Crée un compte gratuit sur **[pythonanywhere.com](https://pythonanywhere.com)**
2. **Web → Add a new web app → Flask → Python 3.10**
3. Upload tes fichiers via l'onglet **Files**
4. Dans **Web → WSGI configuration**, pointe vers `app.app`
5. Clique **Reload**

---

## 🎬 Démo de l'app

L'application a **3 pages** :

| Page | Description |
|---|---|
| 🎯 **Accueil** (`/`) | Formulaire avec sliders interactifs → résultat + radar + recommandations |
| 📁 **Lot** (`/batch`) | Upload CSV → segmentation en masse + export résultats |
| ℹ️ **À propos** (`/about`) | Description du modèle, métriques, profils, API |

### 🔌 API REST

L'app expose aussi une **API JSON** pour intégration externe :

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Quiz_Score_Avg": 75,
    "Progress_Percentage": 60,
    "Time_Spent_Hours": 5,
    "Average_Session_Duration_Min": 35,
    "Login_Frequency": 5,
    "Video_Completion_Rate": 70,
    "Assignments_Submitted": 6,
    "Discussion_Participation": 3
  }'
```

Réponse :
```json
{
  "ok": true,
  "result": {
    "cluster_id": 1,
    "cluster_name": "🔵 Apprenants Intermédiaires",
    "description": "Bon niveau, progression correcte...",
    "color": "#3498db",
    "recommendations": ["...", "...", "...", "..."],
    "proximities": [...]
  }
}
```

---

## 🧠 Stack technique

- **Python 3.10+**
- **Flask** 3.0+ (backend web)
- **Gunicorn** (serveur WSGI production)
- **scikit-learn** 1.3+ (KMeans, StandardScaler)
- **joblib** (sérialisation modèle)
- **Bootstrap 5** + **Chart.js** (frontend, via CDN)

---

## 📊 Performance du modèle

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Silhouette Score** | ~0.12 | Faible (attendu pour données comportementales) — profils business-pertinents |
| **Davies-Bouldin** | ~2.10 | Mesure la séparation moyenne |
| **Inertie (WCSS)** | optimisée | Choisie via méthode du coude à k=4 |
| **Apprenants** | 100 000 | Dataset complet |
| **Temps de prédiction** | < 50 ms | Modèle chargé en cache |

---

## ✅ Checklist avant la soutenance

- [ ] Notebook exécuté de bout en bout sans erreur
- [ ] Le fichier `.joblib` est généré et copié dans `flask_app/`
- [ ] L'app Flask tourne en local : `python app.py` → http://localhost:5000
- [ ] L'app est déployée publiquement (URL `*.onrender.com`)
- [ ] Slide PowerPoint mis à jour avec : k=4, Silhouette ≈ 0.12, lien vers l'app
- [ ] Un CSV de test est prêt pour la démo du mode batch

---

## 🛠️ Troubleshooting

| Problème | Solution |
|---|---|
| `FileNotFoundError: kmeans_segmentation_apprenants.joblib` | Place le fichier `.joblib` à côté de `app.py` |
| `Address already in use` (port 5000) | Change le port : `python app.py` ne marche pas → modifie `port=5000` en `port=5001` dans `app.py` |
| L'app sur Render répond 502 | Vérifie le **Start Command** : doit être `gunicorn app:app` (pas `python app.py`) |
| `ModuleNotFoundError: No module named 'flask'` | `pip install -r requirements.txt` |

---

**Auteur** : équipe BinaryLords · ML SAE2 · 2025-2026

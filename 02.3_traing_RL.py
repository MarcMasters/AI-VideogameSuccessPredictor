"""
02.3_train_RL.py
================
Entrena un LogisticRegression con las features pre-lanzamiento.
Guarda el modelo en models/lr_pipeline.joblib
"""

import pandas as pd
import numpy as np
import os, json, joblib
from sklearn.linear_model      import LogisticRegression
from sklearn.model_selection   import StratifiedKFold
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics           import classification_report, f1_score

DATA_PATH = "data/processed.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
print("Cargando datos preprocesados…")
df = pd.read_parquet(DATA_PATH)
df = df[
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
].copy()

# ─────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────
NUMERIC_BASE = [
    "price", "is_free", "required_age",
    "plat_windows", "plat_mac", "plat_linux",
    "is_multiplayer", "n_languages",
    "desc_length", "release_year", "release_quarter",
    "publisher_success_avg",
]
GENRE_COLS = [c for c in df.columns if c.startswith("genre_")]
TAG_COLS   = [c for c in df.columns if c.startswith("tag_")]
NUMERIC    = NUMERIC_BASE + GENRE_COLS + TAG_COLS
TARGET     = "success_class"

df_model             = df[NUMERIC + [TARGET]].dropna(subset=[TARGET])
df_model[GENRE_COLS] = df_model[GENRE_COLS].fillna(0)
df_model[TAG_COLS]   = df_model[TAG_COLS].fillna(0)
df_model[NUMERIC]    = df_model[NUMERIC].fillna(0)

X = df_model[NUMERIC]
y = df_model[TARGET].astype(int)

print(f"\nDistribución de clases:")
for k, v in y.value_counts().sort_index().items():
    print(f"  {['Fracaso','Moderado','Alto éxito'][k]}: {v} ({v/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. PIPELINE
# LR sí necesita StandardScaler
# class_weight="balanced" soportado nativamente
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class  = "multinomial",
        solver       = "lbfgs",
        max_iter     = 1000,
        C            = 1.0,         # regularización; 1/C = fuerza de L2
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )),
])

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    score = f1_score(y_val, pipeline.predict(X_val), average="macro")
    cv_scores.append(score)
    print(f"  Fold {fold+1}: F1-macro = {score:.3f}")

cv_scores = np.array(cv_scores)
print(f"  Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─────────────────────────────────────────────
# 5. FIT FINAL
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
pipeline.fit(X, y)

y_pred = pipeline.predict(X)
print("\nClassification report (train — referencia):")
print(classification_report(y, y_pred,
      target_names=["Fracaso", "Moderado", "Alto éxito"]))

# ─────────────────────────────────────────────
# 6. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(pipeline, f"{MODEL_DIR}/lr_pipeline.joblib")

metadata = {
    "numeric_features":   NUMERIC,
    "target":             TARGET,
    "classes":            {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"},
    "cv_f1_macro_mean":   float(cv_scores.mean()),
    "cv_f1_macro_std":    float(cv_scores.std()),
    "n_train":            len(X),
}
with open(f"{MODEL_DIR}/lr_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/lr_pipeline.joblib")
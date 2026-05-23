"""
02_train.py
===========
Entrena un XGBClassifier con las features pre-lanzamiento.
Guarda el modelo + el pipeline de encoding en models/
"""

import pandas as pd
import numpy as np
import re, os, json, joblib
from xgboost                   import XGBClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import FunctionTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics           import classification_report, confusion_matrix

DATA_PATH  = "data/processed.parquet"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
print("Cargando datos preprocesados…")
df = pd.read_parquet(DATA_PATH)
print(f"  Filas cargadas: {len(df)}")

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

TARGET = "success_class"

df_model = df[NUMERIC + [TARGET]].dropna(subset=[TARGET])
# XGBoost maneja NaNs nativamente — no rellenamos con 0
# Solo rellenamos las binarias (genre/tag) que siempre son 0/1
df_model[GENRE_COLS] = df_model[GENRE_COLS].fillna(0)
df_model[TAG_COLS]   = df_model[TAG_COLS].fillna(0)

X = df_model[NUMERIC]
y = df_model[TARGET].astype(int)

print(f"\nDistribución de clases:")
vc = y.value_counts().sort_index()
for k, v in vc.items():
    label = {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"}[k]
    print(f"  {label}: {v} ({v/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. MODELO
# ─────────────────────────────────────────────
# XGBoost no necesita StandardScaler → pipeline mínimo
clf = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)

# Pipeline sin preprocesado de escala
pipeline = Pipeline([("clf", clf)])

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION
# Nota: cross_val_score no pasa sample_weight fácilmente,
# así que hacemos CV manual para ser fieles al balanceo.
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.metrics import f1_score

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    sw = compute_sample_weight("balanced", y_tr)
    clf_fold = XGBClassifier(
        n_estimators     = 400,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        objective        = "multi:softprob",
        num_class        = 3,
        eval_metric      = "mlogloss",
        random_state     = 42,
        n_jobs           = -1,
    )
    clf_fold.fit(X_tr, y_tr, sample_weight=sw)
    y_pred_val = clf_fold.predict(X_val)
    score = f1_score(y_val, y_pred_val, average="macro")
    cv_scores.append(score)
    print(f"  Fold {fold+1}: F1-macro = {score:.3f}")

cv_scores = np.array(cv_scores)
print(f"  Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ─────────────────────────────────────────────
# 5. FIT FINAL con sample_weight
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
sample_weights = compute_sample_weight("balanced", y)
pipeline.fit(X, y, clf__sample_weight=sample_weights)

# ─────────────────────────────────────────────
# 6. MÉTRICAS (train — referencia)
# ─────────────────────────────────────────────
y_pred = pipeline.predict(X)
print("\nClassification report (train — referencia):")
print(classification_report(y, y_pred,
      target_names=["Fracaso", "Moderado", "Alto éxito"]))

# ─────────────────────────────────────────────
# 7. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(pipeline, f"{MODEL_DIR}/xgb_pipeline.joblib")

metadata = {
    "numeric_features":     NUMERIC,
    "categorical_features": [],
    "target":               TARGET,
    "classes":              {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"},
    "cv_f1_macro_mean":     float(cv_scores.mean()),
    "cv_f1_macro_std":      float(cv_scores.std()),
    "n_train":              len(X),
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/xgb_pipeline.joblib")
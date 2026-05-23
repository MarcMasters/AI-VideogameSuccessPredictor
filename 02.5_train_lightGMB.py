"""
02.3_train_LGBM.py
==================
LightGBM con feature engineering extendido.
Guarda el modelo en models/lgbm_pipeline.joblib
"""

import pandas as pd
import numpy as np
import os, json, joblib
from lightgbm                   import LGBMClassifier
from sklearn.model_selection    import StratifiedKFold, train_test_split
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import classification_report, f1_score

DATA_PATH = "data/processed_v2.parquet"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGA Y FEATURES
# ─────────────────────────────────────────────
print("Cargando datos…")
df = pd.read_parquet(DATA_PATH)
df = df[
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
].copy()

NUMERIC_BASE = [
    "price", "is_free", "required_age",
    "plat_windows", "plat_mac", "plat_linux",
    "is_multiplayer", "n_languages",
    "desc_length", "release_year", "release_quarter",
    "publisher_success_avg",
    # ★ features nuevas
    "price_per_lang", "n_tags", "n_platforms",
    "mp_x_free", "action_singleplayer",
    "desc_length_log", "publisher_tier",
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

print(f"  Dataset: {len(X)} filas, {len(NUMERIC)} features")
for k, v in y.value_counts().sort_index().items():
    print(f"  {['Fracaso','Moderado','Alto éxito'][k]}: {v} ({v/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. MODELO
# LightGBM soporta class_weight="balanced" nativamente
# y maneja NaNs — no necesita StandardScaler
# ─────────────────────────────────────────────
clf = LGBMClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.03,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,        # silencia logs de LightGBM
)

pipeline = Pipeline([("clf", clf)])

# ─────────────────────────────────────────────
# 3. CV MANUAL CON EARLY STOPPING
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores    = []
train_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    clf_fold = LGBMClassifier(
        n_estimators      = 500,
        max_depth         = 6,
        learning_rate     = 0.03,
        subsample         = 0.7,
        colsample_bytree  = 0.7,
        min_child_samples = 20,
        reg_alpha         = 0.1,
        reg_lambda        = 1.0,
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    clf_fold.fit(
        X_tr, y_tr,
        eval_set              = [(X_val, y_val)],
        callbacks             = [
            __import__("lightgbm").early_stopping(30, verbose=False),
            __import__("lightgbm").log_evaluation(period=-1),
        ],
    )

    val_f1   = f1_score(y_val, clf_fold.predict(X_val), average="macro")
    train_f1 = f1_score(y_tr,  clf_fold.predict(X_tr),  average="macro")
    cv_scores.append(val_f1)
    train_scores.append(train_f1)
    print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
          f"gap={train_f1-val_f1:.3f}  (best iter: {clf_fold.best_iteration_})")

cv_scores    = np.array(cv_scores)
train_scores = np.array(train_scores)
print(f"\n  Val  F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Train F1-macro: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
print(f"  Gap medio:      {(train_scores - cv_scores).mean():.3f}")

# ─────────────────────────────────────────────
# 4. FIT FINAL CON EARLY STOPPING
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
import lightgbm as lgb

X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

final_clf = LGBMClassifier(
    n_estimators      = 500,
    max_depth         = 6,
    learning_rate     = 0.03,
    subsample         = 0.7,
    colsample_bytree  = 0.7,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)
final_clf.fit(
    X_tr_f, y_tr_f,
    eval_set  = [(X_val_f, y_val_f)],
    callbacks = [
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=-1),
    ],
)
print(f"  Mejor iteración: {final_clf.best_iteration_}")

final_pipeline = Pipeline([("clf", final_clf)])

y_pred = final_pipeline.predict(X)
print("\nClassification report (train completo — referencia):")
print(classification_report(y, y_pred,
      target_names=["Fracaso", "Moderado", "Alto éxito"]))

# ─────────────────────────────────────────────
# 5. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(final_pipeline, f"{MODEL_DIR}/lgbm_pipeline.joblib")

metadata = {
    "numeric_features": NUMERIC,
    "target":           TARGET,
    "classes":          {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"},
    "cv_f1_macro_mean": float(cv_scores.mean()),
    "cv_f1_macro_std":  float(cv_scores.std()),
    "n_train":          len(X),
    "data_version":     "processed_v2",
}
with open(f"{MODEL_DIR}/lgbm_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/lgbm_pipeline.joblib")
"""
02.1_train_XGB.py
=================
XGBoost con hiperparámetros optimizados (sin RandomizedSearchCV).
"""

import pandas as pd
import numpy as np
import os, json, joblib
from xgboost                    import XGBClassifier
from sklearn.model_selection    import StratifiedKFold, train_test_split
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight

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
# 2. HIPERPARÁMETROS OPTIMIZADOS
# ─────────────────────────────────────────────
BEST_PARAMS = dict(
    subsample        = 0.6,
    reg_lambda       = 1.0,
    reg_alpha        = 0.1,
    n_estimators     = 600,
    min_child_weight = 100,
    max_depth        = 3,
    learning_rate    = 0.03,
    gamma            = 1.0,
    colsample_bytree = 0.4,
)

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
    sw = compute_sample_weight("balanced", y_tr)

    clf_fold = XGBClassifier(
        **BEST_PARAMS,
        objective             = "multi:softprob",
        num_class             = 3,
        eval_metric           = "mlogloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )
    clf_fold.fit(
        X_tr, y_tr,
        sample_weight = sw,
        eval_set      = [(X_val, y_val)],
        verbose       = False,
    )

    val_f1   = f1_score(y_val, clf_fold.predict(X_val), average="macro")
    train_f1 = f1_score(y_tr,  clf_fold.predict(X_tr),  average="macro")
    cv_scores.append(val_f1)
    train_scores.append(train_f1)
    print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
          f"gap={train_f1-val_f1:.3f}  (best iter: {clf_fold.best_iteration})")

cv_scores    = np.array(cv_scores)
train_scores = np.array(train_scores)
print(f"\n  Val  F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Train F1-macro: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
print(f"  Gap medio:      {(train_scores - cv_scores).mean():.3f}")

# ─────────────────────────────────────────────
# 4. FIT FINAL
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
sw_tr_f = compute_sample_weight("balanced", y_tr_f)

final_clf = XGBClassifier(
    **BEST_PARAMS,
    objective             = "multi:softprob",
    num_class             = 3,
    eval_metric           = "mlogloss",
    early_stopping_rounds = 30,
    random_state          = 42,
    n_jobs                = -1,
)
final_clf.fit(
    X_tr_f, y_tr_f,
    sample_weight = sw_tr_f,
    eval_set      = [(X_val_f, y_val_f)],
    verbose       = False,
)
print(f"  Mejor iteración: {final_clf.best_iteration}")

final_pipeline = Pipeline([("clf", final_clf)])

y_pred = final_pipeline.predict(X)
print("\nClassification report (train completo — referencia):")
print(classification_report(y, y_pred,
      target_names=["Fracaso", "Moderado", "Alto éxito"]))

# ─────────────────────────────────────────────
# 5. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(final_pipeline, f"{MODEL_DIR}/xgb_pipeline.joblib")

metadata = {
    "numeric_features": NUMERIC,
    "tagit get":           TARGET,
    "classes":          {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"},
    "best_params":      BEST_PARAMS,
    "cv_f1_macro_mean": float(cv_scores.mean()),
    "cv_f1_macro_std":  float(cv_scores.std()),
    "n_train":          len(X),
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/xgb_pipeline.joblib")
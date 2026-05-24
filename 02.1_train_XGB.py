"""
02.1_train_XGB.py
=================
Uso:
    python 02.1_train_XGB.py --data processed_v2
    python 02.1_train_XGB.py --data processed_multiclass
    python 02.1_train_XGB.py --data processed_binary
"""

import argparse
import pandas as pd
import numpy as np
import os, json, joblib
from xgboost                    import XGBClassifier
from sklearn.model_selection    import StratifiedKFold, train_test_split
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import classification_report, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True,
                    choices=["processed_v2", "processed_multiclass", "processed_binary"])
args = parser.parse_args()

DATA_PATH = f"data/{args.data}.parquet"
MODEL_DIR = f"models/{args.data}"
os.makedirs(MODEL_DIR, exist_ok=True)

IS_BINARY   = args.data == "processed_binary"
TARGET      = "success_class"
CLASS_NAMES = ["Fracaso", "Éxito"] if IS_BINARY else ["Fracaso", "Moderado", "Alto éxito"]
NUM_CLASS   = 2 if IS_BINARY else 3

print(f"\n{'='*55}")
print(f"  Dataset : {DATA_PATH}")
print(f"  Modo    : {'Binario' if IS_BINARY else 'Multiclase'}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────
# HIPERPARÁMETROS
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
# 1. CARGA Y FEATURES
# ─────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df = df[(df["is_indie"] == 1) & (df["estimated_owners_num"] > 0)].copy()
rev_col = "num_reviews_total" if "num_reviews_total" in df.columns else "total_reviews"
df = df[df[rev_col] > 100].copy()

NUMERIC_BASE_V2 = [
    "price", "is_free", "required_age",
    "plat_windows", "plat_mac", "plat_linux",
    "is_multiplayer", "n_languages",
    "desc_length", "release_year", "release_quarter",
    "publisher_success_avg",
    "price_per_lang", "n_tags", "n_platforms",
    "mp_x_free", "action_singleplayer",
    "desc_length_log", "publisher_tier",
]
NUMERIC_BASE_NEW = [
    "price", "is_free", "required_age",
    "plat_windows", "plat_mac", "plat_linux",
    "is_multiplayer", "languages_market_score",
    "n_screenshots", "n_movies", "n_achievements",
    "release_quarter", "publisher_success_avg",
]

NUMERIC_BASE = NUMERIC_BASE_V2 if args.data == "processed_v2" \
               else [f for f in NUMERIC_BASE_NEW if f in df.columns]

GENRE_COLS = [c for c in df.columns if c.startswith("genre_")]
TAG_COLS   = [c for c in df.columns if c.startswith("tag_")]
NUMERIC    = NUMERIC_BASE + GENRE_COLS + TAG_COLS

df_model          = df[NUMERIC + [TARGET]].dropna(subset=[TARGET])
df_model[NUMERIC] = df_model[NUMERIC].fillna(0)

X = df_model[NUMERIC]
y = df_model[TARGET].astype(int)

print("Distribución de clases:")
for k, v in y.value_counts().sort_index().items():
    print(f"  {CLASS_NAMES[k]}: {v} ({v/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 2. CV MANUAL
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores    = []
train_scores = []
auc_scores   = [] if IS_BINARY else None

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw = compute_sample_weight("balanced", y_tr)

    clf_fold = XGBClassifier(
        **BEST_PARAMS,
        objective             = "binary:logistic" if IS_BINARY else "multi:softprob",
        num_class             = None if IS_BINARY else NUM_CLASS,
        eval_metric           = "logloss" if IS_BINARY else "mlogloss",
        early_stopping_rounds = 30,
        random_state          = 42,
        n_jobs                = -1,
    )
    clf_fold.fit(X_tr, y_tr, sample_weight=sw,
                 eval_set=[(X_val, y_val)], verbose=False)

    val_f1   = f1_score(y_val, clf_fold.predict(X_val), average="macro")
    train_f1 = f1_score(y_tr,  clf_fold.predict(X_tr),  average="macro")
    cv_scores.append(val_f1)
    train_scores.append(train_f1)

    if IS_BINARY:
        auc = roc_auc_score(y_val, clf_fold.predict_proba(X_val)[:, 1])
        auc_scores.append(auc)
        print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
              f"gap={train_f1-val_f1:.3f}  AUC={auc:.3f}  (iter: {clf_fold.best_iteration})")
    else:
        print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
              f"gap={train_f1-val_f1:.3f}  (iter: {clf_fold.best_iteration})")

cv_scores    = np.array(cv_scores)
train_scores = np.array(train_scores)
print(f"\n  Val  F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Train F1-macro: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
print(f"  Gap medio:      {(train_scores - cv_scores).mean():.3f}")
if IS_BINARY:
    auc_scores = np.array(auc_scores)
    print(f"  AUC-ROC:        {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# ─────────────────────────────────────────────
# 3. FIT FINAL
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)
sw_tr_f = compute_sample_weight("balanced", y_tr_f)

final_clf = XGBClassifier(
    **BEST_PARAMS,
    objective             = "binary:logistic" if IS_BINARY else "multi:softprob",
    num_class             = None if IS_BINARY else NUM_CLASS,
    eval_metric           = "logloss" if IS_BINARY else "mlogloss",
    early_stopping_rounds = 30,
    random_state          = 42,
    n_jobs                = -1,
)
final_clf.fit(X_tr_f, y_tr_f, sample_weight=sw_tr_f,
              eval_set=[(X_val_f, y_val_f)], verbose=False)
print(f"  Mejor iteración: {final_clf.best_iteration}")

final_pipeline = Pipeline([("clf", final_clf)])
y_pred = final_pipeline.predict(X)
print("\nClassification report (train — referencia):")
print(classification_report(y, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 4. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(final_pipeline, f"{MODEL_DIR}/xgb_pipeline.joblib")

metadata = {
    "numeric_features": NUMERIC,
    "target":           TARGET,
    "classes":          {i: n for i, n in enumerate(CLASS_NAMES)},
    "is_binary":        IS_BINARY,
    "best_params":      BEST_PARAMS,
    "cv_f1_macro_mean": float(cv_scores.mean()),
    "cv_f1_macro_std":  float(cv_scores.std()),
    "cv_auc_mean":      float(auc_scores.mean()) if IS_BINARY else None,
    "n_train":          len(X),
    "data_version":     args.data,
}
with open(f"{MODEL_DIR}/xgb_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/xgb_pipeline.joblib")
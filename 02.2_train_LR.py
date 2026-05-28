"""
02.2_train_LR.py
================
Uso:
    python 02.2_train_LR.py --data processed_multiclass
    python 02.2_train_LR.py --data processed_binary
"""

import argparse
import pandas as pd
import numpy as np
import os, json, joblib
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import StratifiedKFold
from sklearn.pipeline           import Pipeline
from sklearn.preprocessing      import StandardScaler
from sklearn.metrics            import classification_report, f1_score, roc_auc_score

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

print(f"\n{'='*55}")
print(f"  Dataset : {DATA_PATH}")
print(f"  Modo    : {'Binario' if IS_BINARY else 'Multiclase'}")
print(f"{'='*55}\n")

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
# 2. PIPELINE
# ─────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        multi_class  = "auto",
        solver       = "lbfgs",
        max_iter     = 1000,
        C            = 1.0,
        class_weight = "balanced",
        random_state = 42,
        n_jobs       = -1,
    )),
])

# ─────────────────────────────────────────────
# 3. CV MANUAL
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores    = []
train_scores = []
auc_scores   = [] if IS_BINARY else None

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    pipeline.fit(X_tr, y_tr)
    val_f1   = f1_score(y_val, pipeline.predict(X_val), average="macro")
    train_f1 = f1_score(y_tr,  pipeline.predict(X_tr),  average="macro")
    cv_scores.append(val_f1)
    train_scores.append(train_f1)

    if IS_BINARY:
        auc = roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])
        auc_scores.append(auc)
        print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
              f"gap={train_f1-val_f1:.3f}  AUC={auc:.3f}")
    else:
        print(f"  Fold {fold+1}: train={train_f1:.3f}  val={val_f1:.3f}  "
              f"gap={train_f1-val_f1:.3f}")

cv_scores    = np.array(cv_scores)
train_scores = np.array(train_scores)
print(f"\n  Val  F1-macro: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"  Train F1-macro: {train_scores.mean():.3f} ± {train_scores.std():.3f}")
print(f"  Gap medio:      {(train_scores - cv_scores).mean():.3f}")
if IS_BINARY:
    auc_scores = np.array(auc_scores)
    print(f"  AUC-ROC:        {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# ─────────────────────────────────────────────
# 4. FIT FINAL
# ─────────────────────────────────────────────
print("\nEntrenando modelo final…")
pipeline.fit(X, y)

y_pred = pipeline.predict(X)
print("\nClassification report (train — referencia):")
print(classification_report(y, y_pred, target_names=CLASS_NAMES))

# ─────────────────────────────────────────────
# 5. GUARDAR
# ─────────────────────────────────────────────
joblib.dump(pipeline, f"{MODEL_DIR}/lr_pipeline.joblib")

metadata = {
    "numeric_features": NUMERIC,
    "target":           TARGET,
    "classes":          {i: n for i, n in enumerate(CLASS_NAMES)},
    "is_binary":        IS_BINARY,
    "cv_f1_macro_mean": float(cv_scores.mean()),
    "cv_f1_macro_std":  float(cv_scores.std()),
    "cv_auc_mean":      float(auc_scores.mean()) if IS_BINARY else None,
    "n_train":          len(X),
    "data_version":     args.data,
}
with open(f"{MODEL_DIR}/lr_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/lr_pipeline.joblib")
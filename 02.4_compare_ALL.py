"""
02.4_compare_ALL.py
===================
Uso:
    python 02.4_compare_ALL.py --data processed_multiclass
    python 02.4_compare_ALL.py --data processed_binary
"""

import argparse
import pandas as pd
import numpy as np
import os, json, joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from xgboost                    import XGBClassifier
from sklearn.model_selection    import StratifiedKFold
from sklearn.metrics            import (f1_score, confusion_matrix,
                                        ConfusionMatrixDisplay, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True,
                    choices=["processed_v2", "processed_multiclass", "processed_binary"])
args = parser.parse_args()

DATA_PATH   = f"data/{args.data}.parquet"
MODEL_DIR   = f"models/{args.data}"
OUT_DIR     = f"outputs/{args.data}"
os.makedirs(OUT_DIR, exist_ok=True)

IS_BINARY   = args.data == "processed_binary"
TARGET      = "success_class"
CLASS_NAMES = ["Fracaso", "Éxito"] if IS_BINARY else ["Fracaso", "Moderado", "Alto éxito"]
NUM_CLASS   = 2 if IS_BINARY else 3

COLORS = {"RF": "#3498db", "XGB": "#e74c3c", "LR": "#2ecc71", "LGBM": "#9b59b6"}

XGB_PARAMS = dict(
    subsample        = 0.6,
    reg_lambda       = 1.0,
    reg_alpha        = 0.1,
    n_estimators     = 600,
    min_child_weight = 100,
    max_depth        = 3,
    learning_rate    = 0.03,
    gamma            = 1.0,
    colsample_bytree = 0.4,
    objective        = "binary:logistic" if IS_BINARY else "multi:softprob",
    num_class        = None if IS_BINARY else NUM_CLASS,
    eval_metric      = "logloss" if IS_BINARY else "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)

LGBM_PARAMS = dict(
    n_estimators      = 700,
    max_depth         = 3,
    learning_rate     = 0.03,
    subsample         = 0.7,
    colsample_bytree  = 0.4,
    min_child_samples = 100,
    reg_alpha         = 0.1,
    reg_lambda        = 10.0,
    class_weight      = "balanced",
    objective         = "binary" if IS_BINARY else "multiclass",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)

print(f"\n{'='*55}")
print(f"  Dataset : {DATA_PATH}")
print(f"  Modo    : {'Binario' if IS_BINARY else 'Multiclase'}")
print(f"{'='*55}\n")

# ─────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df = df[(df["is_indie"] == 1) & (df["estimated_owners_num"] > 0)].copy()
rev_col = "num_reviews_total" if "num_reviews_total" in df.columns else "total_reviews"
df = df[df[rev_col] > 100].copy()

with open(f"{MODEL_DIR}/lgbm_metadata.json") as f:
    meta = json.load(f)

NUMERIC    = meta["numeric_features"]
GENRE_COLS = [c for c in df.columns if c.startswith("genre_")]
TAG_COLS   = [c for c in df.columns if c.startswith("tag_")]

df_model          = df[NUMERIC + [TARGET]].dropna(subset=[TARGET])
df_model[NUMERIC] = df_model[NUMERIC].fillna(0)

X = df_model[NUMERIC]
y = df_model[TARGET].astype(int)

print(f"  Dataset: {len(X)} filas, {len(NUMERIC)} features")
for k, v in y.value_counts().sort_index().items():
    print(f"  {CLASS_NAMES[k]}: {v} ({v/len(y)*100:.1f}%)")

rf_pipeline = joblib.load(f"{MODEL_DIR}/rf_pipeline.joblib")
lr_pipeline = joblib.load(f"{MODEL_DIR}/lr_pipeline.joblib")

# ─────────────────────────────────────────────
# 2. CV MANUAL
# ─────────────────────────────────────────────
print("\nCV 5-fold para los 4 modelos…")
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: [] for name in COLORS}

from sklearn.model_selection import train_test_split

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw = compute_sample_weight("balanced", y_tr)

    # RF
    rf_pipeline.fit(X_tr, y_tr)
    y_pred_rf = rf_pipeline.predict(X_val)

    # XGB
    clf_xgb = XGBClassifier(**XGB_PARAMS)
    clf_xgb.fit(X_tr, y_tr, sample_weight=sw)
    y_pred_xgb = clf_xgb.predict(X_val)

    # LR
    lr_pipeline.fit(X_tr, y_tr)
    y_pred_lr = lr_pipeline.predict(X_val)

    # LGBM
    clf_lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf_lgbm.fit(X_tr, y_tr,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=-1)])
    y_pred_lgbm = clf_lgbm.predict(X_val)

    for name, y_pred in [("RF", y_pred_rf), ("XGB", y_pred_xgb),
                          ("LR", y_pred_lr), ("LGBM", y_pred_lgbm)]:
        pipe_obj = rf_pipeline if name == "RF" else \
                   clf_xgb    if name == "XGB" else \
                   lr_pipeline if name == "LR" else clf_lgbm

        f1_pc = f1_score(y_val, y_pred, average=None)
        entry  = {
            "f1_macro":    f1_score(y_val, y_pred, average="macro"),
            "f1_class_0":  f1_pc[0],
            "f1_class_1":  f1_pc[1],
        }
        if not IS_BINARY:
            entry["f1_class_2"] = f1_pc[2]
        if IS_BINARY:
            proba = pipe_obj.predict_proba(X_val)[:, 1]
            entry["auc"] = roc_auc_score(y_val, proba)
        results[name].append(entry)

    line = " | ".join(f"{n}: {results[n][-1]['f1_macro']:.3f}" for n in COLORS)
    print(f"  Fold {fold+1} — {line}")

# ─────────────────────────────────────────────
# 3. PREDICCIONES FINALES
# ─────────────────────────────────────────────
print("\nEntrenando modelos finales…")
sw_full      = compute_sample_weight("balanced", y)
y_preds_full = {}

rf_pipeline.fit(X, y);  y_preds_full["RF"]  = rf_pipeline.predict(X)
lr_pipeline.fit(X, y);  y_preds_full["LR"]  = lr_pipeline.predict(X)

clf_xgb_full = XGBClassifier(**XGB_PARAMS)
clf_xgb_full.fit(X, y, sample_weight=sw_full)
y_preds_full["XGB"] = clf_xgb_full.predict(X)

X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42)
clf_lgbm_full = lgb.LGBMClassifier(**LGBM_PARAMS)
clf_lgbm_full.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                              lgb.log_evaluation(period=-1)])
y_preds_full["LGBM"] = clf_lgbm_full.predict(X)

# ─────────────────────────────────────────────
# 4. FIGURA 1 — F1-macro por fold
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for name, color in COLORS.items():
    scores = [r["f1_macro"] for r in results[name]]
    mean   = np.mean(scores)
    ax.plot(range(1, 6), scores, "o-", color=color,
            label=f"{name} (μ={mean:.3f})", linewidth=2, markersize=6)
    ax.axhline(mean, color=color, linestyle="--", alpha=0.35)
ax.set_title(f"F1-macro por fold — {args.data}", fontsize=13)
ax.set_xlabel("Fold"); ax.set_ylabel("F1-macro")
ax.set_xticks(range(1, 6)); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/compare4_f1_folds.png", dpi=150); plt.close()
print(f"✅ {OUT_DIR}/compare4_f1_folds.png")

# ─────────────────────────────────────────────
# 5. FIGURA 2 — F1 por clase
# ─────────────────────────────────────────────
f1_keys   = ["f1_class_0", "f1_class_1"] + (["f1_class_2"] if not IS_BINARY else [])
x     = np.arange(len(CLASS_NAMES))
width = 0.2

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, color) in enumerate(COLORS.items()):
    means = [np.mean([r[k] for r in results[name]]) for k in f1_keys]
    stds  = [np.std( [r[k] for r in results[name]]) for k in f1_keys]
    ax.bar(x + i*width, means, width, label=name, color=color,
           yerr=stds, capsize=4, alpha=0.85)
ax.set_title(f"F1 por clase — {args.data}", fontsize=13)
ax.set_ylabel("F1-score")
ax.set_xticks(x + width * 1.5); ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/compare4_f1_per_class.png", dpi=150); plt.close()
print(f"✅ {OUT_DIR}/compare4_f1_per_class.png")

# ─────────────────────────────────────────────
# 6. FIGURA 3 — AUC-ROC (solo binario)
# ─────────────────────────────────────────────
if IS_BINARY:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in COLORS.items():
        aucs = [r["auc"] for r in results[name]]
        mean = np.mean(aucs)
        ax.plot(range(1, 6), aucs, "o-", color=color,
                label=f"{name} (μ={mean:.3f})", linewidth=2, markersize=6)
        ax.axhline(mean, color=color, linestyle="--", alpha=0.35)
    ax.set_title("AUC-ROC por fold — binario", fontsize=13)
    ax.set_xlabel("Fold"); ax.set_ylabel("AUC-ROC")
    ax.set_xticks(range(1, 6)); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/compare4_auc_folds.png", dpi=150); plt.close()
    print(f"✅ {OUT_DIR}/compare4_auc_folds.png")

# ─────────────────────────────────────────────
# 7. FIGURA 4 — Matrices de confusión normalizadas
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle(f"Matrices de confusión normalizadas — {args.data}", fontsize=13)
for ax, name in zip(axes, COLORS):
    cm = confusion_matrix(y, y_preds_full[name], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/compare4_confusion_normalized.png", dpi=150); plt.close()
print(f"✅ {OUT_DIR}/compare4_confusion_normalized.png")

# ─────────────────────────────────────────────
# 8. RESUMEN TABULAR
# ─────────────────────────────────────────────
metric_labels = ["F1-macro"] + [f"F1 {c}" for c in CLASS_NAMES]
metric_keys2  = ["f1_macro"] + f1_keys
if IS_BINARY:
    metric_labels.append("AUC-ROC")
    metric_keys2.append("auc")

print(f"\n{'='*68}")
print(f"{'MÉTRICA':<28} {'RF':>8} {'XGB':>8} {'LR':>8} {'LGBM':>8}")
print(f"{'='*68}")
for label, key in zip(metric_labels, metric_keys2):
    vals   = {n: np.mean([r[key] for r in results[n]]) for n in COLORS}
    winner = max(vals, key=vals.get)
    row    = "  " + f"{label:<26}"
    for n in COLORS:
        mark = "*" if n == winner else " "
        row += f" {vals[n]:>7.3f}{mark}"
    print(row)
print(f"{'='*68}")
print("* = mejor modelo en esa métrica\n")
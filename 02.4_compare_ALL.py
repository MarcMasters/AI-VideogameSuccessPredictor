"""
02.4_compare_ALL.py
=================
Compara RF vs XGBoost vs Logistic Regression.
Genera gráficos en outputs/compare3_*.png
"""

import pandas as pd
import numpy as np
import os, json, joblib
import matplotlib.pyplot as plt
from xgboost                    import XGBClassifier
from sklearn.model_selection    import StratifiedKFold
from sklearn.metrics            import (
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight

os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/processed.parquet"
MODELS    = {
    "RF":  "models/rf_pipeline.joblib",
    "XGB": "models/xgb_pipeline.joblib",
    "LR":  "models/lr_pipeline.joblib",
}
META_PATH   = "models/metadata.json"
CLASS_NAMES = ["Fracaso", "Moderado", "Alto éxito"]
COLORS      = {"RF": "#3498db", "XGB": "#e74c3c", "LR": "#2ecc71"}

XGB_PARAMS = dict(
    subsample        = 0.6,
    reg_lambda       = 1.0,
    reg_alpha        = 0.1,
    n_estimators     = 400,
    min_child_weight = 5,
    max_depth        = 5,
    learning_rate    = 0.03,
    gamma            = 1.0,
    colsample_bytree = 0.6,
    objective        = "multi:softprob",
    num_class        = 3,
    eval_metric      = "mlogloss",
    random_state     = 42,
    n_jobs           = -1,
)

# ─────────────────────────────────────────────
# 1. DATOS
# ─────────────────────────────────────────────
print("Cargando datos…")
df = pd.read_parquet(DATA_PATH)
df = df[
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
].copy()

with open(META_PATH) as f:
    meta = json.load(f)

NUMERIC    = meta["numeric_features"]
GENRE_COLS = [c for c in df.columns if c.startswith("genre_")]
TAG_COLS   = [c for c in df.columns if c.startswith("tag_")]
TARGET     = "success_class"

df_model             = df[NUMERIC + [TARGET]].dropna(subset=[TARGET])
df_model[GENRE_COLS] = df_model[GENRE_COLS].fillna(0)
df_model[TAG_COLS]   = df_model[TAG_COLS].fillna(0)
df_model[NUMERIC]    = df_model[NUMERIC].fillna(0)

X = df_model[NUMERIC]
y = df_model[TARGET].astype(int)

pipelines = {name: joblib.load(path) for name, path in MODELS.items()}

# ─────────────────────────────────────────────
# 2. CV MANUAL — mismos splits para los 3
# ─────────────────────────────────────────────
print("\nCV 5-fold para los 3 modelos…")
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: [] for name in ["RF", "XGB", "LR"]}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw = compute_sample_weight("balanced", y_tr)

    # RF
    pipelines["RF"].fit(X_tr, y_tr)
    y_pred_rf = pipelines["RF"].predict(X_val)

    # XGB — instancia nueva sin early_stopping
    clf_xgb = XGBClassifier(**XGB_PARAMS)
    clf_xgb.fit(X_tr, y_tr, sample_weight=sw)
    y_pred_xgb = clf_xgb.predict(X_val)

    # LR
    pipelines["LR"].fit(X_tr, y_tr)
    y_pred_lr = pipelines["LR"].predict(X_val)

    for name, y_pred in [("RF", y_pred_rf), ("XGB", y_pred_xgb), ("LR", y_pred_lr)]:
        f1_pc = f1_score(y_val, y_pred, average=None)
        results[name].append({
            "f1_macro":    f1_score(y_val, y_pred, average="macro"),
            "f1_fracaso":  f1_pc[0],
            "f1_moderado": f1_pc[1],
            "f1_exito":    f1_pc[2],
        })

    line = " | ".join(f"{n}: {results[n][-1]['f1_macro']:.3f}" for n in ["RF", "XGB", "LR"])
    print(f"  Fold {fold+1} — {line}")

# ─────────────────────────────────────────────
# PREDICCIONES FINALES
# ─────────────────────────────────────────────
y_preds_full = {}

pipelines["RF"].fit(X, y)
y_preds_full["RF"] = pipelines["RF"].predict(X)

clf_xgb_full = XGBClassifier(**XGB_PARAMS)
sw_full = compute_sample_weight("balanced", y)
clf_xgb_full.fit(X, y, sample_weight=sw_full)
y_preds_full["XGB"] = clf_xgb_full.predict(X)

pipelines["LR"].fit(X, y)
y_preds_full["LR"] = pipelines["LR"].predict(X)

# ─────────────────────────────────────────────
# 3. FIGURA 1 — F1-macro por fold
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

for name, color in COLORS.items():
    scores = [r["f1_macro"] for r in results[name]]
    mean   = np.mean(scores)
    ax.plot(range(1, 6), scores, "o-", color=color,
            label=f"{name} (μ={mean:.3f})", linewidth=2, markersize=6)
    ax.axhline(mean, color=color, linestyle="--", alpha=0.35)

ax.set_title("F1-macro por fold — RF vs XGB vs LR", fontsize=13)
ax.set_xlabel("Fold")
ax.set_ylabel("F1-macro")
ax.set_xticks(range(1, 6))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/compare3_f1_folds.png", dpi=150)
plt.close()
print("✅ outputs/compare3_f1_folds.png")

# ─────────────────────────────────────────────
# 4. FIGURA 2 — F1 por clase (barras agrupadas)
# ─────────────────────────────────────────────
metric_keys = ["f1_fracaso", "f1_moderado", "f1_exito"]
x     = np.arange(len(CLASS_NAMES))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, (name, color) in enumerate(COLORS.items()):
    means = [np.mean([r[m] for r in results[name]]) for m in metric_keys]
    stds  = [np.std( [r[m] for r in results[name]]) for m in metric_keys]
    ax.bar(x + i*width, means, width, label=name, color=color,
           yerr=stds, capsize=4, alpha=0.85)

ax.set_title("F1 por clase — media CV (con desviación estándar)", fontsize=13)
ax.set_ylabel("F1-score")
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/compare3_f1_per_class.png", dpi=150)
plt.close()
print("✅ outputs/compare3_f1_per_class.png")

# ─────────────────────────────────────────────
# 5. FIGURA 3 — Matrices de confusión normalizadas
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Matrices de confusión normalizadas (% por fila)", fontsize=13)

for ax, name in zip(axes, ["RF", "XGB", "LR"]):
    cm = confusion_matrix(y, y_preds_full[name], normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig("outputs/compare3_confusion_normalized.png", dpi=150)
plt.close()
print("✅ outputs/compare3_confusion_normalized.png")

# ─────────────────────────────────────────────
# 6. RESUMEN TABULAR
# ─────────────────────────────────────────────
metric_labels = ["F1-macro", "F1 Fracaso", "F1 Moderado", "F1 Alto éxito"]
metric_keys2  = ["f1_macro", "f1_fracaso", "f1_moderado", "f1_exito"]

print("\n" + "="*60)
print(f"{'MÉTRICA':<28} {'RF':>8} {'XGB':>8} {'LR':>8}")
print("="*60)
for label, key in zip(metric_labels, metric_keys2):
    vals   = {n: np.mean([r[key] for r in results[n]]) for n in ["RF", "XGB", "LR"]}
    winner = max(vals, key=vals.get)
    row    = "  " + f"{label:<26}"
    for n in ["RF", "XGB", "LR"]:
        mark = "*" if n == winner else " "
        row += f" {vals[n]:>7.3f}{mark}"
    print(row)
print("="*60)
print("* = mejor modelo en esa métrica\n")
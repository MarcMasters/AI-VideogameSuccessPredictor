"""
04_compare_models.py
====================
Compara RF (02_train.py) vs XGBoost (02.1_train.py).
Genera gráficos en outputs/compare_*.png
"""

import pandas as pd
import numpy as np
import os, json, joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection  import StratifiedKFold
from sklearn.metrics          import (
    f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)

DATA_PATH = "data/processed.parquet"
RF_PATH   = "models/rf_pipeline.joblib"
XGB_PATH  = "models/xgb_pipeline.joblib"
META_PATH = "models/metadata.json"

CLASS_NAMES = ["Fracaso", "Moderado", "Alto éxito"]
COLORS      = {"RF": "#3498db", "XGB": "#e74c3c"}

# ─────────────────────────────────────────────
# 1. CARGA DE DATOS Y MODELOS
# ─────────────────────────────────────────────
print("Cargando datos y modelos…")
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

rf_pipeline  = joblib.load(RF_PATH)
xgb_pipeline = joblib.load(XGB_PATH)

# ─────────────────────────────────────────────
# 2. CV MANUAL PARA AMBOS MODELOS
#    (mismo split para comparación justa)
# ─────────────────────────────────────────────
print("\nEjecutando CV comparativa (5-fold)…")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {"RF": [], "XGB": []}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    sw = compute_sample_weight("balanced", y_tr)

    # — RF —
    rf_pipeline.fit(X_tr, y_tr)
    y_pred_rf = rf_pipeline.predict(X_val)
    results["RF"].append({
        "f1_macro":  f1_score(y_val, y_pred_rf, average="macro"),
        "f1_fracaso": f1_score(y_val, y_pred_rf, average=None)[0],
        "f1_moderado": f1_score(y_val, y_pred_rf, average=None)[1],
        "f1_exito":   f1_score(y_val, y_pred_rf, average=None)[2],
        "y_val": y_val, "y_pred": y_pred_rf,
    })

    # — XGB —
    xgb_pipeline.fit(X_tr, y_tr, clf__sample_weight=sw)
    y_pred_xgb = xgb_pipeline.predict(X_val)
    results["XGB"].append({
        "f1_macro":   f1_score(y_val, y_pred_xgb, average="macro"),
        "f1_fracaso": f1_score(y_val, y_pred_xgb, average=None)[0],
        "f1_moderado": f1_score(y_val, y_pred_xgb, average=None)[1],
        "f1_exito":   f1_score(y_val, y_pred_xgb, average=None)[2],
        "y_val": y_val, "y_pred": y_pred_xgb,
    })

    print(f"  Fold {fold+1} — RF: {results['RF'][-1]['f1_macro']:.3f} | XGB: {results['XGB'][-1]['f1_macro']:.3f}")

# Predicciones sobre todo el dataset (para confusion matrix global)
y_pred_rf_full  = rf_pipeline.predict(X)
y_pred_xgb_full = xgb_pipeline.predict(X)

# ─────────────────────────────────────────────
# 3. FIGURA 1 — F1-macro por fold + media
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

for name, color in COLORS.items():
    scores = [r["f1_macro"] for r in results[name]]
    mean   = np.mean(scores)
    ax.plot(range(1, 6), scores, "o-", color=color, label=f"{name} (media={mean:.3f})", linewidth=2)
    ax.axhline(mean, color=color, linestyle="--", alpha=0.4)

ax.set_title("F1-macro por fold (CV 5-fold estratificado)", fontsize=13)
ax.set_xlabel("Fold")
ax.set_ylabel("F1-macro")
ax.set_xticks(range(1, 6))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/compare_f1_folds.png", dpi=150)
plt.close()
print("✅ outputs/compare_f1_folds.png")

# ─────────────────────────────────────────────
# 4. FIGURA 2 — F1 por clase (barras agrupadas)
# ─────────────────────────────────────────────
metrics = ["f1_fracaso", "f1_moderado", "f1_exito"]
labels  = ["Fracaso", "Moderado", "Alto éxito"]

fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(labels))
width  = 0.35

for i, (name, color) in enumerate(COLORS.items()):
    means = [np.mean([r[m] for r in results[name]]) for m in metrics]
    stds  = [np.std([r[m]  for r in results[name]]) for m in metrics]
    bars  = ax.bar(x + i*width, means, width, label=name, color=color,
                   yerr=stds, capsize=4, alpha=0.85)

ax.set_title("F1 por clase — media CV (con desviación estándar)", fontsize=13)
ax.set_ylabel("F1-score")
ax.set_xticks(x + width/2)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/compare_f1_per_class.png", dpi=150)
plt.close()
print("✅ outputs/compare_f1_per_class.png")

# ─────────────────────────────────────────────
# 5. FIGURA 3 — Matrices de confusión (train full)
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrices de confusión (dataset completo)", fontsize=13)

for ax, (name, y_pred) in zip(axes, [("RF", y_pred_rf_full), ("XGB", y_pred_xgb_full)]):
    cm = confusion_matrix(y, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig("outputs/compare_confusion_matrices.png", dpi=150)
plt.close()
print("✅ outputs/compare_confusion_matrices.png")

# ─────────────────────────────────────────────
# 6. FIGURA 4 — Matrices de confusión normalizadas
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrices de confusión normalizadas (% por fila)", fontsize=13)

for ax, (name, y_pred) in zip(axes, [("RF", y_pred_rf_full), ("XGB", y_pred_xgb_full)]):
    cm = confusion_matrix(y, y_pred, normalize="true")
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(name)

plt.tight_layout()
plt.savefig("outputs/compare_confusion_normalized.png", dpi=150)
plt.close()
print("✅ outputs/compare_confusion_normalized.png")

# ─────────────────────────────────────────────
# 7. FIGURA 5 — Resumen tabular impreso
# ─────────────────────────────────────────────
print("\n" + "="*55)
print(f"{'MÉTRICA':<30} {'RF':>10} {'XGB':>10}")
print("="*55)

metric_keys = {
    "F1-macro (CV media)": "f1_macro",
    "F1 Fracaso (CV)":     "f1_fracaso",
    "F1 Moderado (CV)":    "f1_moderado",
    "F1 Alto éxito (CV)":  "f1_exito",
}

for label, key in metric_keys.items():
    rf_mean  = np.mean([r[key] for r in results["RF"]])
    xgb_mean = np.mean([r[key] for r in results["XGB"]])
    winner   = "←" if xgb_mean > rf_mean else ""
    print(f"  {label:<28} {rf_mean:>8.3f} {xgb_mean:>10.3f}  {winner}")

print("="*55)
print("\nClassification report RF (train):")
print(classification_report(y, y_pred_rf_full, target_names=CLASS_NAMES))
print("Classification report XGB (train):")
print(classification_report(y, y_pred_xgb_full, target_names=CLASS_NAMES))
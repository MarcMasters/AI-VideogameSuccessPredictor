"""
02_train.py
===========
Entrena un RandomForestClassifier con las features pre-lanzamiento.
Guarda el modelo + el pipeline de encoding en models/
"""

import pandas as pd
import numpy as np
import re, os, json, joblib
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.metrics           import (classification_report,
                                       confusion_matrix,
                                       ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

DATA_PATH  = "data/processed.parquet"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
print("Cargando datos preprocesados…")
df = pd.read_parquet(DATA_PATH)
print(f"  Filas cargadas: {len(df)}")

# Entrenamos solo con indies activos
# df = df[df["is_indie"] == 1].copy()
df = df[
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
].copy()

# ─────────────────────────────────────────────
# 2. FEATURES SELECTION
# ─────────────────────────────────────────────
NUMERIC_BASE = [
    "price", "is_free", "required_age",
    "plat_windows", "plat_mac", "plat_linux",
    "is_multiplayer", "n_languages",
    "desc_length", "release_year", "release_quarter",
    "publisher_success_avg",   # media de éxito del publisher en juegos indie
]

GENRE_COLS  = [c for c in df.columns if c.startswith("genre_")]
TAG_COLS    = [c for c in df.columns if c.startswith("tag_")]
NUMERIC     = NUMERIC_BASE + GENRE_COLS + TAG_COLS
CATEGORICAL = []   # publisher ya es numérico, no necesitamos one-hot

TARGET = "success_class"

# Drop NAs en features
feature_cols = NUMERIC + CATEGORICAL
df_model = df[feature_cols + [TARGET]].dropna(subset=[TARGET])

# Rellena NAs numéricos con 0
df_model[NUMERIC] = df_model[NUMERIC].fillna(0)
df_model[CATEGORICAL] = df_model[CATEGORICAL].fillna("Unknown")

X = df_model[feature_cols]
y = df_model[TARGET].astype(int)

print(f"\nDistribución de clases en dataset de entrenamiento:")
vc = y.value_counts().sort_index()
for k, v in vc.items():
    label = {0:"Fracaso", 1:"Moderado", 2:"Alto éxito"}[k]
    print(f"  {label}: {v} ({v/len(y)*100:.1f}%)")

# ─────────────────────────────────────────────
# 3. PIPELINE
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC),
])

clf = RandomForestClassifier(
    n_estimators    = 400,
    max_depth       = 15,   # limita profundidad para reducir overfitting
    min_samples_leaf= 20,   # sube de 5 a 20: cada hoja necesita más ejemplos
    max_features    = "sqrt",
    class_weight    = "balanced",
    random_state    = 42,
    n_jobs          = -1,
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf",  clf),
])

# ─────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\nCross-validation (5-fold estratificado)…")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
print(f"  F1-macro: {scores.mean():.3f} ± {scores.std():.3f}")

# ─────────────────────────────────────────────
# 5. FIT FINAL
# ─────────────────────────────────────────────
print("\nEntrenando modelo final sobre todos los datos…")
pipeline.fit(X, y)

# ─────────────────────────────────────────────
# 6. MÉTRICAS & GRÁFICOS
# ─────────────────────────────────────────────
y_pred = pipeline.predict(X)
print("\nClassification report (train — referencia):")
print(classification_report(y, y_pred,
      target_names=["Fracaso", "Moderado", "Alto éxito"]))

# Confusion matrix
cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Fracaso", "Moderado", "Alto éxito"]).plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix (train)")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# Feature importances
rf_model       = pipeline.named_steps["clf"]
all_feat_names = NUMERIC

importances = pd.Series(rf_model.feature_importances_, index=all_feat_names)
top20 = importances.nlargest(20)

# Diagnóstico: ¿qué publishers tienen mayor publisher_success_avg?
pub_scores = df[["publisher_clean", "publisher_success_avg"]].drop_duplicates()
print(pub_scores.sort_values("publisher_success_avg", ascending=False).head(20))

fig, ax = plt.subplots(figsize=(8, 6))
top20.sort_values().plot(kind="barh", ax=ax, color="#2ecc71")
ax.set_title("Top 20 Feature Importances")
ax.set_xlabel("Importancia (Gini)")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png", dpi=150)
plt.close()
print("  Gráficos guardados en outputs/")

# ─────────────────────────────────────────────
# 7. GUARDAR MODELO + METADATA
# ─────────────────────────────────────────────
joblib.dump(pipeline, f"{MODEL_DIR}/rf_pipeline.joblib")

metadata = {
    "numeric_features":     NUMERIC,
    "categorical_features": CATEGORICAL,
    "target":               TARGET,
    "classes":              {0:"Fracaso", 1:"Moderado", 2:"Alto éxito"},
    "cv_f1_macro_mean":     float(scores.mean()),
    "cv_f1_macro_std":      float(scores.std()),
    "n_train":              len(X),
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Modelo guardado en {MODEL_DIR}/rf_pipeline.joblib")

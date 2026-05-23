"""
01_preprocess.py
================
Limpieza, ingeniería de features y cálculo del success_score.
Ajustado para contexto INDIE:
  - Normalización basada en percentiles del subconjunto indie
  - Umbrales de clase por percentil (p50 / p80) en lugar de fijos
  - Publisher: feature numérica de media de éxito (opción 2)
Genera: data/processed.parquet
"""

import pandas as pd
import numpy as np
import ast, re, os, json

RAW_PATH = "data/games.csv"
OUT_PATH = "data/processed.parquet"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def parse_list_col(s):
    if pd.isna(s): return []
    try:
        return [x.strip() for x in ast.literal_eval(s)]
    except:
        return [x.strip() for x in str(s).split(",")]

def parse_owners(s):
    if pd.isna(s): return np.nan
    m = re.findall(r"\d+", str(s).replace(",", ""))
    if len(m) >= 2: return (int(m[0]) + int(m[1])) / 2
    elif len(m) == 1: return int(m[0])
    return np.nan

# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
print("Cargando dataset…")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"  Shape inicial: {df.shape}")

# ─────────────────────────────────────────────
# 2. COLUMNAS TARGET (post-lanzamiento)
# ─────────────────────────────────────────────
df["estimated_owners_num"] = df["estimated_owners"].apply(parse_owners)

for c in ["metacritic_score", "peak_ccu", "positive", "negative", "recommendations"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["total_reviews"] = (
    df.get("positive", pd.Series(0, index=df.index)).fillna(0)
  + df.get("negative", pd.Series(0, index=df.index)).fillna(0)
)
df["pct_positive"] = np.where(
    df["total_reviews"] > 0,
    df["positive"] / df["total_reviews"],
    np.nan
)

# Filtro de ruido temprano
df["review_tier"] = pd.cut(
    df["total_reviews"],
    bins=[-1, 100, 5000, np.inf],
    labels=["ruido", "moderado", "exito"]
)

# ─────────────────────────────────────────────
# 3. IDENTIFICAR JUEGOS INDIE
# ─────────────────────────────────────────────
genre_col = "genres" if "genres" in df.columns else "genre"
tag_col   = "tags"   if "tags"   in df.columns else None

df["genres_list"] = df[genre_col].apply(parse_list_col) if genre_col in df.columns \
                    else pd.Series([[] for _ in range(len(df))], index=df.index)
df["tags_list"]   = df[tag_col].apply(parse_list_col)   if tag_col in df.columns \
                    else pd.Series([[] for _ in range(len(df))], index=df.index)

def is_indie(row):
    genres = [g.lower() for g in row["genres_list"]]
    tags   = [t.lower() for t in row["tags_list"]]
    return int("indie" in genres or "indie" in tags)

df["is_indie"] = df.apply(is_indie, axis=1)

n_indie = df["is_indie"].sum()
print(f"\n  Juegos con tag/género Indie: {n_indie} ({n_indie/len(df)*100:.1f}% del total)")

# ─────────────────────────────────────────────
# 4. SUCCESS SCORE — normalización sobre indie
# ─────────────────────────────────────────────
score_cols = {
    "estimated_owners_num": 0.40,
    "peak_ccu":             0.25,
    "total_reviews":        0.20,
    "pct_positive":         0.10,
    "metacritic_score":     0.05,
}

# Filtra filas con al menos 3 de 5 targets presentes
df["valid_targets"] = df[list(score_cols.keys())].notna().sum(axis=1)
df = df[df["valid_targets"] >= 3].copy()
print(f"  Filas con targets suficientes: {len(df)}")

# Recalcula la máscara indie tras el filtro
indie_mask = df["is_indie"] == 1
df_indie   = df[indie_mask]
print(f"  Indies con targets suficientes: {indie_mask.sum()}")

# Normaliza usando percentiles del subconjunto indie como referencia
# → el p99 de un indie es el techo; todo se mide contra ese baremo
normed = {}
indie_norm_params = {}  # guardamos para reproducibilidad

for col in score_cols:
    s          = df[col].copy()
    indie_vals = df_indie[col].dropna()

    cap   = indie_vals.quantile(0.99) if len(indie_vals) > 0 else s.quantile(0.99)
    floor = indie_vals.quantile(0.01) if len(indie_vals) > 0 else s.quantile(0.01)

    indie_norm_params[col] = {"floor": float(floor), "cap": float(cap)}

    s = s.clip(lower=floor, upper=cap).fillna(floor)
    normed[col + "_norm"] = (s - floor) / (cap - floor + 1e-9)

normed_df = pd.DataFrame(normed, index=df.index)
df        = pd.concat([df, normed_df], axis=1)

df["success_score"] = sum(
    df[col + "_norm"] * w for col, w in score_cols.items()
) * 100  # escala 0–100

# ─────────────────────────────────────────────
# 5. UMBRALES DE CLASE — por percentil indie
#    Fracaso   : < p50  (mitad inferior de indies)
#    Moderado  : p50–p80
#    Alto éxito: > p80  (top 20% de indies)
#
#    IMPORTANTE: percentiles calculados solo sobre indies con actividad
#    real (owners > 0 y reviews > 100) para evitar que la masa de juegos
#    muertos comprima los umbrales hacia 0.
# ─────────────────────────────────────────────
indie_active_mask = (
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
)
indie_active_scores = df.loc[indie_active_mask, "success_score"]
print(f"\n  Indies con actividad real (owners>0, reviews>100): {indie_active_mask.sum()}")

p50 = indie_active_scores.quantile(0.50)
p80 = indie_active_scores.quantile(0.80)

print(f"\n  Umbrales de clase (percentiles indie):")
print(f"    Fracaso    → score < {p50:.1f}  (p50 indie)")
print(f"    Moderado   → {p50:.1f} – {p80:.1f}  (p50–p80 indie)")
print(f"    Alto éxito → score > {p80:.1f}  (p80 indie)")

def score_to_class(s):
    if s < p50:  return 0   # Fracaso
    elif s < p80: return 1  # Moderado
    else:         return 2  # Alto éxito

df["success_class"] = df["success_score"].apply(score_to_class)

labels = {0: "Fracaso", 1: "Moderado", 2: "Alto éxito"}

print("\nDistribución de clases (todos los juegos):")
for k, v in df["success_class"].value_counts().sort_index().items():
    print(f"  {labels[k]:12s}: {v:6d}  ({v/len(df)*100:.1f}%)")

print("\nDistribución de clases (solo indie):")
vc_i = df.loc[indie_mask, "success_class"].value_counts().sort_index()
n_i  = indie_mask.sum()
for k, v in vc_i.items():
    print(f"  {labels[k]:12s}: {v:6d}  ({v/n_i*100:.1f}%)")

# ─────────────────────────────────────────────
# 6. FEATURES PRE-LANZAMIENTO
# ─────────────────────────────────────────────

# --- Precio ---
df["price"]   = pd.to_numeric(df.get("price", np.nan), errors="coerce").fillna(0)
df["is_free"] = (df["price"] == 0).astype(int)

# --- Edad requerida ---
df["required_age"] = pd.to_numeric(df.get("required_age", 0), errors="coerce").fillna(0)

# --- Plataformas ---
for plat in ["windows", "mac", "linux"]:
    if plat in df.columns:
        df[f"plat_{plat}"] = df[plat].map(
            {True: 1, False: 0, "True": 1, "False": 0}
        ).fillna(0).astype(int)
    else:
        df[f"plat_{plat}"] = 0

# --- Multiplayer ---
def has_multiplayer(tags):
    if pd.isna(tags): return 0
    return int("multiplayer" in str(tags).lower() or "co-op" in str(tags).lower())

df["is_multiplayer"] = df.get("tags", pd.Series(dtype=str)).apply(has_multiplayer)

# --- Número de idiomas ---
def count_languages(s):
    if pd.isna(s): return 0
    try:    return len(ast.literal_eval(s))
    except: return len(str(s).split(","))

df["n_languages"] = df.get(
    "supported_languages", pd.Series(dtype=str)
).apply(count_languages)

# --- Longitud de descripción ---
df["desc_length"] = df.get(
    "short_description", pd.Series(dtype=str)
).fillna("").str.len()

# --- Géneros (one-hot, top 15 del subconjunto indie) ---
indie_genre_counts = pd.Series(
    [g for lst in df.loc[indie_mask, "genres_list"] for g in lst]
).value_counts()
TOP_GENRES = indie_genre_counts.head(15).index.tolist()
print(f"\n  Top géneros indie: {TOP_GENRES}")

for g in TOP_GENRES:
    safe = re.sub(r"[^a-z0-9]", "_", g.lower())
    df[f"genre_{safe}"] = df["genres_list"].apply(lambda lst: int(g in lst))

# --- Tags (top 30 del subconjunto indie) ---
indie_tag_counts = pd.Series(
    [t for lst in df.loc[indie_mask, "tags_list"] for t in lst]
).value_counts()
TOP_TAGS = indie_tag_counts.head(30).index.tolist()

for t in TOP_TAGS:
    safe = re.sub(r"[^a-z0-9]", "_", t.lower())
    df[f"tag_{safe}"] = df["tags_list"].apply(lambda lst: int(t in lst))

# --- Publisher: media de éxito sobre juegos indie (mín. 3 juegos) ---
pub_col = "publishers" if "publishers" in df.columns else "publisher"
if pub_col in df.columns:
    df["publisher_clean"] = df[pub_col].apply(
        lambda s: parse_list_col(s)[0] if parse_list_col(s) else "Unknown"
    )
else:
    df["publisher_clean"] = "Unknown"

pub_stats = (
    df[indie_mask]
    .groupby("publisher_clean")["success_score"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "pub_success_avg", "count": "pub_game_count"})
)
# Solo publishers con al menos 3 juegos indie son fiables
pub_avg_map      = pub_stats[pub_stats["pub_game_count"] >= 3]["pub_success_avg"]
global_indie_avg = df.loc[indie_mask, "success_score"].mean()

df["publisher_success_avg"] = (
    df["publisher_clean"]
    .map(pub_avg_map)
    .fillna(global_indie_avg)   # fallback: media global indie
)

print(f"\n  Publishers con ≥3 juegos indie: {len(pub_avg_map)}")
print(f"  Media global indie (fallback):  {global_indie_avg:.1f}")

# --- Release date ---
if "release_date" in df.columns:
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]        = df["release_date_parsed"].dt.year.fillna(0).astype(int)
    df["release_quarter"]     = df["release_date_parsed"].dt.quarter.fillna(0).astype(int)
else:
    df["release_year"]    = 0
    df["release_quarter"] = 0

# ─────────────────────────────────────────────
# 7. GUARDAR DATOS + METADATOS
# ─────────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# Umbrales y parámetros de normalización para usar en predicción
thresholds = {
    "p50":              float(p50),
    "p80":              float(p80),
    "global_indie_avg": float(global_indie_avg),
    "norm_params":      indie_norm_params,
    "top_genres":       TOP_GENRES,
    "top_tags":         TOP_TAGS,
    "score_weights":    score_cols,
}
with open("models/thresholds.json", "w") as f:
    json.dump(thresholds, f, indent=2)

df.to_parquet(OUT_PATH, index=False)
print(f"\n✅ Guardado en {OUT_PATH}  ({len(df)} filas, {df.shape[1]} columnas)")
print(f"✅ Umbrales y parámetros guardados en models/thresholds.json")

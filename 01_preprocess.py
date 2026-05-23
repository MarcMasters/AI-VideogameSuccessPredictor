"""
01_preprocess_v2.py
===================
Igual que 01_preprocess.py pero con features adicionales de ingeniería.
Genera: data/processed_v2.parquet
"""

import pandas as pd
import numpy as np
import ast, re, os, json

RAW_PATH = "data/games.csv"
OUT_PATH = "data/processed_v2.parquet"

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
# 2. COLUMNAS TARGET
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
print(f"\n  Juegos indie: {n_indie} ({n_indie/len(df)*100:.1f}%)")

# ─────────────────────────────────────────────
# 4. SUCCESS SCORE
# ─────────────────────────────────────────────
score_cols = {
    "estimated_owners_num": 0.40,
    "peak_ccu":             0.25,
    "total_reviews":        0.20,
    "pct_positive":         0.10,
    "metacritic_score":     0.05,
}

df["valid_targets"] = df[list(score_cols.keys())].notna().sum(axis=1)
df = df[df["valid_targets"] >= 3].copy()

indie_mask = df["is_indie"] == 1
df_indie   = df[indie_mask]

normed = {}
indie_norm_params = {}

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
) * 100

# ─────────────────────────────────────────────
# 5. UMBRALES DE CLASE
# ─────────────────────────────────────────────
indie_active_mask = (
    (df["is_indie"] == 1) &
    (df["estimated_owners_num"] > 0) &
    (df["total_reviews"] > 100)
)
indie_active_scores = df.loc[indie_active_mask, "success_score"]

p50 = indie_active_scores.quantile(0.50)
p80 = indie_active_scores.quantile(0.80)

print(f"\n  Umbrales: Fracaso < {p50:.1f} | Moderado {p50:.1f}–{p80:.1f} | Alto éxito > {p80:.1f}")

def score_to_class(s):
    if s < p50:   return 0
    elif s < p80: return 1
    else:         return 2

df["success_class"] = df["success_score"].apply(score_to_class)

# ─────────────────────────────────────────────
# 6. FEATURES PRE-LANZAMIENTO BASE
# ─────────────────────────────────────────────
df["price"]        = pd.to_numeric(df.get("price", np.nan), errors="coerce").fillna(0)
df["is_free"]      = (df["price"] == 0).astype(int)
df["required_age"] = pd.to_numeric(df.get("required_age", 0), errors="coerce").fillna(0)

for plat in ["windows", "mac", "linux"]:
    if plat in df.columns:
        df[f"plat_{plat}"] = df[plat].map(
            {True: 1, False: 0, "True": 1, "False": 0}
        ).fillna(0).astype(int)
    else:
        df[f"plat_{plat}"] = 0

def has_multiplayer(tags):
    if pd.isna(tags): return 0
    return int("multiplayer" in str(tags).lower() or "co-op" in str(tags).lower())

df["is_multiplayer"] = df.get("tags", pd.Series(dtype=str)).apply(has_multiplayer)

def count_languages(s):
    if pd.isna(s): return 0
    try:    return len(ast.literal_eval(s))
    except: return len(str(s).split(","))

df["n_languages"] = df.get("supported_languages", pd.Series(dtype=str)).apply(count_languages)
df["desc_length"] = df.get("short_description", pd.Series(dtype=str)).fillna("").str.len()

# Géneros one-hot
indie_genre_counts = pd.Series(
    [g for lst in df.loc[indie_mask, "genres_list"] for g in lst]
).value_counts()
TOP_GENRES = indie_genre_counts.head(15).index.tolist()

for g in TOP_GENRES:
    safe = re.sub(r"[^a-z0-9]", "_", g.lower())
    df[f"genre_{safe}"] = df["genres_list"].apply(lambda lst: int(g in lst))

# Tags one-hot
indie_tag_counts = pd.Series(
    [t for lst in df.loc[indie_mask, "tags_list"] for t in lst]
).value_counts()
TOP_TAGS = indie_tag_counts.head(30).index.tolist()

for t in TOP_TAGS:
    safe = re.sub(r"[^a-z0-9]", "_", t.lower())
    df[f"tag_{safe}"] = df["tags_list"].apply(lambda lst: int(t in lst))

# Publisher success avg
pub_col = "publishers" if "publishers" in df.columns else "publisher"
if pub_col in df.columns:
    df["publisher_clean"] = df[pub_col].apply(
        lambda s: parse_list_col(s)[0] if parse_list_col(s) else "Unknown"
    )
else:
    df["publisher_clean"] = "Unknown"

pub_stats    = df[indie_mask].groupby("publisher_clean")["success_score"].agg(["mean", "count"])
pub_avg_map  = pub_stats[pub_stats["count"] >= 3]["mean"]
global_indie_avg = df.loc[indie_mask, "success_score"].mean()

df["publisher_success_avg"] = df["publisher_clean"].map(pub_avg_map).fillna(global_indie_avg)

# Release date
if "release_date" in df.columns:
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]        = df["release_date_parsed"].dt.year.fillna(0).astype(int)
    df["release_quarter"]     = df["release_date_parsed"].dt.quarter.fillna(0).astype(int)
else:
    df["release_year"]    = 0
    df["release_quarter"] = 0

# ─────────────────────────────────────────────
# 7. FEATURE ENGINEERING NUEVO ★
# ─────────────────────────────────────────────
print("\n  Aplicando feature engineering…")

# Ratio precio/idiomas
df["price_per_lang"] = df["price"] / (df["n_languages"] + 1)

# Densidad de tags
df["n_tags"] = df["tags_list"].apply(len)

# Número de plataformas
df["n_platforms"] = df["plat_windows"] + df["plat_mac"] + df["plat_linux"]

# Interacción multiplayer × free
df["mp_x_free"] = df["is_multiplayer"] * df["is_free"]

# Interacción Action × Singleplayer (comprueba que las columnas existen)
if "genre_action" in df.columns and "tag_singleplayer" in df.columns:
    df["action_singleplayer"] = df["genre_action"] * df["tag_singleplayer"]
else:
    df["action_singleplayer"] = 0

# Ratio reviews positivas ponderado por volumen (proxy de confianza)
# Solo como feature de contexto — NO es post-lanzamiento pura porque
# viene del score, pero la longitud de descripción sí es pre-lanzamiento
df["desc_length_log"] = np.log1p(df["desc_length"])

# Publisher tier: segmenta en cuartiles para no depender solo de la media
publisher_quantiles = df.loc[indie_mask, "publisher_success_avg"].quantile([0.25, 0.5, 0.75])
def publisher_tier(val):
    if val <= publisher_quantiles[0.25]: return 0
    elif val <= publisher_quantiles[0.50]: return 1
    elif val <= publisher_quantiles[0.75]: return 2
    else: return 3

df["publisher_tier"] = df["publisher_success_avg"].apply(publisher_tier)

print("  Features nuevas: price_per_lang, n_tags, n_platforms, mp_x_free, "
      "action_singleplayer, desc_length_log, publisher_tier")

# ─────────────────────────────────────────────
# 8. GUARDAR
# ─────────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

thresholds = {
    "p50":              float(p50),
    "p80":              float(p80),
    "global_indie_avg": float(global_indie_avg),
    "norm_params":      indie_norm_params,
    "top_genres":       TOP_GENRES,
    "top_tags":         TOP_TAGS,
    "score_weights":    score_cols,
}
with open("models/thresholds_v2.json", "w") as f:
    json.dump(thresholds, f, indent=2)

df.to_parquet(OUT_PATH, index=False)
print(f"\n✅ Guardado en {OUT_PATH}  ({len(df)} filas, {df.shape[1]} columnas)")
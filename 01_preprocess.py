"""
01_preprocess.py
================
Limpieza, ingeniería de features y cálculo del success_score.
Genera: data/processed.parquet
"""

import pandas as pd
import numpy as np
import ast, re, os
from sklearn.preprocessing import MinMaxScaler

RAW_PATH  = "data/games.csv"
OUT_PATH  = "data/processed.parquet"

# ─────────────────────────────────────────────
# 1. CARGA
# ─────────────────────────────────────────────
print("Cargando dataset…")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"  Shape inicial: {df.shape}")

# ─────────────────────────────────────────────
# 2. COLUMNAS TARGET (post-lanzamiento)
# ─────────────────────────────────────────────
TARGET_COLS = [
    "metacritic_score", "peak_ccu",
    "positive", "negative",
    "estimated_owners",
    "recommendations",          # proxy de total reviews recientes
]

# Estimated owners viene como string "1000 - 5000", tomamos el punto medio
def parse_owners(s):
    if pd.isna(s):
        return np.nan
    m = re.findall(r"\d+", str(s).replace(",", ""))
    if len(m) >= 2:
        return (int(m[0]) + int(m[1])) / 2
    elif len(m) == 1:
        return int(m[0])
    return np.nan

df["estimated_owners_num"] = df["estimated_owners"].apply(parse_owners)

# Asegura que columnas numéricas sean numéricas
for c in ["metacritic_score", "peak_ccu", "positive", "negative", "recommendations"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Total reviews
df["total_reviews"] = df.get("positive", 0).fillna(0) + df.get("negative", 0).fillna(0)
df["pct_positive"]  = np.where(df["total_reviews"] > 0,
                                df["positive"] / df["total_reviews"], np.nan)

# ─────────────────────────────────────────────
# 3. SUCCESS SCORE
# ─────────────────────────────────────────────
scaler = MinMaxScaler()

score_cols = {
    "estimated_owners_num": 0.40,
    "peak_ccu":             0.25,
    "total_reviews":        0.20,
    "pct_positive":         0.10,
    "metacritic_score":     0.05,
}

# Filtra filas con al menos 3 de 5 valores presentes
df["valid_targets"] = df[list(score_cols.keys())].notna().sum(axis=1)
df = df[df["valid_targets"] >= 3].copy()
print(f"  Filas con targets suficientes: {len(df)}")

# Normaliza cada columna al rango [0,1] (clip de outliers al p99)
normed = {}
for col in score_cols:
    s = df[col].copy()
    cap = s.quantile(0.99)
    s = s.clip(upper=cap).fillna(0)
    s_min, s_max = s.min(), s.max()
    normed[col + "_norm"] = (s - s_min) / (s_max - s_min + 1e-9)

normed_df = pd.DataFrame(normed, index=df.index)
df = pd.concat([df, normed_df], axis=1)

df["success_score"] = sum(
    df[col + "_norm"] * w for col, w in score_cols.items()
) * 100  # escala 0–100

# Clases
def score_to_class(s):
    if s < 40:   return 0   # Fracaso
    elif s < 70: return 1   # Moderado
    else:        return 2   # Alto éxito

df["success_class"] = df["success_score"].apply(score_to_class)

# Filtro por reviews (ruido)
df["review_tier"] = pd.cut(df["total_reviews"],
                            bins=[-1, 100, 5000, np.inf],
                            labels=["ruido", "moderado", "exito"])

print("\nDistribución de clases:")
print(df["success_class"].value_counts().sort_index()
        .rename({0:"Fracaso", 1:"Moderado", 2:"Alto éxito"}))

# ─────────────────────────────────────────────
# 4. FEATURES PRE-LANZAMIENTO
# ─────────────────────────────────────────────

# --- Precio ---
df["price"] = pd.to_numeric(df.get("price", np.nan), errors="coerce").fillna(0)
df["is_free"] = (df["price"] == 0).astype(int)

# --- Edad requerida ---
df["required_age"] = pd.to_numeric(df.get("required_age", 0), errors="coerce").fillna(0)

# --- Plataformas ---
for plat in ["windows", "mac", "linux"]:
    col = plat if plat in df.columns else None
    if col:
        df[f"plat_{plat}"] = df[col].map({True:1, False:0, "True":1, "False":0}).fillna(0).astype(int)
    else:
        df[f"plat_{plat}"] = 0

# --- Multiplayer ---
def has_multiplayer(tags):
    if pd.isna(tags): return 0
    return int("multiplayer" in str(tags).lower() or "co-op" in str(tags).lower())

df["is_multiplayer"] = df.get("tags", pd.Series(dtype=str)).apply(has_multiplayer)

# --- Número de idiomas soportados ---
def count_languages(s):
    if pd.isna(s): return 0
    try:
        lst = ast.literal_eval(s)
        return len(lst)
    except:
        return len(str(s).split(","))

df["n_languages"] = df.get("supported_languages", pd.Series(dtype=str)).apply(count_languages)

# --- Longitud de descripción ---
df["desc_length"] = df.get("short_description", pd.Series(dtype=str)).fillna("").str.len()

# --- Géneros (one-hot, top 15) ---
def parse_list_col(s):
    if pd.isna(s): return []
    try:
        return [x.strip() for x in ast.literal_eval(s)]
    except:
        return [x.strip() for x in str(s).split(",")]

genre_col = "genres" if "genres" in df.columns else "genre"
if genre_col in df.columns:
    df["genres_list"] = df[genre_col].apply(parse_list_col)
    all_genres = pd.Series([g for lst in df["genres_list"] for g in lst]).value_counts()
    TOP_GENRES = all_genres.head(15).index.tolist()
    for g in TOP_GENRES:
        safe = re.sub(r"[^a-z0-9]", "_", g.lower())
        df[f"genre_{safe}"] = df["genres_list"].apply(lambda lst: int(g in lst))
else:
    TOP_GENRES = []

# --- Tags (top 30) ---
if "tags" in df.columns:
    df["tags_list"] = df["tags"].apply(parse_list_col)
    all_tags = pd.Series([t for lst in df["tags_list"] for t in lst]).value_counts()
    TOP_TAGS = all_tags.head(30).index.tolist()
    for t in TOP_TAGS:
        safe = re.sub(r"[^a-z0-9]", "_", t.lower())
        df[f"tag_{safe}"] = df["tags_list"].apply(lambda lst: int(t in lst))
else:
    TOP_TAGS = []

# --- Publisher (top 160) ---
pub_col = "publishers" if "publishers" in df.columns else "publisher"
if pub_col in df.columns:
    df["publisher_clean"] = df[pub_col].apply(lambda s: parse_list_col(s)[0] if parse_list_col(s) else "Unknown")
    top_pubs = df["publisher_clean"].value_counts().head(160).index.tolist()
    df["publisher_top"] = df["publisher_clean"].apply(lambda p: p if p in top_pubs else "Other")
else:
    df["publisher_top"] = "Unknown"

# --- Release zone (por año, luego cuarto) ---
if "release_date" in df.columns:
    df["release_date_parsed"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]    = df["release_date_parsed"].dt.year.fillna(0).astype(int)
    df["release_quarter"] = df["release_date_parsed"].dt.quarter.fillna(0).astype(int)
else:
    df["release_year"] = 0
    df["release_quarter"] = 0

# ─────────────────────────────────────────────
# 5. GUARDAR
# ─────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df.to_parquet(OUT_PATH, index=False)
print(f"\n✅ Guardado en {OUT_PATH}  ({len(df)} filas, {df.shape[1]} columnas)")

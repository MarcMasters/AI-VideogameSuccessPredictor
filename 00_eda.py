"""
00_eda.py
=========
Análisis exploratorio rápido del dataset de Steam.
Genera visualizaciones en outputs/eda_*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import re, ast, os

RAW_PATH = "data/games.csv"
os.makedirs("outputs", exist_ok=True)

print("Cargando dataset…")
df = pd.read_csv(RAW_PATH, low_memory=False)
print(f"Shape: {df.shape}")
print(f"\nColumnas:\n{df.columns.tolist()}")
print(f"\nNulos por columna (top 20):\n{df.isnull().sum().sort_values(ascending=False).head(20)}")

# ── Precio ──────────────────────────────────────────────
df["price"] = pd.to_numeric(df.get("price", np.nan), errors="coerce")

# ── Estimated owners ────────────────────────────────────
def parse_owners(s):
    if pd.isna(s): return np.nan
    m = re.findall(r"\d+", str(s).replace(",", ""))
    if len(m) >= 2: return (int(m[0]) + int(m[1])) / 2
    elif len(m) == 1: return int(m[0])
    return np.nan

df["owners_num"] = df["estimated_owners"].apply(parse_owners)

# ── Reviews ─────────────────────────────────────────────
for c in ["positive", "negative", "metacritic_score", "peak_ccu"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["total_reviews"] = df.get("positive", 0).fillna(0) + df.get("negative", 0).fillna(0)
df["pct_positive"]  = np.where(df["total_reviews"] > 0,
                                df["positive"] / df["total_reviews"], np.nan)

# ─────────────────────────────────────────────────────────
# FIGURA 1: Distribuciones de targets
# ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Distribución de variables TARGET", fontsize=14, y=1.01)

plot_data = [
    ("owners_num",       "Estimated Owners",       True),
    ("peak_ccu",         "Peak CCU",                True),
    ("total_reviews",    "Total Reviews",           True),
    ("pct_positive",     "% Positive Reviews",      False),
    ("metacritic_score", "Metacritic Score",        False),
    ("price",            "Precio (USD)",            True),
]

for ax, (col, title, log) in zip(axes.flat, plot_data):
    s = df[col].dropna()
    if log: # logarithm
        s = s[s > 0]
        ax.hist(np.log10(s + 1), bins=50, color="#3498db", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(f"log10({col})")
    else:
        ax.hist(s, bins=50, color="#2ecc71", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(col)
    ax.set_title(title)
    ax.set_ylabel("Frecuencia")

plt.tight_layout()
plt.savefig("outputs/eda_targets.png", dpi=150, bbox_inches="tight")
plt.close()
print("✅ outputs/eda_targets.png")

# ─────────────────────────────────────────────────────────
# FIGURA 2: Top géneros
# ─────────────────────────────────────────────────────────
genre_col = "genres" if "genres" in df.columns else "genre"
if genre_col in df.columns:
    def parse_list(s):
        if pd.isna(s): return []
        try: return ast.literal_eval(s)
        except: return [x.strip() for x in str(s).split(",")]

    genre_counts = pd.Series(
        [g for lst in df[genre_col].apply(parse_list) for g in lst]
    ).value_counts().head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    genre_counts.sort_values().plot(kind="barh", ax=ax, color="#e74c3c")
    ax.set_title("Top 15 Géneros en Steam")
    ax.set_xlabel("Número de juegos")
    plt.tight_layout()
    plt.savefig("outputs/eda_genres.png", dpi=150)
    plt.close()
    print("EXPORTED: outputs/eda_genres.png")

# ─────────────────────────────────────────────────────────
# FIGURA 3: Correlación precio vs owners
# ─────────────────────────────────────────────────────────
sub = df[["price", "owners_num", "pct_positive"]].dropna()
sub = sub[(sub["price"] < 100) & (sub["owners_num"] > 0)]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(sub["price"], np.log10(sub["owners_num"]+1),
                alpha=0.15, s=5, color="#9b59b6")
axes[0].set_xlabel("Precio (USD)")
axes[0].set_ylabel("log10(Estimated Owners)")
axes[0].set_title("Precio vs Propietarios estimados")

axes[1].scatter(sub["price"], sub["pct_positive"],
                alpha=0.15, s=5, color="#e67e22")
axes[1].set_xlabel("Precio (USD)")
axes[1].set_ylabel("% Reviews positivas")
axes[1].set_title("Precio vs Valoración")

plt.tight_layout()
plt.savefig("outputs/eda_price_scatter.png", dpi=150)
plt.close()
print("✅ outputs/eda_price_scatter.png")

print("\nEDA completo.")

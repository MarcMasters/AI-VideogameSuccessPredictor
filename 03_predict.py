"""
03_predict.py
=============
CLI interactivo para predecir el éxito de un nuevo juego
usando el modelo entrenado.

Uso:
    python 03_predict.py
    python 03_predict.py --json examples/my_game.json
"""

import json, argparse, joblib
import pandas as pd
import numpy as np
# from pathlib import Path

MODEL_PATH = "models/rf_pipeline.joblib"
META_PATH  = "models/metadata.json"

CLASS_LABELS = {0: "🔴 FRACASO", 1: "🟡 MODERADO", 2: "🟢 ALTO ÉXITO"}
CLASS_RANGES = {0: "0–40", 1: "40–70", 2: "70–100"}

TOP_GENRES = [
    "Action", "Indie", "Adventure", "RPG", "Strategy",
    "Simulation", "Casual", "Sports", "Racing", "Free to Play",
    "Massively Multiplayer", "Early Access", "Puzzle", "Horror", "Platformer",
]
TOP_TAGS_SAMPLE = [
    "Singleplayer", "Multiplayer", "Action", "Indie", "Adventure",
    "RPG", "Strategy", "Simulation", "Casual", "Sports",
    "Co-op", "Open World", "Sandbox", "Horror", "Puzzle",
    "Racing", "Shooter", "Platformer", "Survival", "Roguelike",
    "Tower Defense", "Card Game", "Visual Novel", "Anime", "Sci-fi",
    "Fantasy", "Stealth", "First-Person", "Third Person", "Story Rich",
]


def load_model():
    pipeline = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    return pipeline, meta


def build_feature_dict(game: dict, meta: dict) -> dict:
    """Convierte los datos del juego en el vector de features esperado."""
    row = {}

    # Numéricas base
    row["price"]          = float(game.get("price", 0))
    row["is_free"]        = int(row["price"] == 0)
    row["required_age"]   = int(game.get("required_age", 0))
    row["plat_windows"]   = int(game.get("windows", True))
    row["plat_mac"]       = int(game.get("mac", False))
    row["plat_linux"]     = int(game.get("linux", False))
    row["is_multiplayer"] = int(game.get("multiplayer", False))
    row["n_languages"]    = int(game.get("n_languages", 1))
    row["desc_length"]    = len(game.get("description", ""))
    row["release_year"]   = int(game.get("release_year", 2024))
    row["release_quarter"]= int(game.get("release_quarter", 1))

    # Géneros
    genres_input = [g.strip() for g in game.get("genres", [])]
    for g in TOP_GENRES:
        import re
        safe = re.sub(r"[^a-z0-9]", "_", g.lower())
        row[f"genre_{safe}"] = int(g in genres_input)

    # Tags
    tags_input = [t.strip() for t in game.get("tags", [])]
    for t in TOP_TAGS_SAMPLE:
        import re
        safe = re.sub(r"[^a-z0-9]", "_", t.lower())
        row[f"tag_{safe}"] = int(t in tags_input)

    # Publisher
    row["publisher_top"] = game.get("publisher", "Other")

    # Rellenar features que puedan faltar del modelo
    for feat in meta["numeric_features"]:
        if feat not in row:
            row[feat] = 0

    return row


def predict_game(game: dict, pipeline, meta: dict):
    row     = build_feature_dict(game, meta)
    X       = pd.DataFrame([row])

    proba   = pipeline.predict_proba(X)[0]
    pred    = int(pipeline.predict(X)[0])

    return pred, proba


def print_result(game_name: str, pred: int, proba: np.ndarray):
    print("\n" + "═"*50)
    print(f"  Juego: {game_name}")
    print("═"*50)
    print(f"  Predicción:  {CLASS_LABELS[pred]}  (rango {CLASS_RANGES[pred]})")
    print()
    print("  Probabilidades:")
    for i, label in CLASS_LABELS.items():
        bar = "█" * int(proba[i] * 30)
        print(f"    {label:20s}  {bar:30s}  {proba[i]*100:.1f}%")
    print("═"*50 + "\n")


def interactive_mode(pipeline, meta):
    print("\n🎮  PREDICTOR DE ÉXITO DE VIDEOJUEGOS  🎮")
    print("Introduce los datos de tu juego (pre-lanzamiento)\n")

    game = {}
    game["name"]          = input("Nombre del juego: ").strip()
    game["price"]         = float(input("Precio (0 si es free-to-play): ") or 0)
    game["required_age"]  = int(input("Edad mínima requerida (0, 7, 12, 16, 18): ") or 0)
    game["n_languages"]   = int(input("Número de idiomas soportados: ") or 1)
    game["multiplayer"]   = input("¿Tiene multijugador/co-op? (s/n): ").strip().lower() == "s"

    print(f"\nPlataformas de lanzamiento:")
    game["windows"] = input("  Windows (s/n): ").strip().lower() != "n"
    game["mac"]     = input("  Mac     (s/n): ").strip().lower() == "s"
    game["linux"]   = input("  Linux   (s/n): ").strip().lower() == "s"

    print(f"\nGéneros disponibles: {', '.join(TOP_GENRES)}")
    genres_raw = input("Géneros de tu juego (separados por coma): ")
    game["genres"] = [g.strip() for g in genres_raw.split(",") if g.strip()]

    print(f"\nEjemplos de tags: {', '.join(TOP_TAGS_SAMPLE[:10])}…")
    tags_raw = input("Tags de tu juego (separados por coma): ")
    game["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]

    game["publisher"] = input("\nPublisher (o 'Other' si es indie sin publisher conocido): ").strip() or "Other"
    game["description"] = input("Descripción corta del juego: ").strip()

    game["release_year"]    = int(input("Año previsto de lanzamiento (ej. 2025): ") or 2025)
    game["release_quarter"] = int(input("Trimestre de lanzamiento (1-4): ") or 2)

    pred, proba = predict_game(game, pipeline, meta)
    print_result(game["name"], pred, proba)


def json_mode(json_path: str, pipeline, meta):
    with open(json_path) as f:
        game = json.load(f)
    pred, proba = predict_game(game, pipeline, meta)
    print_result(game.get("name", json_path), pred, proba)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor de éxito de videojuegos")
    parser.add_argument("--json", help="Ruta a un JSON con los datos del juego")
    args = parser.parse_args()

    pipeline, meta = load_model()

    if args.json:
        json_mode(args.json, pipeline, meta)
    else:
        interactive_mode(pipeline, meta)

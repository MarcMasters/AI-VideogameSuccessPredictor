"""
03_predict.py
=============
CLI interactivo y lector de JSON adaptable para predecir el éxito
de un nuevo videojuego. Carga y mapea dinámicamente según la 
arquitectura elegida (LGBM Binario/Multiclase).

Uso:
    python 03_predict.py
    python 03_predict.py --json examples/my_game.json
"""

import json
import argparse
import joblib
import os
import pandas as pd
import numpy as np

# Diccionario oficial de pesos idiomáticos del proyecto
LANGUAGE_WEIGHTS = {
    "English": 1.00, "Simplified Chinese": 0.85, "Russian": 0.55, 
    "German": 0.45, "French": 0.40, "Spanish - Spain": 0.35, "Portuguese - Brazil": 0.33,
    "Japanese": 0.30, "Korean": 0.28, "Polish": 0.22, "Turkish": 0.20, 
    "Traditional Chinese": 0.18, "Italian": 0.17, "Dutch": 0.15, 
    "Spanish - Latin America": 0.15, "Czech": 0.13, "Hungarian": 0.12, 
    "Romanian": 0.11, "Swedish": 0.10, "Norwegian": 0.09, "Danish": 0.09, 
    "Finnish": 0.08, "Ukrainian": 0.08, "Thai": 0.07, "Arabic": 0.07, "Portuguese - Portugal": 0.06
}
LANGUAGE_WEIGHTS_DEFAULT = 0.04

def ask_prediction_mode():
    """Pregunta al usuario qué infraestructura de modelo desea levantar."""
    print("\n🎛️  SELECCIÓN DE ARQUITECTURA DE IA")
    print("  [1] Modo Binario      (Éxito vs Fracaso)")
    print("  [2] Modo Multiclase   (Fracaso, Moderado, Alto Éxito)")
    
    while True:
        opc = input("\nSelecciona el modo (1 o 2): ").strip()
        if opc == "1":
            return "models/processed_binary/lgbm_pipeline.joblib", "models/processed_binary/lgbm_metadata.json", "models/thresholds_binary.json"
        elif opc == "2":
            return "models/processed_multiclass/lgbm_pipeline.joblib", "models/processed_multiclass/lgbm_metadata.json", "models/thresholds_multiclass.json"
        print("❌ Opción inválida. Introduce 1 o 2.")

def load_resources(p_path, m_path, t_path):
    """Carga los pipelines y archivos de metadatos comprobando su existencia."""
    if not os.path.exists(p_path) or not os.path.exists(m_path) or not os.path.exists(t_path):
        raise FileNotFoundError(
            f"❌ No se encuentran los archivos del modelo en la ruta especificada.\n"
            f"Asegúrate de haber ejecutado antes el entrenamiento correspondiente."
        )

    pipeline = joblib.load(p_path)
    with open(m_path) as f:
        meta_train = json.load(f)
    with open(t_path) as f:
        thresholds = json.load(f)
            
    return pipeline, meta_train, thresholds

def calculate_language_score(langs_list):
    """Calcula el score ponderado de mercado basado en vuestro diccionario."""
    if not langs_list: return 0
    return sum(LANGUAGE_WEIGHTS.get(l.strip(), LANGUAGE_WEIGHTS_DEFAULT) for l in langs_list if l.strip())

def build_feature_vector(game: dict, meta_train: dict, thresholds: dict) -> dict:
    """Construye el diccionario de variables mapeando dinámicamente los metadatos."""
    row = {}

    # 1. Variables Numéricas Directas
    row["price"] = float(game.get("price", 0))
    row["is_free"] = int(row["price"] == 0)
    row["required_age"] = int(game.get("required_age", 0))
    row["plat_windows"] = int(game.get("windows", True))
    row["plat_mac"] = int(game.get("mac", False))
    row["plat_linux"] = int(game.get("linux", False))
    row["is_multiplayer"] = int(game.get("multiplayer", False))
    row["release_quarter"] = int(game.get("release_quarter", 1))

    # 2. Score de Idiomas Ponderado (Tu algoritmo de pesos)
    langs_input = game.get("supported_languages", [])
    if isinstance(langs_input, str):
        langs_input = [l.strip() for l in langs_input.split(",") if l.strip()]
    row["languages_market_score"] = calculate_language_score(langs_input)

    # 3. Inputs Multimedia con Capado Estadístico Basado en su Threshold JSON
    row["n_screenshots"] = int(game.get("n_screenshots", 0))
    row["n_movies"] = int(game.get("n_movies", 0))
    row["n_achievements"] = int(game.get("n_achievements", 0))
    
    norm_p = thresholds.get("norm_params", {})
    for col in ["n_screenshots", "n_movies", "n_achievements"]:
        if col in norm_p:
            row[col] = min(row[col], norm_p[col]["cap"])

    # 4. Sinergias Avanzadas Pre-Lanzamiento
    row["n_platforms"] = row["plat_windows"] + row["plat_mac"] + row["plat_linux"]
    row["mp_x_free"] = row["is_multiplayer"] * row["is_free"]

    # 5. Mapeo Dinámico de Géneros One-Hot basados en vuestro metadata.json
    genres_input = [g.strip().lower() for g in game.get("genres", [])]
    if isinstance(genres_input, str): genres_input = [g.strip().lower() for g in genres_input.split(",")]
    
    model_genres = [f for f in meta_train["numeric_features"] if f.startswith("genre_")]
    for feat in model_genres:
        clean_name = feat.replace("genre_", "").replace("_", " ")
        row[feat] = int(any(clean_name in g for g in genres_input))

    # 6. Mapeo Dinámico de Tags One-Hot basados en vuestro metadata.json
    tags_input = [t.strip().lower() for t in game.get("tags", [])]
    if isinstance(tags_input, str): tags_input = [t.strip().lower() for t in tags_input.split(",")]
    
    model_tags = [f for f in meta_train["numeric_features"] if f.startswith("tag_")]
    for feat in model_tags:
        clean_name = feat.replace("tag_", "").replace("_", " ")
        row[feat] = int(any(clean_name in t for t in tags_input))

    # 7. Constante del Publisher (Auto-publicado o Desconocido)
    row["publisher_success_avg"] = thresholds.get("self_published_avg", 15.0)

    # 8. Compatibilidad y salvaguarda con vuestro JSON histórico de pruebas
    row["n_languages"] = len(langs_input)
    row["desc_length"] = len(game.get("description", ""))
    row["release_year"] = int(game.get("release_year", 2025))

    # Rellenar con ceros cualquier característica huérfana para evitar crashes de LightGBM
    for feat in meta_train["numeric_features"]:
        if feat not in row:
            row[feat] = 0

    return row

def print_result(game_name: str, pred: int, proba: np.ndarray, meta_train: dict):
    """Pinta los resultados leyendo las etiquetas mapeadas directamente en el metadata.json."""
    classes_dict = meta_train.get("classes", {"0": "Fracaso", "1": "Éxito"})
    icons = {"0": "🔴 FRACASO", "1": "🟡 MODERADO", "2": "🟢 ALTO ÉXITO"}
    
    print("\n" + "═"*60)
    print(f" 📊  INFORME PREDICTIVO INDIE: {game_name.upper()}")
    print("═"*60)
    print(f"  VERDICTO FINAL:  {icons.get(str(pred), '✨ CLASE ' + str(pred))} ({classes_dict.get(str(pred))})")
    print()
    print("  Desglose de probabilidad estadística:")
    
    for idx_str, label in classes_dict.items():
        idx = int(idx_str)
        val_proba = proba[idx] if idx < len(proba) else 0.0
        bar_len = int(val_proba * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {icons.get(idx_str, '📦'):15s} ({label[:10]:10s})  [{bar}]  {val_proba*100:.1f}%")
    print("═"*60 + "\n")

def interactive_mode(meta_train, thresholds):
    """Interfaz conversacional limpia por consola."""
    print("\n🎮  SIMULADOR WEB DE VIABILIDAD COMERCIAL INDIE  🎮")
    print("Introduce los parámetros de diseño de tu proyecto:\n")

    game = {}
    game["name"] = input("• Nombre del proyecto: ").strip() or "Proyecto Desconocido"
    game["price"] = float(input("• Precio de venta en € (0 si es Free-to-Play): ") or 0)
    game["required_age"] = int(input("• Restricción de edad (0, 7, 12, 16, 18): ") or 0)
    game["multiplayer"] = input("• ¿Es multijugador o cooperativo? (s/n): ").strip().lower() == "s"

    print(f"\n🌍  Localización:")
    langs_raw = input("   Idiomas soportados (separados por comas, ej: English, Spanish - Spain): ")
    game["supported_languages"] = [l.strip() for l in langs_raw.split(",") if l.strip()]

    print(f"\n📸  Multimedia en Tienda:")
    game["n_screenshots"] = int(input("   - Número de capturas de pantalla: ") or 0)
    game["n_movies"] = int(input("   - Número de trailers: ") or 0)
    game["n_achievements"] = int(input("   - Número de logros de Steam: ") or 0)

    print(f"\n💻  Sistemas Operativos:")
    game["windows"] = input("   - ¿Soporta Windows? (s/n, por defecto Sí): ").strip().lower() != "n"
    game["mac"] = input("   - ¿Soporta macOS? (s/n): ").strip().lower() == "s"
    game["linux"] = input("   - ¿Soporta Linux? (s/n): ").strip().lower() == "s"

    print(f"\n🎭  Géneros y Etiquetas:")
    genres_raw = input("   Géneros (separados por comas, ej: Indie, Action): ")
    game["genres"] = [g.strip() for g in genres_raw.split(",") if g.strip()]

    tags_raw = input("   Tags descriptivos (separados por comas, ej: Singleplayer, Pixel Art): ")
    game["tags"] = [t.strip() for t in tags_raw.split(",") if t.strip()]

    game["publisher"] = input("\n• Publisher (vacío si es auto-publicado): ").strip() or "Unknown"
    game["release_quarter"] = int(input("• Trimestre objetivo de lanzamiento (1-4): ") or 1)

    return game

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor de éxito de videojuegos")
    parser.add_argument("--json", help="Ruta a un archivo JSON con los datos del juego")
    args = parser.parse_args()

    # Selección directa de la carpeta del modelo a evaluar
    p_path, m_path, t_path = ask_prediction_mode()
    pipeline, meta_train, thresholds = load_resources(p_path, m_path, t_path)

    if args.json:
        with open(args.json) as f:
            game_data = json.load(f)
    else:
        game_data = interactive_mode(meta_train, thresholds)

    # Construcción estructurada del vector leyendo las columnas que dicte el metadata.json
    feature_row = build_feature_vector(game_data, meta_train, thresholds)
    X_input = pd.DataFrame([feature_row])
    
    # Reordenación forzosa en tiempo de ejecución para cumplir estrictamente con el modelo
    X_input = X_input[meta_train["numeric_features"]]

    pred = int(pipeline.predict(X_input)[0])
    proba = pipeline.predict_proba(X_input)[0]

    # Imprimir informe final adaptado automáticamente a la estructura
    game_title = game_data.get("name", args.json if args.json else "Juego Analizado")
    print_result(game_title, pred, proba, meta_train)
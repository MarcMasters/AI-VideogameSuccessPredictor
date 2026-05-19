# 🎮 Steam Game Success Predictor

Predice si un videojuego tendrá **Fracaso / Éxito Moderado / Alto Éxito** en Steam
usando únicamente features conocidas **antes del lanzamiento**.

---

## Estructura del proyecto

```
steam_predictor/
├── 00_eda.py           → Análisis exploratorio (opcional, recomendado)
├── 01_preprocess.py    → Limpieza + ingeniería de features + success_score
├── 02_train.py         → Entrenamiento del RandomForest + métricas
├── 03_predict.py       → Predicción interactiva o por JSON
├── examples/
│   └── my_game.json    → Ejemplo de juego para predecir
├── data/
│   └── games.csv       ← AQUÍ VA TU CSV de Kaggle
├── models/             → Generado automáticamente
├── outputs/            → Gráficos generados
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

---

## Uso paso a paso

### 1. Descarga el dataset

Descarga `games.csv` de [Kaggle](https://www.kaggle.com/datasets/artermiloff/steam-games-dataset)
y colócalo en `data/games.csv`.

### 2. (Opcional) Análisis exploratorio

```bash
python 00_eda.py
```
Genera gráficos en `outputs/eda_*.png`.

### 3. Preprocesado

```bash
python 01_preprocess.py
```
Genera `data/processed.parquet` con:
- Features pre-lanzamiento (géneros, tags, precio, plataformas…)
- `success_score` (0–100) calculado desde variables post-lanzamiento
- `success_class` (0=Fracaso / 1=Moderado / 2=Alto éxito)

### 4. Entrenamiento

```bash
python 02_train.py
```
- Cross-validation 5-fold estratificado
- Guarda `models/rf_pipeline.joblib` + `models/metadata.json`
- Guarda gráficos de confusion matrix y feature importances en `outputs/`

### 5. Predicción

**Modo interactivo (CLI):**
```bash
python 03_predict.py
```

**Modo JSON:**
```bash
python 03_predict.py --json examples/my_game.json
```

---

## Fórmula de éxito

```
success_score =
  0.40 × estimated_owners_norm
+ 0.25 × peak_ccu_norm
+ 0.20 × total_reviews_norm
+ 0.10 × pct_positive_norm
+ 0.05 × metacritic_score_norm
```

| Rango     | Clase        |
|-----------|--------------|
| 0 – 40    | 🔴 Fracaso   |
| 40 – 70   | 🟡 Moderado  |
| 70 – 100  | 🟢 Alto éxito |

**Filtro de ruido:** juegos con < 100 reviews totales se excluyen del entrenamiento.

---

## Features pre-lanzamiento utilizadas

| Feature          | Descripción                            |
|------------------|----------------------------------------|
| `price`          | Precio en USD (0 = free-to-play)       |
| `required_age`   | Clasificación de edad (0/7/12/16/18)   |
| `n_languages`    | Número de idiomas soportados           |
| `is_multiplayer` | 1 si tiene multijugador o co-op        |
| `plat_*`         | Windows / Mac / Linux                  |
| `genre_*`        | One-hot top 15 géneros de Steam        |
| `tag_*`          | One-hot top 30 tags de Steam           |
| `publisher_top`  | Publisher (top 160 o "Other")          |
| `desc_length`    | Longitud de la descripción corta       |
| `release_year`   | Año de lanzamiento                     |
| `release_quarter`| Trimestre de lanzamiento (1–4)         |

---

## Modelo

**RandomForestClassifier** (scikit-learn)
- `n_estimators=400`
- `class_weight="balanced"` — compensa el desbalance Fracaso >> Éxito
- `max_features="sqrt"` — reduce correlación entre árboles
- Pipeline con `StandardScaler` + `OneHotEncoder`

---

## Notas

- El modelo predice probabilidades para las 3 clases, no solo la clase ganadora.
- El `publisher_top` puede tener un impacto alto: publishers conocidos tienen ventaja.
- La descripción no se procesa con NLP en esta versión (solo longitud); una mejora
  futura sería añadir embeddings con `sentence-transformers`.

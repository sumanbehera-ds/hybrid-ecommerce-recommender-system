from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/popularity_baseline.pkl")


def train_popularity_model(
    interactions: pd.DataFrame,
    item_col: str = "itemid",
    strength_col: str = "event_strength",
) -> pd.Series:
    return (
        interactions
        .groupby(item_col)[strength_col]
        .sum()
        .sort_values(ascending=False)
    )


def recommend_popular(popularity_model, top_n=10, excluded_items=None):
    top_n = max(int(top_n), 0)
    excluded_items = set(excluded_items or [])

    if hasattr(popularity_model, "index"):
        candidates = popularity_model.index.tolist()
    else:
        candidates = list(popularity_model)

    recommendations = []

    for item in candidates:
        if len(recommendations) >= top_n:
            break

        if item in excluded_items or item in recommendations:
            continue

        recommendations.append(item)

    return recommendations


def save_popularity_model(popularity_model, path: Path = MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity_model, path)


def load_popularity_model(path: Path = MODEL_PATH):
    return joblib.load(path)

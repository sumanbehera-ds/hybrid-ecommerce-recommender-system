from pathlib import Path

import joblib
import pandas as pd

from src.evaluation.metrics import hit_rate_at_k
from src.models.popularity_recommender import (
    MODEL_PATH,
    recommend_popular,
    train_popularity_model,
)


DATA_PATH = Path("data/processed/filtered_events.csv")
DEPLOY_MODEL_PATH = Path("deploy_models/popularity_baseline.pkl")


def temporal_leave_one_out(interactions: pd.DataFrame):
    if "datetime" not in interactions.columns and "timestamp" in interactions.columns:
        interactions = interactions.copy()
        interactions["datetime"] = pd.to_datetime(
            interactions["timestamp"],
            unit="ms"
        )

    interactions = interactions.sort_values(["visitorid", "datetime"])
    test_indices = interactions.groupby("visitorid").tail(1).index

    train_df = interactions.drop(test_indices).reset_index(drop=True)
    test_df = interactions.loc[test_indices].reset_index(drop=True)

    return train_df, test_df


def evaluate_popularity(popularity_model, train_df, test_df, k=10):
    user_history = (
        train_df
        .groupby("visitorid")["itemid"]
        .apply(set)
        .to_dict()
    )

    scores = []

    for row in test_df.itertuples(index=False):
        excluded_items = user_history.get(row.visitorid, set())
        recommendations = recommend_popular(
            popularity_model,
            top_n=k,
            excluded_items=excluded_items,
        )
        scores.append(hit_rate_at_k([row.itemid], recommendations, k=k))

    return sum(scores) / len(scores) if scores else 0.0


def log_mlflow_run(metric_name, metric_value):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Popularity_Baseline"):
        mlflow.log_param("model", "popularity")
        mlflow.log_metric(metric_name, metric_value)


def train_and_evaluate(data_path=DATA_PATH, model_path=MODEL_PATH, k=10):
    interactions = pd.read_csv(data_path)
    train_df, test_df = temporal_leave_one_out(interactions)

    popularity_model = train_popularity_model(train_df)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(popularity_model, model_path)

    hit_rate = evaluate_popularity(popularity_model, train_df, test_df, k=k)
    log_mlflow_run(f"hitrate_at_{k}", hit_rate)

    return popularity_model, hit_rate


if __name__ == "__main__":
    model, score = train_and_evaluate()

    print("Saved popularity baseline to:", MODEL_PATH)
    print("Popularity HitRate@10:", score)

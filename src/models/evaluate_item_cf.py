import argparse
import random
from pathlib import Path

import pandas as pd

from src.evaluation.metrics import hit_rate_at_k
from src.models.recommend_item_cf import (
    load_item_cf_artifacts,
    recommend_item_cf,
)


DATA_PATH = Path("data/processed/filtered_events.csv")


def temporal_user_sequences(interactions: pd.DataFrame):
    if "datetime" not in interactions.columns and "timestamp" in interactions.columns:
        interactions = interactions.copy()
        interactions["datetime"] = pd.to_datetime(
            interactions["timestamp"],
            unit="ms"
        )

    interactions = interactions.sort_values(["visitorid", "datetime"])

    for user_id, user_data in interactions.groupby("visitorid"):
        items = user_data["itemid"].tolist()

        if len(items) < 2:
            continue

        yield user_id, items[:-1], items[-1]


def evaluate_item_cf(
    data_path=DATA_PATH,
    top_n=10,
    sample_users=1000,
    random_state=42,
):
    artifacts = load_item_cf_artifacts()
    interactions = pd.read_csv(data_path)
    user_sequences = list(temporal_user_sequences(interactions))

    if sample_users:
        sample_size = min(sample_users, len(user_sequences))
        user_sequences = random.Random(random_state).sample(
            user_sequences,
            sample_size,
        )

    scores = []

    for _user_id, sequence, target_item in user_sequences:
        recommendations = recommend_item_cf(
            sequence,
            artifacts["item_encoder"],
            artifacts["item_similarity"],
            top_n=top_n,
        )
        scores.append(hit_rate_at_k([target_item], recommendations, k=top_n))

    return sum(scores) / len(scores) if scores else 0.0


def log_mlflow_run(metric_name, metric_value, sample_users):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Item_CF_Evaluation"):
        mlflow.log_param("model", "item_cf")
        mlflow.log_param("sample_users", sample_users)
        mlflow.log_metric(metric_name, metric_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-users", type=int, default=1000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score = evaluate_item_cf(
        data_path=args.data_path,
        top_n=args.top_n,
        sample_users=args.sample_users,
        random_state=args.random_state,
    )
    log_mlflow_run(f"hitrate_at_{args.top_n}", score, args.sample_users)
    print(f"Item-CF HitRate@{args.top_n}:", score)

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import hit_rate_at_k
from src.models.train_ncf import (
    MODEL_PATH,
    NCFDataset,
    NeuralCollaborativeFiltering,
)


TRAIN_PATH = Path("data/processed/ncf_train.csv")
TEST_PATH = Path("data/processed/ncf_test.csv")


def load_ncf_checkpoint(model_path=MODEL_PATH):
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    model = NeuralCollaborativeFiltering(
        num_users=checkpoint["num_users"],
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dims=checkpoint.get("hidden_dims", (128, 64, 32)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def load_ncf_model(model_path=MODEL_PATH):
    model, _checkpoint = load_ncf_checkpoint(model_path)
    return model


def filter_known_indices(frame: pd.DataFrame, checkpoint):
    return frame[
        (frame["user_idx"] < checkpoint["num_users"])
        & (frame["item_idx"] < checkpoint["num_items"])
    ].copy()


def evaluate_ncf_mse(test_path=TEST_PATH, model_path=MODEL_PATH, batch_size=4096):
    test_df = pd.read_csv(test_path)
    model, checkpoint = load_ncf_checkpoint(model_path)
    test_df = filter_known_indices(test_df, checkpoint)
    dataset = NCFDataset(test_df)
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.MSELoss(reduction="sum")

    total_loss = 0.0

    with torch.no_grad():
        for users, items, targets in loader:
            predictions = model(users, items)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(dataset) if len(dataset) else 0.0


def build_user_history(train_df: pd.DataFrame):
    return {
        int(user_idx): set(group["item_idx"].astype(int))
        for user_idx, group in train_df.groupby("user_idx")
    }


def sample_negative_items(num_items, excluded_items, sample_size, rng):
    if sample_size <= 0:
        return []

    sample_size = min(sample_size, num_items - len(excluded_items))

    if sample_size <= 0:
        return []

    if sample_size > num_items // 3:
        candidates = [
            item_idx
            for item_idx in range(num_items)
            if item_idx not in excluded_items
        ]
        return rng.sample(candidates, sample_size)

    negatives = set()
    attempts = 0
    max_attempts = sample_size * 50

    while len(negatives) < sample_size and attempts < max_attempts:
        item_idx = rng.randrange(num_items)
        attempts += 1

        if item_idx not in excluded_items:
            negatives.add(item_idx)

    if len(negatives) < sample_size:
        remaining = [
            item_idx
            for item_idx in range(num_items)
            if item_idx not in excluded_items and item_idx not in negatives
        ]
        negatives.update(rng.sample(remaining, sample_size - len(negatives)))

    return list(negatives)


def rank_ncf_candidates(model, user_idx, candidate_items, top_n):
    users = torch.full(
        (len(candidate_items),),
        user_idx,
        dtype=torch.long,
    )
    items = torch.tensor(candidate_items, dtype=torch.long)

    with torch.no_grad():
        scores = model(users, items)
        ranked_positions = torch.argsort(scores, descending=True).tolist()

    return [
        candidate_items[position]
        for position in ranked_positions[:top_n]
    ]


def evaluate_ncf_hit_rate_at_k(
    test_path=TEST_PATH,
    train_path=TRAIN_PATH,
    model_path=MODEL_PATH,
    top_n=10,
    sample_size=1000,
    negative_samples=100,
    random_state=42,
):
    test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    model, checkpoint = load_ncf_checkpoint(model_path)
    train_df = filter_known_indices(train_df, checkpoint)
    test_df = filter_known_indices(test_df, checkpoint)

    if sample_size:
        test_df = test_df.sample(
            n=min(sample_size, len(test_df)),
            random_state=random_state,
        )

    user_history = build_user_history(train_df)
    rng = random.Random(random_state)
    scores = []

    for row in test_df.itertuples(index=False):
        user_idx = int(row.user_idx)
        target_item = int(row.item_idx)
        excluded_items = set(user_history.get(user_idx, set()))
        excluded_items.add(target_item)
        negatives = sample_negative_items(
            checkpoint["num_items"],
            excluded_items,
            negative_samples,
            rng,
        )

        if not negatives:
            continue

        candidate_items = [target_item] + negatives
        recommendations = rank_ncf_candidates(
            model,
            user_idx,
            candidate_items,
            top_n=top_n,
        )
        scores.append(hit_rate_at_k([target_item], recommendations, k=top_n))

    return sum(scores) / len(scores) if scores else 0.0


def log_mlflow_run(mse, hit_rate, top_n, sample_size, negative_samples):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Neural_CF_Evaluation"):
        mlflow.log_param("model", "ncf")
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("negative_samples", negative_samples)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric(f"hitrate_at_{top_n}", hit_rate)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--negative-samples", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mse = evaluate_ncf_mse(
        test_path=args.test_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    hit_rate = evaluate_ncf_hit_rate_at_k(
        test_path=args.test_path,
        train_path=args.train_path,
        model_path=args.model_path,
        top_n=args.top_n,
        sample_size=args.sample_size,
        negative_samples=args.negative_samples,
        random_state=args.random_state,
    )
    log_mlflow_run(
        mse,
        hit_rate,
        args.top_n,
        args.sample_size,
        args.negative_samples,
    )
    print("NCF Test MSE:", mse)
    print(f"NCF HitRate@{args.top_n}:", hit_rate)

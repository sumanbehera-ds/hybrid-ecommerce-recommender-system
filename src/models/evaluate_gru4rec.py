import argparse
from pathlib import Path

import pandas as pd
import torch

from src.evaluation.metrics import hit_rate_at_k
from src.models.lightweight_recommender import GRU4Rec
from src.models.train_gru4rec import MODEL_PATH, parse_sequence


TEST_PATH = Path("data/processed/gru_test.csv")


def load_gru_checkpoint(model_path=MODEL_PATH):
    checkpoint = torch.load(
        model_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )

    model = GRU4Rec(
        num_items=checkpoint["num_items"],
        embedding_dim=checkpoint.get("embedding_dim", 64),
        hidden_dim=checkpoint.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def recommend_gru(model, checkpoint, sequence, top_n=10):
    item_to_idx = checkpoint["item_to_idx"]
    idx_to_item = checkpoint["idx_to_item"]

    encoded_sequence = [
        item_to_idx[item]
        for item in sequence
        if item in item_to_idx
    ]

    if not encoded_sequence:
        return []

    sequence_tensor = torch.tensor([encoded_sequence], dtype=torch.long)

    with torch.no_grad():
        logits = model(sequence_tensor)
        k = min(top_n + len(encoded_sequence) + 10, logits.shape[-1])
        top_indices = torch.topk(logits, k=k).indices.squeeze(0).tolist()

    seen_items = set(sequence)
    recommendations = []

    for idx in top_indices:
        if idx == 0 or idx not in idx_to_item:
            continue

        item = idx_to_item[idx]

        if item in seen_items or item in recommendations:
            continue

        recommendations.append(item)

        if len(recommendations) >= top_n:
            break

    return recommendations


def evaluate_gru4rec(
    test_path=TEST_PATH,
    model_path=MODEL_PATH,
    top_n=10,
    sample_size=10000,
):
    test_df = pd.read_csv(test_path)

    if sample_size:
        test_df = test_df.sample(
            n=min(sample_size, len(test_df)),
            random_state=42,
        )

    model, checkpoint = load_gru_checkpoint(model_path)
    scores = []

    for row in test_df.itertuples(index=False):
        sequence = parse_sequence(row.input_sequence)
        recommendations = recommend_gru(
            model,
            checkpoint,
            sequence,
            top_n=top_n,
        )
        scores.append(hit_rate_at_k([row.target_item], recommendations, k=top_n))

    return sum(scores) / len(scores) if scores else 0.0


def log_mlflow_run(metric_name, metric_value, sample_size):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="GRU4Rec_Evaluation"):
        mlflow.log_param("model", "gru4rec")
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_metric(metric_name, metric_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=10000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score = evaluate_gru4rec(
        test_path=args.test_path,
        model_path=args.model_path,
        top_n=args.top_n,
        sample_size=args.sample_size,
    )
    log_mlflow_run(f"hitrate_at_{args.top_n}", score, args.sample_size)
    print(f"GRU4Rec HitRate@{args.top_n}:", score)

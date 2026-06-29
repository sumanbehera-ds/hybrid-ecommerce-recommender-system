import argparse
import pandas as pd
from pathlib import Path

from src.models.hybrid_recommender import load_artifacts, hybrid_recommend
from src.models.train_gru4rec import parse_sequence
from src.evaluation.metrics import hit_rate_at_k


TEST_PATH = Path("data/processed/gru_test.csv")


def evaluate_hybrid(test_path=TEST_PATH, top_n=10, sample_size=1000):
    test_df = pd.read_csv(test_path)

    if sample_size:
        test_df = test_df.sample(
            n=min(sample_size, len(test_df)),
            random_state=42,
        )

    artifacts = load_artifacts()
    scores = []

    for row in test_df.itertuples(index=False):
        sequence = parse_sequence(row.input_sequence)
        actual_item = [row.target_item]

        recommendations = hybrid_recommend(
            user_sequence=sequence,
            artifacts=artifacts,
            top_n=top_n,
        )

        scores.append(hit_rate_at_k(actual_item, recommendations, k=top_n))

    return sum(scores) / len(scores) if scores else 0.0


def log_mlflow_run(metric_name, metric_value, sample_size):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Hybrid_Evaluation"):
        mlflow.log_param("model", "hybrid")
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_metric(metric_name, metric_value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--sample-size", type=int, default=1000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    score = evaluate_hybrid(
        test_path=args.test_path,
        top_n=args.top_n,
        sample_size=args.sample_size,
    )
    log_mlflow_run(f"hitrate_at_{args.top_n}", score, args.sample_size)

    print(f"Hybrid HitRate@{args.top_n}:", score)

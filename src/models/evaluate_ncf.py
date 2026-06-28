import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.train_ncf import (
    MODEL_PATH,
    NCFDataset,
    NeuralCollaborativeFiltering,
)


TEST_PATH = Path("data/processed/ncf_test.csv")


def load_ncf_model(model_path=MODEL_PATH):
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

    return model


def evaluate_ncf_mse(test_path=TEST_PATH, model_path=MODEL_PATH, batch_size=4096):
    test_df = pd.read_csv(test_path)
    dataset = NCFDataset(test_df)
    loader = DataLoader(dataset, batch_size=batch_size)
    model = load_ncf_model(model_path)
    criterion = nn.MSELoss(reduction="sum")

    total_loss = 0.0

    with torch.no_grad():
        for users, items, targets in loader:
            predictions = model(users, items)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(dataset) if len(dataset) else 0.0


def log_mlflow_run(mse):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Neural_CF_Evaluation"):
        mlflow.log_param("model", "ncf")
        mlflow.log_metric("test_mse", mse)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", type=Path, default=TEST_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mse = evaluate_ncf_mse(
        test_path=args.test_path,
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    log_mlflow_run(mse)
    print("NCF Test MSE:", mse)

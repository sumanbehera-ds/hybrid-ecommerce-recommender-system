import argparse
import ast
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.models.lightweight_recommender import GRU4Rec


TRAIN_PATH = Path("data/processed/gru_train.csv")
MODEL_PATH = Path("models/gru4rec_model.pth")
DEPLOY_MODEL_PATH = Path("deploy_models/gru4rec_model.pth")


def parse_sequence(value):
    if isinstance(value, list):
        return value

    if isinstance(value, str):
        return ast.literal_eval(value)

    raise TypeError(f"Unsupported sequence value: {value!r}")


def build_item_mappings(frame: pd.DataFrame):
    items = set(frame["target_item"].tolist())

    for value in frame["input_sequence"]:
        items.update(parse_sequence(value))

    sorted_items = sorted(items)
    item_to_idx = {
        item: idx + 1
        for idx, item in enumerate(sorted_items)
    }
    idx_to_item = {
        idx: item
        for item, idx in item_to_idx.items()
    }

    return item_to_idx, idx_to_item


class GRU4RecDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, item_to_idx):
        self.samples = []

        for row in frame.itertuples(index=False):
            sequence = [
                item_to_idx[item]
                for item in parse_sequence(row.input_sequence)
                if item in item_to_idx
            ]
            target = item_to_idx.get(row.target_item)

            if sequence and target:
                self.samples.append((sequence, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence, target = self.samples[idx]

        return (
            torch.tensor(sequence, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )


def collate_sequences(batch):
    sequences, targets = zip(*batch)

    return (
        pad_sequence(sequences, batch_first=True, padding_value=0),
        torch.stack(targets),
    )


def train_gru4rec(
    train_path=TRAIN_PATH,
    model_path=MODEL_PATH,
    embedding_dim=64,
    hidden_dim=128,
    batch_size=1024,
    epochs=1,
    learning_rate=0.001,
    sample_size=100000,
):
    train_df = pd.read_csv(train_path)

    if sample_size:
        train_df = train_df.sample(
            n=min(sample_size, len(train_df)),
            random_state=42,
        )

    item_to_idx, idx_to_item = build_item_mappings(train_df)
    dataset = GRU4RecDataset(train_df, item_to_idx)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRU4Rec(
        num_items=len(item_to_idx),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)

        final_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {final_loss:.6f}")

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "num_items": len(item_to_idx),
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    log_mlflow_run(final_loss, epochs, batch_size, sample_size)

    return checkpoint, final_loss


def log_mlflow_run(final_loss, epochs, batch_size, sample_size):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="GRU4Rec"):
        mlflow.log_param("model", "gru4rec")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_metric("final_train_loss", final_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--deploy-copy", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint, loss = train_gru4rec(
        train_path=args.train_path,
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
    )

    if args.deploy_copy:
        DEPLOY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, DEPLOY_MODEL_PATH)
        print("Copied GRU4Rec model to:", DEPLOY_MODEL_PATH)

    print("Saved GRU4Rec model to:", args.model_path)
    print("Final train loss:", loss)

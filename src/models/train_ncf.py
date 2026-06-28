import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


TRAIN_PATH = Path("data/processed/ncf_train.csv")
MODEL_PATH = Path("models/ncf_model.pth")


class NCFDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.users = torch.tensor(frame["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(frame["item_idx"].values, dtype=torch.long)
        self.targets = torch.tensor(
            frame["event_strength"].values,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.targets[idx]


class NeuralCollaborativeFiltering(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64,
        hidden_dims=(128, 64, 32),
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, user_indices, item_indices):
        user_embedding = self.user_embedding(user_indices)
        item_embedding = self.item_embedding(item_indices)
        features = torch.cat([user_embedding, item_embedding], dim=1)

        return self.network(features).squeeze(1)


def infer_cardinality(frame: pd.DataFrame):
    num_users = int(frame["user_idx"].max()) + 1
    num_items = int(frame["item_idx"].max()) + 1

    return num_users, num_items


def train_ncf(
    train_path=TRAIN_PATH,
    model_path=MODEL_PATH,
    embedding_dim=64,
    batch_size=1024,
    epochs=3,
    learning_rate=0.001,
    sample_size=None,
):
    train_df = pd.read_csv(train_path)

    if sample_size:
        train_df = train_df.sample(
            n=min(sample_size, len(train_df)),
            random_state=42,
        )

    num_users, num_items = infer_cardinality(train_df)

    dataset = NCFDataset(train_df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for users, items, targets in loader:
            users = users.to(device)
            items = items.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(users, items)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)

        final_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs} - train_loss: {final_loss:.6f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "num_users": num_users,
            "num_items": num_items,
            "embedding_dim": embedding_dim,
            "hidden_dims": (128, 64, 32),
        },
        model_path,
    )

    log_mlflow_run(final_loss, epochs, batch_size, embedding_dim)

    return model, final_loss


def log_mlflow_run(final_loss, epochs, batch_size, embedding_dim):
    try:
        import mlflow
    except ModuleNotFoundError:
        return

    mlflow.set_experiment("ecommerce_recommender_system")

    with mlflow.start_run(run_name="Neural_CF"):
        mlflow.log_param("model", "ncf")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("embedding_dim", embedding_dim)
        mlflow.log_metric("final_train_mse", final_loss)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _model, loss = train_ncf(
        train_path=args.train_path,
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        sample_size=args.sample_size,
    )

    print("Saved NCF model to:", args.model_path)
    print("Final train MSE:", loss)

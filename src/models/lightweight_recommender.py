import joblib
import torch
import torch.nn as nn
from pathlib import Path


DEPLOY_MODEL_DIR = Path("deploy_models")

GRU_MODEL_PATH = DEPLOY_MODEL_DIR / "gru4rec_model.pth"
POPULARITY_PATH = DEPLOY_MODEL_DIR / "popularity_baseline.pkl"


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim=64, hidden_dim=128):
        super().__init__()

        self.embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, num_items + 1)

    def forward(self, sequences):
        embedded = self.embedding(sequences)
        output, hidden = self.gru(embedded)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)

        return logits


def load_artifacts():
    popularity_model = joblib.load(POPULARITY_PATH)

    checkpoint = torch.load(
        GRU_MODEL_PATH,
        map_location=torch.device("cpu")
    )

    num_items = checkpoint["num_items"]

    model = GRU4Rec(num_items=num_items)

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    artifacts = {
        "gru_model": model,
        "idx_to_item": checkpoint["idx_to_item"],
        "item_to_idx": checkpoint["item_to_idx"],
        "popularity_model": popularity_model
    }

    return artifacts


def recommend_popular(popularity_model, top_n=10):
    return popularity_model.head(top_n).index.tolist()


def hybrid_recommend(user_sequence, artifacts, top_n=10):
    try:
        item_to_idx = artifacts["item_to_idx"]
        idx_to_item = artifacts["idx_to_item"]
        model = artifacts["gru_model"]

        encoded_sequence = []

        for item in user_sequence:
            if item in item_to_idx:
                encoded_sequence.append(item_to_idx[item])

        if len(encoded_sequence) == 0:
            return recommend_popular(
                artifacts["popularity_model"],
                top_n
            )

        sequence_tensor = torch.tensor(
            [encoded_sequence],
            dtype=torch.long
        )

        with torch.no_grad():
            logits = model(sequence_tensor)

            top_indices = torch.topk(
                logits,
                k=top_n
            ).indices.squeeze().tolist()

        recommendations = []

        for idx in top_indices:
            if idx in idx_to_item:
                recommendations.append(idx_to_item[idx])

        return recommendations

    except Exception:
        return recommend_popular(
            artifacts["popularity_model"],
            top_n
        )
import logging
from pathlib import Path
from typing import Iterable

import joblib
import torch
import torch.nn as nn


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOY_MODEL_DIR = PROJECT_ROOT / "deploy_models"

GRU_MODEL_FILENAME = "gru4rec_model.pth"
POPULARITY_FILENAME = "popularity_baseline.pkl"


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
        _output, hidden = self.gru(embedded)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)

        return logits


def _load_checkpoint(path: Path):
    try:
        return torch.load(
            path,
            map_location=torch.device("cpu"),
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=torch.device("cpu")
        )


def _build_model(checkpoint):
    state_dict = checkpoint["model_state_dict"]
    num_items = int(checkpoint["num_items"])

    embedding_dim = state_dict["embedding.weight"].shape[1]
    hidden_dim = state_dict["gru.weight_hh_l0"].shape[1]

    model = GRU4Rec(
        num_items=num_items,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_artifacts(model_dir=None):
    model_dir = Path(model_dir) if model_dir else DEFAULT_DEPLOY_MODEL_DIR

    gru_model_path = model_dir / GRU_MODEL_FILENAME
    popularity_path = model_dir / POPULARITY_FILENAME

    missing_paths = [
        str(path)
        for path in (gru_model_path, popularity_path)
        if not path.exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            "Missing deployment artifact(s): " + ", ".join(missing_paths)
        )

    popularity_model = joblib.load(popularity_path)
    checkpoint = _load_checkpoint(gru_model_path)

    required_keys = {
        "model_state_dict",
        "num_items",
        "idx_to_item",
        "item_to_idx"
    }

    missing_keys = required_keys.difference(checkpoint)

    if missing_keys:
        raise KeyError(
            "GRU checkpoint is missing required key(s): "
            + ", ".join(sorted(missing_keys))
        )

    model = _build_model(checkpoint)

    return {
        "gru_model": model,
        "idx_to_item": checkpoint["idx_to_item"],
        "item_to_idx": checkpoint["item_to_idx"],
        "popularity_model": popularity_model
    }


def _normalise_top_indices(top_indices):
    if hasattr(top_indices, "tolist"):
        top_indices = top_indices.tolist()

    if isinstance(top_indices, int):
        return [top_indices]

    return [int(idx) for idx in top_indices]


def _append_unique(recommendations, candidates: Iterable, top_n, excluded_items):
    for item in candidates:
        if len(recommendations) >= top_n:
            break

        if item in excluded_items or item in recommendations:
            continue

        recommendations.append(item)


def recommend_popular(popularity_model, top_n=10, excluded_items=None):
    top_n = max(int(top_n), 0)

    if top_n == 0:
        return []

    excluded_items = set(excluded_items or [])

    if hasattr(popularity_model, "index"):
        candidates = popularity_model.index.tolist()
    else:
        candidates = list(popularity_model)

    recommendations = []
    _append_unique(recommendations, candidates, top_n, excluded_items)

    return recommendations


def hybrid_recommend(user_sequence, artifacts, top_n=10):
    top_n = max(int(top_n), 0)

    if top_n == 0:
        return []

    excluded_items = set(user_sequence)

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
            candidate_count = logits.shape[-1]
            k = min(
                max(top_n + len(encoded_sequence) + 10, top_n),
                candidate_count
            )

            top_indices = torch.topk(
                logits,
                k=k
            ).indices.squeeze(0)

        recommendations = []

        for idx in _normalise_top_indices(top_indices):
            if idx == 0 or idx not in idx_to_item:
                continue

            item_id = idx_to_item[idx]

            if item_id in excluded_items or item_id in recommendations:
                continue

            recommendations.append(item_id)

            if len(recommendations) >= top_n:
                break

        if len(recommendations) < top_n:
            fallback_items = recommend_popular(
                artifacts["popularity_model"],
                top_n - len(recommendations),
                excluded_items=excluded_items.union(recommendations)
            )
            recommendations.extend(fallback_items)

        return recommendations

    except Exception as exc:
        LOGGER.warning(
            "GRU inference failed; falling back to popularity: %s",
            exc,
            exc_info=True
        )
        return recommend_popular(
            artifacts["popularity_model"],
            top_n,
            excluded_items=excluded_items
        )

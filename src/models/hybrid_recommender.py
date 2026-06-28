import logging
from pathlib import Path

import joblib
import torch

from src.models.lightweight_recommender import (
    DEFAULT_DEPLOY_MODEL_DIR,
    GRU_MODEL_FILENAME,
    POPULARITY_FILENAME,
    load_artifacts as load_lightweight_artifacts,
    recommend_popular,
)
from src.models.recommend_item_cf import recommend_item_cf


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_DIR = PROJECT_ROOT / "models"
ITEM_SIMILARITY_PATH = LOCAL_MODEL_DIR / "item_similarity.pkl"
ITEM_ENCODER_PATH = LOCAL_MODEL_DIR / "item_encoder.pkl"


def _append_unique(recommendations, candidates, top_n):
    for item in candidates:
        if len(recommendations) >= top_n:
            break

        if item in recommendations:
            continue

        recommendations.append(item)


def _load_gru_and_popularity():
    local_gru_path = LOCAL_MODEL_DIR / GRU_MODEL_FILENAME
    local_popularity_path = LOCAL_MODEL_DIR / POPULARITY_FILENAME

    if local_gru_path.exists() and local_popularity_path.exists():
        return load_lightweight_artifacts(model_dir=LOCAL_MODEL_DIR)

    return load_lightweight_artifacts(model_dir=DEFAULT_DEPLOY_MODEL_DIR)


def load_artifacts():
    artifacts = _load_gru_and_popularity()

    artifacts["item_cf_loaded"] = False
    artifacts["item_similarity"] = None
    artifacts["item_encoder"] = None

    if ITEM_SIMILARITY_PATH.exists() and ITEM_ENCODER_PATH.exists():
        artifacts["item_similarity"] = joblib.load(ITEM_SIMILARITY_PATH)
        artifacts["item_encoder"] = joblib.load(ITEM_ENCODER_PATH)
        artifacts["item_cf_loaded"] = True

    return artifacts


def recommend_gru(user_sequence, artifacts, top_n=10):
    top_n = max(int(top_n), 0)

    if top_n == 0:
        return []

    item_to_idx = artifacts["item_to_idx"]
    idx_to_item = artifacts["idx_to_item"]
    model = artifacts["gru_model"]
    seen_items = set(user_sequence)

    encoded_sequence = [
        item_to_idx[item]
        for item in user_sequence
        if item in item_to_idx
    ]

    if not encoded_sequence:
        return []

    sequence_tensor = torch.tensor([encoded_sequence], dtype=torch.long)

    with torch.no_grad():
        logits = model(sequence_tensor)
        k = min(top_n + len(encoded_sequence) + 10, logits.shape[-1])
        top_indices = torch.topk(logits, k=k).indices.squeeze(0).tolist()

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


def recommend_item_cf_safe(user_sequence, artifacts, top_n=10):
    if not artifacts.get("item_cf_loaded"):
        return []

    try:
        return recommend_item_cf(
            user_sequence=user_sequence,
            item_encoder=artifacts["item_encoder"],
            item_similarity=artifacts["item_similarity"],
            top_n=top_n,
        )
    except Exception as exc:
        LOGGER.warning(
            "Item-CF inference failed; continuing hybrid fallback: %s",
            exc,
            exc_info=True,
        )
        return []


def hybrid_recommend(user_sequence, artifacts, top_n=10):
    top_n = max(int(top_n), 0)

    if top_n == 0:
        return []

    recommendations = []
    seen_items = set(user_sequence)

    try:
        gru_candidates = recommend_gru(user_sequence, artifacts, top_n=top_n)
        _append_unique(recommendations, gru_candidates, top_n)
    except Exception as exc:
        LOGGER.warning(
            "GRU inference failed inside hybrid recommender: %s",
            exc,
            exc_info=True,
        )

    if len(recommendations) < top_n:
        item_cf_candidates = recommend_item_cf_safe(
            user_sequence,
            artifacts,
            top_n=top_n - len(recommendations),
        )
        _append_unique(recommendations, item_cf_candidates, top_n)

    if len(recommendations) < top_n:
        popularity_candidates = recommend_popular(
            artifacts["popularity_model"],
            top_n=top_n - len(recommendations),
            excluded_items=seen_items.union(recommendations),
        )
        _append_unique(recommendations, popularity_candidates, top_n)

    return recommendations


def artifact_status(artifacts):
    return {
        "gru_loaded": artifacts.get("gru_model") is not None,
        "popularity_loaded": artifacts.get("popularity_model") is not None,
        "item_cf_loaded": bool(artifacts.get("item_cf_loaded")),
    }

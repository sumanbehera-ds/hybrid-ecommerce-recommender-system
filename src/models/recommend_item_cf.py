from pathlib import Path

import joblib
import numpy as np


ITEM_ENCODER_PATH = Path("models/item_encoder.pkl")
SIMILARITY_PATH = Path("models/item_similarity.pkl")


def load_item_cf_artifacts(
    item_encoder_path: Path = ITEM_ENCODER_PATH,
    similarity_path: Path = SIMILARITY_PATH,
):
    return {
        "item_encoder": joblib.load(item_encoder_path),
        "item_similarity": joblib.load(similarity_path),
    }


def build_item_to_idx(item_encoder):
    return {
        item: idx
        for idx, item in enumerate(item_encoder.classes_)
    }


def recommend_item_cf(
    user_sequence,
    item_encoder,
    item_similarity,
    top_n=10,
    excluded_items=None,
):
    top_n = max(int(top_n), 0)
    excluded_items = set(excluded_items or user_sequence)

    if top_n == 0:
        return []

    item_to_idx = build_item_to_idx(item_encoder)
    candidate_scores = {}

    for item in user_sequence:
        item_idx = item_to_idx.get(item)

        if item_idx is None:
            continue

        row = item_similarity.getrow(item_idx)

        if row.nnz == 0:
            continue

        for idx, score in zip(row.indices, row.data):
            item_id = item_encoder.classes_[idx].item() if hasattr(item_encoder.classes_[idx], "item") else item_encoder.classes_[idx]

            if item_id in excluded_items:
                continue

            candidate_scores[item_id] = max(
                candidate_scores.get(item_id, 0.0),
                float(score),
            )

    ranked_items = sorted(
        candidate_scores,
        key=candidate_scores.get,
        reverse=True,
    )

    return ranked_items[:top_n]


if __name__ == "__main__":
    artifacts = load_item_cf_artifacts()
    sample_sequence = [int(artifacts["item_encoder"].classes_[0])]
    print(
        recommend_item_cf(
            sample_sequence,
            artifacts["item_encoder"],
            artifacts["item_similarity"],
            top_n=10,
        )
    )

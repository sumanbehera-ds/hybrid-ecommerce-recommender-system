import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity


MATRIX_PATH = Path("models/interaction_matrix.npz")
SIMILARITY_PATH = Path("models/item_similarity.pkl")


def load_interaction_matrix(path: Path = MATRIX_PATH):
    try:
        return load_npz(path)
    except Exception:
        return joblib.load(path)


def keep_top_k_sparse_rows(matrix: csr_matrix, top_k: int):
    matrix = matrix.tocsr()
    rows = []
    cols = []
    values = []

    for row_idx in range(matrix.shape[0]):
        row_start, row_end = matrix.indptr[row_idx], matrix.indptr[row_idx + 1]
        row_cols = matrix.indices[row_start:row_end]
        row_values = matrix.data[row_start:row_end]

        if row_values.size == 0:
            continue

        if row_values.size > top_k:
            top_positions = np.argpartition(row_values, -top_k)[-top_k:]
            row_cols = row_cols[top_positions]
            row_values = row_values[top_positions]

        rows.extend([row_idx] * len(row_cols))
        cols.extend(row_cols.tolist())
        values.extend(row_values.tolist())

    return csr_matrix(
        (values, (rows, cols)),
        shape=matrix.shape,
        dtype=matrix.dtype,
    )


def train_item_similarity(interaction_matrix, top_k=None):
    item_user_matrix = interaction_matrix.T.tocsr()
    item_similarity = cosine_similarity(
        item_user_matrix,
        dense_output=False,
    ).tocsr()

    item_similarity.setdiag(0)
    item_similarity.eliminate_zeros()

    if top_k:
        item_similarity = keep_top_k_sparse_rows(item_similarity, top_k)

    return item_similarity


def save_item_similarity(item_similarity, path: Path = SIMILARITY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(item_similarity, path, compress=3)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-path", type=Path, default=MATRIX_PATH)
    parser.add_argument("--output-path", type=Path, default=SIMILARITY_PATH)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    matrix = load_interaction_matrix(args.matrix_path)
    similarity = train_item_similarity(matrix, top_k=args.top_k)
    save_item_similarity(similarity, args.output_path)

    print("Item similarity shape:", similarity.shape)
    print("Saved item similarity to:", args.output_path)

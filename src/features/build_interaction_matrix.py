import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder


INPUT_PATH = Path("data/processed/filtered_events.csv")

USER_ENCODER_PATH = Path("models/user_encoder.pkl")
ITEM_ENCODER_PATH = Path("models/item_encoder.pkl")
MATRIX_PATH = Path("models/interaction_matrix.npz")


def load_data(path):
    return pd.read_csv(path)


def encode_users_items(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["visitorid"])
    df["item_idx"] = item_encoder.fit_transform(df["itemid"])

    return df, user_encoder, item_encoder


def build_sparse_matrix(df):
    matrix = csr_matrix(
        (
            df["event_strength"],
            (df["user_idx"], df["item_idx"])
        )
    )

    return matrix


if __name__ == "__main__":
    df = load_data(INPUT_PATH)

    df, user_encoder, item_encoder = encode_users_items(df)

    interaction_matrix = build_sparse_matrix(df)

    USER_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(user_encoder, USER_ENCODER_PATH)
    joblib.dump(item_encoder, ITEM_ENCODER_PATH)
    joblib.dump(interaction_matrix, MATRIX_PATH)

    print("Interaction matrix shape:", interaction_matrix.shape)
    print("Saved interaction matrix and encoders.")
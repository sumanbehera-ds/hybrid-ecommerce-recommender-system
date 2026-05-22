import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/processed/filtered_events.csv")

OUTPUT_TRAIN = Path("data/processed/ncf_train.csv")
OUTPUT_TEST = Path("data/processed/ncf_test.csv")

USER_ENCODER_PATH = Path("models/user_encoder.pkl")
ITEM_ENCODER_PATH = Path("models/item_encoder.pkl")


def load_data():
    return pd.read_csv(DATA_PATH)


def encode_data(df, user_encoder, item_encoder):
    df["user_idx"] = user_encoder.transform(df["visitorid"])
    df["item_idx"] = item_encoder.transform(df["itemid"])
    return df


if __name__ == "__main__":
    df = load_data()

    user_encoder = joblib.load(USER_ENCODER_PATH)
    item_encoder = joblib.load(ITEM_ENCODER_PATH)

    df = encode_data(df, user_encoder, item_encoder)

    ncf_df = df[["user_idx", "item_idx", "event_strength"]]

    train_df, test_df = train_test_split(
        ncf_df,
        test_size=0.2,
        random_state=42
    )

    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Saved NCF train/test data.")
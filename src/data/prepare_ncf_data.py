import pandas as pd
import joblib
from pathlib import Path


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


def temporal_leave_one_out_split(df: pd.DataFrame):
    if df.empty:
        return df.copy(), df.copy()

    if "datetime" not in df.columns and "timestamp" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    sort_columns = ["visitorid"]

    if "datetime" in df.columns:
        sort_columns.append("datetime")

    df = df.sort_values(sort_columns).reset_index(drop=True)
    user_counts = df.groupby("visitorid").size()
    eligible_users = user_counts[user_counts > 1].index
    eligible_rows = df[df["visitorid"].isin(eligible_users)]
    test_indices = eligible_rows.groupby("visitorid").tail(1).index

    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(test_indices).reset_index(drop=True)

    item_column = "item_idx" if "item_idx" in df.columns else "itemid"
    train_items = set(train_df[item_column])
    valid_test_mask = test_df[item_column].isin(train_items)

    if not valid_test_mask.all():
        cold_item_rows = test_df.loc[~valid_test_mask]
        train_df = pd.concat([train_df, cold_item_rows], ignore_index=True)
        test_df = test_df.loc[valid_test_mask].reset_index(drop=True)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data()

    user_encoder = joblib.load(USER_ENCODER_PATH)
    item_encoder = joblib.load(ITEM_ENCODER_PATH)

    df = encode_data(df, user_encoder, item_encoder)

    columns = [
        "visitorid",
        "itemid",
        "user_idx",
        "item_idx",
        "event_strength",
    ]

    if "timestamp" in df.columns:
        columns.append("timestamp")

    if "datetime" in df.columns:
        columns.append("datetime")

    ncf_df = df[columns]
    train_df, test_df = temporal_leave_one_out_split(ncf_df)

    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Saved NCF train/test data.")

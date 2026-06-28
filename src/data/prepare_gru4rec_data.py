import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/filtered_events.csv")

TRAIN_OUTPUT = Path("data/processed/gru_train.csv")
TEST_OUTPUT = Path("data/processed/gru_test.csv")


def create_sequences(df):
    if "datetime" not in df.columns and "timestamp" in df.columns:
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.sort_values(["visitorid", "datetime"])

    sequences = []

    for user_id, user_data in df.groupby("visitorid"):
        items = user_data["itemid"].tolist()
        times = user_data["datetime"].tolist()

        if len(items) < 3:
            continue

        for i in range(1, len(items)):
            input_seq = items[:i]
            target_item = items[i]

            sequences.append({
                "visitorid": user_id,
                "input_sequence": input_seq,
                "target_item": target_item,
                "target_datetime": times[i]
            })

    return pd.DataFrame(sequences)


def leave_one_out_split(sequence_df):
    if sequence_df.empty:
        return sequence_df.copy(), sequence_df.copy()

    sort_columns = ["visitorid"]

    if "target_datetime" in sequence_df.columns:
        sort_columns.append("target_datetime")

    sequence_df = sequence_df.sort_values(sort_columns).reset_index(drop=True)
    test_indices = sequence_df.groupby("visitorid").tail(1).index

    test_df = sequence_df.loc[test_indices].reset_index(drop=True)
    train_df = sequence_df.drop(test_indices).reset_index(drop=True)

    return train_df, test_df


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    sequence_df = create_sequences(df)

    train_df, test_df = leave_one_out_split(sequence_df)

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Example sequence:")
    print(train_df.head())

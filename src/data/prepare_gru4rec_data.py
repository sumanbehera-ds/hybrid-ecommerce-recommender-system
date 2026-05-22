import pandas as pd
from pathlib import Path


INPUT_PATH = Path("data/processed/filtered_events.csv")

TRAIN_OUTPUT = Path("data/processed/gru_train.csv")
TEST_OUTPUT = Path("data/processed/gru_test.csv")


def create_sequences(df):
    df = df.sort_values(["visitorid", "datetime"])

    sequences = []

    for user_id, user_data in df.groupby("visitorid"):
        items = user_data["itemid"].tolist()

        if len(items) < 3:
            continue

        for i in range(1, len(items)):
            input_seq = items[:i]
            target_item = items[i]

            sequences.append({
                "visitorid": user_id,
                "input_sequence": input_seq,
                "target_item": target_item
            })

    return pd.DataFrame(sequences)


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    sequence_df = create_sequences(df)

    split_idx = int(len(sequence_df) * 0.8)

    train_df = sequence_df.iloc[:split_idx]
    test_df = sequence_df.iloc[split_idx:]

    TRAIN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("Example sequence:")
    print(train_df.head())
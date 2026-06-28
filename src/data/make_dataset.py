from pathlib import Path

from src.data.prepare_interactions import (
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    load_events,
    prepare_interactions,
    save_data,
)


def build_dataset(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
    min_user_interactions: int = 5,
):
    events = load_events(raw_path)
    interactions = prepare_interactions(
        events,
        min_user_interactions=min_user_interactions
    )
    save_data(interactions, processed_path)

    return interactions


if __name__ == "__main__":
    dataset = build_dataset()
    print("Saved filtered interactions to:", PROCESSED_DATA_PATH)
    print("Shape:", dataset.shape)

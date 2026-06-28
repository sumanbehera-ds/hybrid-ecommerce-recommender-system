from pathlib import Path

import pandas as pd


INPUT_PATH = Path("data/processed/filtered_events.csv")
ITEM_FEATURES_OUTPUT = Path("data/processed/item_features.csv")
USER_FEATURES_OUTPUT = Path("data/processed/user_features.csv")


def build_item_features(interactions: pd.DataFrame) -> pd.DataFrame:
    item_features = (
        interactions
        .groupby("itemid")
        .agg(
            interaction_count=("event_strength", "size"),
            total_strength=("event_strength", "sum"),
            unique_visitors=("visitorid", "nunique"),
            last_interaction_at=("datetime", "max"),
        )
        .reset_index()
    )

    return item_features


def build_user_features(interactions: pd.DataFrame) -> pd.DataFrame:
    user_features = (
        interactions
        .groupby("visitorid")
        .agg(
            interaction_count=("event_strength", "size"),
            total_strength=("event_strength", "sum"),
            unique_items=("itemid", "nunique"),
            last_interaction_at=("datetime", "max"),
        )
        .reset_index()
    )

    return user_features


def build_features(interactions: pd.DataFrame):
    if "datetime" not in interactions.columns and "timestamp" in interactions.columns:
        interactions = interactions.copy()
        interactions["datetime"] = pd.to_datetime(
            interactions["timestamp"],
            unit="ms"
        )

    return (
        build_item_features(interactions),
        build_user_features(interactions),
    )


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    item_features, user_features = build_features(df)

    ITEM_FEATURES_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    item_features.to_csv(ITEM_FEATURES_OUTPUT, index=False)
    user_features.to_csv(USER_FEATURES_OUTPUT, index=False)

    print("Saved item features to:", ITEM_FEATURES_OUTPUT)
    print("Saved user features to:", USER_FEATURES_OUTPUT)

import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/raw/events.csv")
PROCESSED_DATA_PATH = Path("data/processed/filtered_events.csv")


def load_events(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_interactions(events: pd.DataFrame, min_user_interactions: int = 5) -> pd.DataFrame:
    events = events.drop_duplicates().copy()

    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")
    events = events.sort_values(["visitorid", "datetime"])

    session_lengths = events.groupby("visitorid").size()
    valid_users = session_lengths[session_lengths >= min_user_interactions].index

    filtered_events = events[events["visitorid"].isin(valid_users)].copy()

    event_weights = {
        "view": 1,
        "addtocart": 3,
        "transaction": 5,
    }

    filtered_events["event_strength"] = filtered_events["event"].map(event_weights)

    return filtered_events


def save_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    events = load_events(RAW_DATA_PATH)
    filtered_events = prepare_interactions(events)
    save_data(filtered_events, PROCESSED_DATA_PATH)

    print("Saved filtered interactions to:", PROCESSED_DATA_PATH)
    print("Shape:", filtered_events.shape)
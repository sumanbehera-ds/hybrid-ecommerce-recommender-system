import unittest


try:
    import pandas as pd
    from src.models.popularity_recommender import (
        recommend_popular,
        train_popularity_model,
    )
    from src.models.train_baseline import (
        evaluate_popularity,
        temporal_leave_one_out,
    )
except ModuleNotFoundError:
    pd = None


@unittest.skipIf(pd is None, "pandas is not installed")
class BaselineTest(unittest.TestCase):
    def test_popularity_model_ranks_by_weighted_strength(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 2, 3],
                "itemid": [10, 20, 20],
                "event_strength": [5, 1, 2],
            }
        )

        model = train_popularity_model(interactions)

        self.assertEqual(model.index.tolist(), [10, 20])

    def test_recommend_popular_excludes_seen_items(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 2, 3],
                "itemid": [10, 20, 30],
                "event_strength": [5, 4, 3],
            }
        )
        model = train_popularity_model(interactions)

        self.assertEqual(
            recommend_popular(model, top_n=2, excluded_items={10}),
            [20, 30],
        )

    def test_temporal_leave_one_out_uses_last_user_event(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 1, 2, 2],
                "itemid": [10, 11, 20, 21],
                "timestamp": [1000, 2000, 1000, 2000],
                "event_strength": [1, 1, 1, 1],
            }
        )

        train_df, test_df = temporal_leave_one_out(interactions)

        self.assertEqual(test_df.sort_values("visitorid")["itemid"].tolist(), [11, 21])
        self.assertEqual(train_df.sort_values("visitorid")["itemid"].tolist(), [10, 20])

    def test_evaluate_popularity_returns_mean_hit_rate(self):
        train_df = pd.DataFrame(
            {
                "visitorid": [1, 2],
                "itemid": [10, 20],
                "event_strength": [1, 5],
            }
        )
        test_df = pd.DataFrame(
            {
                "visitorid": [1, 2],
                "itemid": [20, 10],
                "event_strength": [1, 1],
            }
        )
        model = train_popularity_model(train_df)

        self.assertEqual(evaluate_popularity(model, train_df, test_df, k=1), 1.0)


if __name__ == "__main__":
    unittest.main()

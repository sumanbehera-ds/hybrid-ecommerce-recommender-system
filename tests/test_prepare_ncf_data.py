import unittest


try:
    import pandas as pd
    from src.data.prepare_ncf_data import temporal_leave_one_out_split
except ModuleNotFoundError:
    pd = None


@unittest.skipIf(pd is None, "pandas is not installed")
class PrepareNCFDataTest(unittest.TestCase):
    def test_temporal_split_uses_last_known_item_per_repeat_user_as_test(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 1, 1, 2, 2, 3, 4, 4],
                "itemid": [10, 20, 30, 30, 40, 50, 40, 20],
                "user_idx": [0, 0, 0, 1, 1, 2, 3, 3],
                "item_idx": [0, 1, 2, 2, 3, 4, 3, 1],
                "event_strength": [1.0] * 8,
                "timestamp": [1000, 2000, 3000, 1000, 2000, 1000, 1000, 2000],
            }
        )

        train_df, test_df = temporal_leave_one_out_split(interactions)

        self.assertEqual(set(test_df["visitorid"]), {1, 2, 4})
        self.assertEqual(
            test_df.sort_values("visitorid")["itemid"].tolist(),
            [30, 40, 20],
        )
        self.assertIn(3, set(train_df["visitorid"]))

    def test_temporal_split_keeps_cold_test_items_in_train(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 1, 2, 2],
                "itemid": [10, 99, 20, 10],
                "user_idx": [0, 0, 1, 1],
                "item_idx": [0, 99, 1, 0],
                "event_strength": [1.0] * 4,
                "timestamp": [1000, 2000, 1000, 2000],
            }
        )

        train_df, test_df = temporal_leave_one_out_split(interactions)

        self.assertEqual(test_df["visitorid"].tolist(), [2])
        self.assertIn(99, set(train_df["item_idx"]))


if __name__ == "__main__":
    unittest.main()

import unittest


try:
    import pandas as pd
    from src.data.prepare_gru4rec_data import (
        create_sequences,
        leave_one_out_split,
    )
except ModuleNotFoundError:
    pd = None


@unittest.skipIf(pd is None, "pandas is not installed")
class PrepareGru4RecDataTest(unittest.TestCase):
    def test_leave_one_out_split_uses_last_sequence_per_user_as_test(self):
        interactions = pd.DataFrame(
            {
                "visitorid": [1, 1, 1, 2, 2, 2],
                "itemid": [10, 11, 12, 20, 21, 22],
                "timestamp": [1000, 2000, 3000, 1000, 2000, 3000],
            }
        )

        sequences = create_sequences(interactions)
        train_df, test_df = leave_one_out_split(sequences)

        self.assertEqual(set(test_df["visitorid"]), {1, 2})
        self.assertEqual(test_df.sort_values("visitorid")["target_item"].tolist(), [12, 22])
        self.assertEqual(train_df.sort_values("visitorid")["target_item"].tolist(), [11, 21])


if __name__ == "__main__":
    unittest.main()

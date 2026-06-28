import unittest


try:
    import torch
    from src.models.lightweight_recommender import (
        hybrid_recommend,
        recommend_popular,
    )
except ModuleNotFoundError:
    torch = None


class _FakeIndex:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakePopularity:
    def __init__(self, values):
        self.index = _FakeIndex(values)


class _FakeModel:
    def __call__(self, _sequence_tensor):
        return torch.tensor([[0.0, 0.2, 0.9, 0.4]])


@unittest.skipIf(torch is None, "torch is not installed")
class LightweightRecommenderTest(unittest.TestCase):
    def test_recommend_popular_excludes_seen_items(self):
        popularity = _FakePopularity([1, 2, 3])

        self.assertEqual(
            recommend_popular(popularity, top_n=2, excluded_items={1}),
            [2, 3],
        )

    def test_hybrid_recommend_handles_top_n_one_without_scalar_loop_bug(self):
        artifacts = {
            "item_to_idx": {99: 1},
            "idx_to_item": {1: 99, 2: 42, 3: 13},
            "gru_model": _FakeModel(),
            "popularity_model": _FakePopularity([99, 7, 8]),
        }

        self.assertEqual(
            hybrid_recommend([99], artifacts, top_n=1),
            [42],
        )

    def test_hybrid_recommend_falls_back_for_unknown_sequence(self):
        artifacts = {
            "item_to_idx": {99: 1},
            "idx_to_item": {1: 99},
            "gru_model": _FakeModel(),
            "popularity_model": _FakePopularity([7, 8, 9]),
        }

        self.assertEqual(
            hybrid_recommend([123456], artifacts, top_n=2),
            [7, 8],
        )


if __name__ == "__main__":
    unittest.main()

import unittest


try:
    import numpy as np
    from src.models.recommend_item_cf import recommend_item_cf
except ModuleNotFoundError:
    np = None


class _FakeEncoder:
    classes_ = np.array([10, 20, 30]) if np is not None else []


class _FakeRow:
    indices = np.array([1, 2]) if np is not None else []
    data = np.array([0.9, 0.2]) if np is not None else []
    nnz = 2


class _FakeSimilarity:
    def getrow(self, _idx):
        return _FakeRow()


@unittest.skipIf(np is None, "numpy is not installed")
class ItemCFTest(unittest.TestCase):
    def test_recommend_item_cf_ranks_similarity_and_excludes_seen(self):
        recommendations = recommend_item_cf(
            user_sequence=[10],
            item_encoder=_FakeEncoder(),
            item_similarity=_FakeSimilarity(),
            top_n=2,
        )

        self.assertEqual(recommendations, [20, 30])


if __name__ == "__main__":
    unittest.main()

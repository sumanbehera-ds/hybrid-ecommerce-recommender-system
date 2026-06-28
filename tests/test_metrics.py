import unittest

from src.evaluation.metrics import hit_rate_at_k, recall_at_k


class MetricsTest(unittest.TestCase):
    def test_hit_rate_is_binary_hit(self):
        self.assertEqual(hit_rate_at_k([1, 2], [9, 2, 3], k=3), 1.0)
        self.assertEqual(hit_rate_at_k([1, 2], [9, 8, 7], k=3), 0.0)

    def test_recall_at_k_keeps_ratio_behavior(self):
        self.assertEqual(recall_at_k([1, 2, 3], [1, 9, 3], k=3), 2 / 3)

    def test_empty_actual_items_return_zero(self):
        self.assertEqual(hit_rate_at_k([], [1, 2], k=2), 0.0)
        self.assertEqual(recall_at_k([], [1, 2], k=2), 0.0)


if __name__ == "__main__":
    unittest.main()

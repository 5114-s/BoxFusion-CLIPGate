import unittest

import torch

from boxfusion.spgroup_official_adapter import _knn, _scatter_max, _scatter_mean


class AdapterPrimitiveTest(unittest.TestCase):
    def test_scatter(self) -> None:
        source = torch.tensor([[1.0, 4.0], [3.0, 2.0], [5.0, 8.0]])
        index = torch.tensor([0, 0, 1])
        self.assertTrue(torch.equal(_scatter_mean(source, index), torch.tensor([[2.0, 3.0], [5.0, 8.0]])))
        maximum, _ = _scatter_max(source, index)
        self.assertTrue(torch.equal(maximum, torch.tensor([[3.0, 4.0], [5.0, 8.0]])))

    def test_knn_shape_and_values(self) -> None:
        support = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [5.0, 0.0, 0.0]]])
        query = torch.tensor([[[1.8, 0.0, 0.0], [4.7, 0.0, 0.0]]])
        result = _knn(2, support, query)
        self.assertEqual(tuple(result.shape), (1, 2, 2))
        self.assertEqual(int(result[0, 0, 0]), 1)
        self.assertEqual(int(result[0, 0, 1]), 2)


if __name__ == "__main__":
    unittest.main()

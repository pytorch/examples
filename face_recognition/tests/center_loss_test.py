import unittest

import torch
import numpy

from loss import get_center_delta, compute_center_loss
from device import device


class CenterLossTest(unittest.TestCase):

    def setUp(self):
        # Mock features, centers and targets
        self.features = torch.tensor(
            ((1, 2, 3), (4, 5, 6), (7, 8, 9))
        ).float().to(device)

        self.centers = torch.tensor(
            ((1, 1, 1), (2, 2, 2), (3, 3, 3), (5, 5, 5))
        ).float().to(device)

        self.targets = torch.tensor((1, 3, 1)).to(device)
        self.alpha = 0.1

    def test_get_center_delta(self):
        result = get_center_delta(
            self.features, self.centers, self.targets, self.alpha)
        # size should match
        self.assertTrue(result.size() == self.centers.size())
        # for class 1
        class1_result = -(
            (self.features[0] + self.features[2]) -
            2 * self.centers[1]) / 3 * self.alpha

        self.assertEqual(3, torch.sum(result[1] == class1_result).item())
        # for class 3
        class3_result = -(self.features[1] - self.centers[3]) / 2 * self.alpha
        self.assertEqual(3, torch.sum(result[3] == class3_result).item())

        # others should all be zero
        sum_others = torch.sum(result[(0, 2), :]).item()
        self.assertEqual(0, sum_others)

    def test_compute_center_loss(self):

        loss = torch.mean(
            (self.features[(0, 2, 1), :] - self.centers[(1, 1, 3), :]) ** 2)

        self.assertEqual(loss, compute_center_loss(
            self.features, self.centers, self.targets))

if __name__ == '__main__':
    unittest.main()
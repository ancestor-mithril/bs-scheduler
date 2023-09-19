import math
import unittest

from torch.optim.lr_scheduler import LambdaLR

from bs_scheduler.batch_size_schedulers import LambdaBS
from tests.test_utils import create_dataloader, dataloader_len, step_n_epochs


class TestLambdaBS(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.dataloader = create_dataloader(batch_size=self.batch_size)
        self.dataset_len = len(self.dataloader.dataset)

    def assert_real_eq_inferred(self, real, inferred):
        self.assertEqual(real, inferred, "Dataloader __len__ does not return the real length. The real length should "
                                         "always be equal to the inferred length except for Iterable Datasets for "
                                         "which the __len__ could be inaccurate.")

    def assert_real_eq_expected(self, real, expected):
        self.assertEqual(real, expected, f"Expected {expected}, got {real}")

    def test_sanity(self):
        real, inferred = dataloader_len(self.dataloader)
        self.assert_real_eq_inferred(real, inferred)

        self.dataloader.batch_sampler.batch_size = 32
        real, inferred = dataloader_len(self.dataloader)
        self.assert_real_eq_inferred(real, inferred)

    def test_dataloader_lengths(self):
        fn = lambda epoch: (1 + epoch) ** 1.05
        self.scheduler = LambdaBS(self.dataloader, fn)
        n_epochs = 3

        lengths = step_n_epochs(self.dataloader, self.scheduler, n_epochs)

        expected_batch_sizes = [int(self.batch_size * fn(epoch)) for epoch in range(n_epochs)]
        expected_lengths = [int(math.ceil(self.dataset_len / bs)) for bs in expected_batch_sizes]

        self.assert_real_eq_expected(lengths, expected_lengths)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

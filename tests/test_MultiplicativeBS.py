import math
import unittest

from bs_scheduler.batch_size_schedulers import MultiplicativeBS
from tests.test_utils import create_dataloader, iterate, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs


class TestMultiplicativeBS(unittest.TestCase):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def assert_real_eq_expected(self, real, expected):
        self.assertEqual(real, expected, f"Expected {expected}, got {real}")

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 1
        scheduler = MultiplicativeBS(dataloader, fn)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [self.base_batch_size]
        for epoch in range(n_epochs):
            expected_batch_sizes.append(int(expected_batch_sizes[-1] * fn(epoch)))
        expected_batch_sizes.pop(0)  # Removing first
        expected_lengths = [int(math.ceil(len(self.dataset) / bs)) for bs in expected_batch_sizes]

        self.assert_real_eq_expected(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 1
        scheduler = MultiplicativeBS(dataloader, fn)
        n_epochs = 300

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [int(self.base_batch_size * fn(epoch)) for epoch in range(n_epochs)]

        self.assert_real_eq_expected(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

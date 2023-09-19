import math
import unittest

from torch.optim.lr_scheduler import LambdaLR

from bs_scheduler.batch_size_schedulers import LambdaBS
from tests.test_utils import create_dataloader, iterate, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs


class TestLambdaBS(unittest.TestCase):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def assert_real_eq_inferred(self, real, inferred):
        self.assertEqual(real, inferred, "Dataloader __len__ does not return the real length. The real length should "
                                         "always be equal to the inferred length except for Iterable Datasets for "
                                         "which the __len__ could be inaccurate.")

    def assert_real_eq_expected(self, real, expected):
        self.assertEqual(real, expected, f"Expected {expected}, got {real}")

    def test_sanity(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        real, inferred = iterate(dataloader)
        self.assert_real_eq_inferred(real, inferred)

        dataloader.batch_sampler.batch_size = 256
        real, inferred = iterate(dataloader)
        self.assert_real_eq_inferred(real, inferred)

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: (1 + epoch) ** 1.05
        scheduler = LambdaBS(dataloader, fn)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)

        expected_batch_sizes = [int(self.base_batch_size * fn(epoch)) for epoch in range(n_epochs)]
        expected_lengths = [int(math.ceil(len(self.dataset) / bs)) for bs in expected_batch_sizes]

        self.assert_real_eq_expected(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: 1 + epoch / 20
        scheduler = LambdaBS(dataloader, fn)
        n_epochs = 300

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [int(self.base_batch_size * fn(epoch)) for epoch in range(n_epochs)]

        self.assert_real_eq_expected(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

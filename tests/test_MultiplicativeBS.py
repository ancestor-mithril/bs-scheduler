import unittest

from bs_scheduler import MultiplicativeBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip


class TestMultiplicativeBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, fn, min_batch_size, max_batch_size):
        expected_batch_sizes = [base_batch_size]
        for epoch in range(1, epochs):
            batch_size = int(expected_batch_sizes[-1] * fn(epoch))
            batch_size = clip(batch_size, min_batch_size, max_batch_size)
            expected_batch_sizes.append(batch_size)
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 1
        scheduler = MultiplicativeBS(dataloader, fn)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, fn,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)

        self.assert_real_eq_expected(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 2
        scheduler = MultiplicativeBS(dataloader, fn)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, fn,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)

        self.assert_real_eq_expected(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

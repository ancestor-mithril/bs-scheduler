import unittest

from bs_scheduler import ConstantBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip


class TestConstantBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, step_size, gamma, min_batch_size, max_batch_size):
        return None
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        for epoch in range(epochs):
            last_batch_size = expected_batch_sizes[-1]
            if epoch == 0 or epoch % step_size != 0:
                expected_batch_sizes.append(last_batch_size)
            else:
                expected_batch_sizes.append(clip(int(last_batch_size * gamma), min_batch_size, max_batch_size))
        expected_batch_sizes.pop(0)  # Removing base batch size.
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        factor = 5.0
        milestone = 5
        scheduler = ConstantBS(dataloader, factor=factor, milestone=milestone)
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [int(self.base_batch_size * factor)] * 5 + [self.base_batch_size] * 5
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)

        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        factor = 5.0
        milestone = 5
        scheduler = ConstantBS(dataloader, factor=factor, milestone=milestone, max_batch_size=100, verbose=False)
        n_epochs = 15
        self.assertAlmostEqual(scheduler.factor, scheduler.max_batch_size / self.base_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [int(self.base_batch_size * scheduler.factor)] * 5 + [self.base_batch_size] * 10

        self.assertEqual(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

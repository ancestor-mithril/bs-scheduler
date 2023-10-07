import unittest

from bs_scheduler import IncreaseBSOnPlateau
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    BSTest


class TestIncreaseBSOnPlateau(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.
        # TODO: Check that create_dataloader throws errors when it should.

    def test_constant_metric(self):
        base_batch_size = 10
        max_batch_size = 100

        n_epochs = 100
        metrics = [{"metric": 0.1}] * n_epochs

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = IncreaseBSOnPlateau(dataloader, mode='min', threshold_mode='rel', max_batch_size=max_batch_size)
        epoch_lengths1 = simulate_n_epochs(dataloader, scheduler1, metrics)

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler2 = IncreaseBSOnPlateau(dataloader, mode='min', threshold_mode='abs', max_batch_size=max_batch_size)
        epoch_lengths2 = simulate_n_epochs(dataloader, scheduler2, metrics)

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler3 = IncreaseBSOnPlateau(dataloader, mode='max', threshold_mode='rel', max_batch_size=max_batch_size)
        epoch_lengths3 = simulate_n_epochs(dataloader, scheduler3, metrics)

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler4 = IncreaseBSOnPlateau(dataloader, mode='max', threshold_mode='abs', max_batch_size=max_batch_size)
        epoch_lengths4 = simulate_n_epochs(dataloader, scheduler4, metrics)

        expected_batch_sizes = [10] * 12 + [20] * 11 + [40] * 11 + [80] * 11 + [100] * 55
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths1, expected_lengths)
        self.assertEqual(epoch_lengths2, expected_lengths)
        self.assertEqual(epoch_lengths3, expected_lengths)
        self.assertEqual(epoch_lengths4, expected_lengths)

        # TODO: Test different factors
        # TODO: Test patience
        # TODO: Test threshold
        # TODO: Test threshold mode and mode
        # TODO: Test cooldown


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

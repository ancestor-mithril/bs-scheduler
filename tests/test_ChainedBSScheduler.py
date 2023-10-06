import unittest

from bs_scheduler import ChainedBSScheduler, ConstantBS, ExponentialBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest


class TestChainedBSScheduler(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.
        # TODO: Test more combinations of batch size schedulers.
        # TODO: Check that create_dataloader throws errors when it should.

    def test_dataloader_lengths(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = ChainedBSScheduler([scheduler1, scheduler2])
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [100, 110, 121, 133, 14, 15, 16, 18, 20, 22]
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = ChainedBSScheduler([scheduler1, scheduler2])
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [100, 110, 121, 133, 14, 15, 16, 18, 20, 22]

        self.assertEqual(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

import unittest

from bs_scheduler import CosineAnnealingBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest


class TestCosineAnnealingBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def test_dataloader_lengths(self):
        base_batch_size = 10
        total_iters = 5
        n_epochs = 50
        max_batch_size = 100

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 19, 41, 69, 91, 100, 91, 67, 37, 13] * 5
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        total_iters = 5
        n_epochs = 50
        max_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 19, 41, 69, 91, 100, 91, 67, 37, 13] * 5

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        total_iters = 5
        max_batch_size = 100
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.max_batch_size, max_batch_size)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

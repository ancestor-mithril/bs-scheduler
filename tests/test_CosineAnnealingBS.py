import os
import unittest

from bs_scheduler import CosineAnnealingBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, batched_dataset


class TestCosineAnnealingBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'total_iters': 5,
            'max_batch_size': 100,
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, CosineAnnealingBS, batch_size_manager, **kwargs)

    def test_dataloader_lengths(self):
        base_batch_size = 10
        total_iters = 5
        n_epochs = 50
        max_batch_size = 100

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 19, 41, 69, 91, 100, 91, 69, 41, 19] * 5
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        total_iters = 50
        n_epochs = 200
        max_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 10, 10, 11, 11, 12, 13, 14, 16, 17, 19, 20, 22, 24, 26, 29, 31, 33, 36, 38, 41, 44,
                                47, 49, 52, 55, 58, 61, 63, 66, 69, 72, 74, 77, 79, 81, 84, 86, 88, 90, 91, 93, 94, 96,
                                97, 98, 99, 99, 100, 100, 100, 100, 100, 99, 99, 98, 97, 96, 94, 93, 91, 90, 88, 86, 84,
                                81, 79, 77, 74, 72, 69, 66, 63, 61, 58, 55, 52, 49, 47, 44, 41, 38, 36, 33, 31, 29, 26,
                                24, 22, 20, 19, 17, 16, 14, 13, 12, 11, 11, 10, 10] * 2

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

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        total_iters = 5
        n_epochs = 50
        max_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBS(dataloader, total_iters=total_iters, max_batch_size=max_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CosineAnnealingBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=0.001)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CosineAnnealingLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

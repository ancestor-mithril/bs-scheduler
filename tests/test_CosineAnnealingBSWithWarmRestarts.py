import os
import unittest

from bs_scheduler import CosineAnnealingBSWithWarmRestarts, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, batched_dataset


class TestCosineAnnealingBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            't_0': 5,
            'max_batch_size': 100,
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, CosineAnnealingBSWithWarmRestarts, batch_size_manager, **kwargs)

    def test_dataloader_lengths(self):
        base_batch_size = 10
        t_0 = 5
        n_epochs = 60
        max_batch_size = 100

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, max_batch_size=max_batch_size)

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 19, 41, 69, 91, 100] * 10
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        t_0 = 5
        n_epochs = 60
        max_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, max_batch_size=max_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 19, 41, 69, 91, 100] * 10

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        t_0 = 5
        max_batch_size = 100
        scheduler = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, max_batch_size=max_batch_size)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.t_0, t_0)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        t_0 = 5
        n_epochs = 60
        max_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, factor=t_0 // 2,
                                                      max_batch_size=max_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CosineAnnealingBSWithWarmRestarts.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=t_0 // 2,
                                                                         eta_min=0.001)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CosineAnnealingWarmRestarts.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

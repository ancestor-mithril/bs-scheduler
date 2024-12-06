import os
import unittest

import torch

from bs_scheduler import LinearBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip, rint, batched_dataset


class TestLinearBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'start_factor': 10.0,
            'end_factor': 5.0,
            'milestone': 5
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, LinearBS, batch_size_manager, **kwargs)

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, start_factor, end_factor, milestone, min_batch_size,
                                     max_batch_size):
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        factors = torch.linspace(start_factor, end_factor, milestone + 1)
        factors[1:] = factors[1:] / factors[:-1]
        for epoch in range(epochs):
            last_batch_size = expected_batch_sizes[-1]
            if epoch > milestone:
                expected_batch_sizes.append(last_batch_size)
            else:
                last_batch_size *= factors[epoch].item()
                last_batch_size = clip(rint(last_batch_size), min_batch_size, max_batch_size)
                expected_batch_sizes.append(last_batch_size)
        expected_batch_sizes.pop(0)  # Removing base batch size.
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        start_factor = 12.0
        end_factor = 3.0
        milestone = 5
        scheduler = LinearBS(dataloader, verbose=False, start_factor=start_factor, end_factor=end_factor,
                             milestone=milestone)
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size,
                                                                 start_factor=start_factor, end_factor=end_factor,
                                                                 milestone=milestone,
                                                                 min_batch_size=scheduler.min_batch_size,
                                                                 max_batch_size=scheduler.max_batch_size)
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        start_factor = 6.0
        end_factor = 1.0
        milestone = 5
        scheduler = LinearBS(dataloader, start_factor=start_factor, end_factor=end_factor, milestone=milestone,
                             max_batch_size=100, verbose=False)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [60, 50, 40, 30, 20] + [10] * 10

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        start_factor = 6.0
        end_factor = 1.0
        milestone = 5
        scheduler = LinearBS(dataloader, start_factor=start_factor, end_factor=end_factor, milestone=milestone,
                             max_batch_size=100, verbose=False)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.milestone, milestone)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        start_factor = 6.0
        end_factor = 1.0
        milestone = 5
        scheduler = LinearBS(dataloader, start_factor=start_factor, end_factor=end_factor, milestone=milestone,
                             max_batch_size=100, verbose=False)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/LinearBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / start_factor, end_factor=end_factor,
                                                      total_iters=milestone)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/LinearLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

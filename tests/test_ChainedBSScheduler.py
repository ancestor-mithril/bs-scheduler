import os
import unittest

from bs_scheduler import ChainedBSScheduler, ConstantBS, ExponentialBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest


class TestChainedBSScheduler(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test more combinations of batch size schedulers.

    def test_dataloader_lengths(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = ChainedBSScheduler([scheduler1, scheduler2])
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [100, 110, 121, 133, 14, 16, 17, 19, 21, 23]
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
        expected_batch_sizes = [100, 110, 121, 133, 14, 16, 17, 19, 21, 23]

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        factor = 10
        scheduler1 = ConstantBS(dataloader, factor=factor, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = ChainedBSScheduler([scheduler1, scheduler2])

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.schedulers[0].factor, factor)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = ChainedBSScheduler([scheduler1, scheduler2])
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/ChainedBSScheduler.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=4)
        scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0 / 1.1)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        n_epochs = 10
        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/ChainedScheduler.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

import os
import unittest

from bs_scheduler import ConstantBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, rint, batched_dataset


class TestConstantBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'factor': 5.0,
            'milestone': 5
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, ConstantBS, batch_size_manager, **kwargs)

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        factor = 5.0
        milestone = 5
        scheduler = ConstantBS(dataloader, factor=factor, milestone=milestone)
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [rint(self.base_batch_size * factor)] * 5 + [self.base_batch_size] * 5
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
        expected_batch_sizes = [rint(self.base_batch_size * scheduler.factor)] * 5 + [self.base_batch_size] * 10

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        factor = 5.0
        milestone = 5
        scheduler = ConstantBS(dataloader, factor=factor, milestone=milestone, max_batch_size=100, verbose=True)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertTrue(scheduler.verbose)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        factor = 5.0
        milestone = 5
        scheduler = ConstantBS(dataloader, factor=factor, milestone=milestone, max_batch_size=100, verbose=False)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/ConstantBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=4)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/ConstantLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

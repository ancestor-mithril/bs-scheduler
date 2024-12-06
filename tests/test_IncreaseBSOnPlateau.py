import os
import unittest

from bs_scheduler import IncreaseBSOnPlateau, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    BSTest, get_batch_sizes_across_epochs, batched_dataset


class TestIncreaseBSOnPlateau(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'mode': 'min',
            'threshold_mode': 'rel',
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, IncreaseBSOnPlateau, batch_size_manager, **kwargs)

    def test_constant_metric(self):
        base_batch_size = 10
        max_batch_size = 100

        n_epochs = 100
        metrics = [{"metrics": 0.1}] * n_epochs

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

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        mode = 'min'
        threshold_mode = 'rel'
        scheduler = IncreaseBSOnPlateau(dataloader, mode=mode, threshold_mode=threshold_mode)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step(metrics=10)
        self.assertEqual(scheduler.mode, mode)
        self.assertEqual(scheduler.threshold_mode, threshold_mode)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        max_batch_size = 100

        n_epochs = 100
        metrics = [{"metrics": 0.1}] * n_epochs

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = IncreaseBSOnPlateau(dataloader, mode='min', threshold_mode='rel', max_batch_size=max_batch_size)
        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, metrics)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/IncreaseBSOnPlateau.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold_mode='rel',
                                                               min_lr=0.001, factor=0.5)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        metrics = [{"metrics": 0.1}] * n_epochs
        for d in metrics:
            learning_rates.append(get_lr(optimizer))
            scheduler.step(**d)
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/ReduceLROnPlateau.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

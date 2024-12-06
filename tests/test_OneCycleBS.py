import os
import unittest

from bs_scheduler import OneCycleBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, rint, batched_dataset


class TestOneCycleBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'max_batch_size': 100,
            'min_batch_size': 10,
            'total_steps': 100,
            'decay_percentage': 0.3,
            'strategy': 'linear',
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, OneCycleBS, batch_size_manager, **kwargs)

    def test_dataloader_batch_size_linear(self):
        base_batch_size = 40
        n_epochs = 120
        max_batch_size = 80
        min_batch_size = 10
        total_steps = 100
        decay_percentage = 0.3
        strategy = 'linear'

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage,
                               strategy=strategy, max_batch_size=max_batch_size, min_batch_size=min_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        phase_1 = list(range(40, 10, -1))
        phase_2 = list(range(10, 80, 1))
        end = [max_batch_size] * 20
        expected_batch_sizes = phase_1 + phase_2 + end

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_dataloader_batch_size_cos(self):
        base_batch_size = 40
        n_epochs = 120
        max_batch_size = 80
        min_batch_size = 10
        total_steps = 100
        decay_percentage = 0.3
        strategy = 'cos'

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage,
                               strategy=strategy, max_batch_size=max_batch_size, min_batch_size=min_batch_size)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        phase_1 = [40, 40, 40, 39, 39, 38, 37, 36, 35, 34, 32, 31, 30, 28, 27, 25, 23, 22, 20, 19, 18, 16, 15, 14, 13,
                   12, 11, 11, 10, 10, 10]
        phase_2 = [10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30,
                   31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 48, 50, 51, 53, 54, 56, 57, 59, 60, 62, 63, 64, 66, 67,
                   68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 77, 78, 78, 79, 79, 79, 80, 80, 80]
        end = [80] * 20
        expected_batch_sizes = phase_1 + phase_2 + end

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        base_batch_size = 40
        max_batch_size = 80
        min_batch_size = 10
        total_steps = 100
        decay_percentage = 0.3
        strategy = 'cos'

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage,
                               strategy=strategy, max_batch_size=max_batch_size, min_batch_size=min_batch_size)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.end_step_1, rint(decay_percentage * total_steps))

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 40
        n_epochs = 120
        max_batch_size = 80
        min_batch_size = 10
        total_steps = 120
        decay_percentage = 0.3
        strategy = 'cos'

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage,
                               strategy=strategy, max_batch_size=max_batch_size, min_batch_size=min_batch_size)
        # TODO: not equivalent, OneCycleBS can step more than total_steps
        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/OneCycleBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=total_steps, max_lr=0.05,
                                                        final_div_factor=8.0, div_factor=4.0)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/OneCycleLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

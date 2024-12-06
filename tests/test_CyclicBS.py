import os
import random
import unittest

from bs_scheduler import CyclicBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, fashion_mnist, get_batch_sizes_across_epochs, BSTest, rint, \
    batched_dataset


class TestConstantBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'base_batch_size': 100,
            'step_size_down': 10,
            'mode': 'triangular',
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, CyclicBS, batch_size_manager, **kwargs)

    def test_dataloader_batch_size_triangular(self):
        base_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 200
        min_batch_size = 10
        step_size_down = 10
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_down=step_size_down, mode='triangular', min_batch_size=min_batch_size)
        n_epochs = 4 * step_size_down

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        step = rint((base_batch_size - min_batch_size) / step_size_down)
        decreasing_range = list(range(base_batch_size, min_batch_size, -step))
        increasing_range = list(range(min_batch_size, base_batch_size, step))
        expected_batch_sizes = (decreasing_range + increasing_range) * 2

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_dataloader_batch_size_triangular2(self):
        # This test is not reliable, it has rounding errors that need to be handled by hand
        base_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 200
        min_batch_size = 10
        step_size_down = 10
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_down=step_size_down, mode='triangular2', min_batch_size=min_batch_size)
        n_epochs = 4 * step_size_down

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        step = rint((base_batch_size - min_batch_size) / step_size_down)
        decreasing_range = list(range(base_batch_size, min_batch_size, -step))
        increasing_range = list(range(min_batch_size, base_batch_size, step))
        decreasing_range_2 = [base_batch_size]
        for _ in range(step_size_down - 1):
            decreasing_range_2.append(decreasing_range_2[-1] - step / 2.0)
        decreasing_range_2[-1] -= 1e-8  # Rounding error
        increasing_range_2 = [decreasing_range_2[-1] - step / 2.0] + list(reversed(decreasing_range_2))[:-1]
        decreasing_range_2 = list(map(rint, decreasing_range_2))
        increasing_range_2 = list(map(rint, increasing_range_2))
        expected_batch_sizes = decreasing_range + increasing_range + decreasing_range_2 + increasing_range_2

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_dataloader_batch_size_exp_range(self):
        base_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 200
        min_batch_size = 10
        step_size_down = 10
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_down=step_size_down, mode='exp_range', gamma=gamma,
                             min_batch_size=min_batch_size)
        n_epochs = 4 * step_size_down

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        def calculate_increasing_and_decreasing_range(base_batch_size, step_size_down, step, count):
            decreasing_range = [base_batch_size]
            count += 1
            for i in range(1, step_size_down):
                decreasing_range.append(base_batch_size - step * i * gamma ** count)
                count += 1
            increasing_range = [base_batch_size - step * step_size_down * gamma ** count]
            count += 1
            for i in range(step_size_down - 1, 0, -1):
                increasing_range.append(base_batch_size - step * i * gamma ** count)
                count += 1

            return decreasing_range, increasing_range, count

        step = rint((base_batch_size - min_batch_size) / step_size_down)
        count = 0

        decreasing_range, increasing_range, count = calculate_increasing_and_decreasing_range(
            base_batch_size, step_size_down, step, count)
        decreasing_range_2, increasing_range_2, count = calculate_increasing_and_decreasing_range(
            base_batch_size, step_size_down, step, count)

        decreasing_range = list(map(rint, decreasing_range))
        increasing_range = list(map(rint, increasing_range))
        decreasing_range_2 = list(map(rint, decreasing_range_2))
        increasing_range_2 = list(map(rint, increasing_range_2))
        expected_batch_sizes = decreasing_range + increasing_range + decreasing_range_2 + increasing_range_2

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading_triangular(self):
        dataloader = create_dataloader(self.dataset)
        upper_batch_size = 300
        scheduler = CyclicBS(dataloader, dataloader.batch_size, upper_batch_size, mode='triangular')
        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.scale_fn(rint(random.random() * 1000)), 1.0)

    def test_loading_and_unloading_triangular2(self):
        dataloader = create_dataloader(self.dataset)
        upper_batch_size = 300
        scheduler = CyclicBS(dataloader, dataloader.batch_size, upper_batch_size, mode='triangular2')
        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.scale_fn(2), 0.5)
        self.assertEqual(scheduler.scale_fn(1), 1.0)
        self.assertEqual(scheduler.scale_fn(3), 0.25)

    def test_loading_and_unloading_exp_range(self):
        dataloader = create_dataloader(self.dataset)
        upper_batch_size = 300
        gamma = 1.1
        scheduler = CyclicBS(dataloader, dataloader.batch_size, upper_batch_size, mode='exp_range', gamma=gamma)
        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.scale_fn(2), gamma ** 2.0)
        self.assertEqual(scheduler.scale_fn(1), gamma ** 1.0)
        self.assertEqual(scheduler.scale_fn(3), gamma ** 3.0)

    def test_loading_and_unloading_scale_fn(self):
        dataloader = create_dataloader(self.dataset)
        upper_batch_size = 300
        scheduler = CyclicBS(dataloader, dataloader.batch_size, upper_batch_size, scale_fn=lambda x: 0.25)
        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.scale_fn(rint(random.random() * 1000)), 0.25)

    def test_graphic_triangular2(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 200
        step_size_down = 10
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_down=step_size_down, mode='triangular2', gamma=gamma)
        n_epochs = 4 * step_size_down

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CyclicBS-triangular2.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='triangular2',
                                                      gamma=gamma, step_size_up=step_size_down)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CyclicLR-triangular2.png")
        plt.close()

    def test_graphic_exp_range(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 100
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 200
        step_size_down = 25
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_down=step_size_down, mode='exp_range', gamma=gamma)
        n_epochs = 6 * step_size_down

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CyclicBS-exp_range.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='exp_range',
                                                      gamma=gamma, step_size_down=step_size_down)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/CyclicLR-exp_range.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

import random
import unittest

from bs_scheduler import CyclicBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, rint


class TestConstantBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def test_dataloader_batch_size_triangular(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 100
        step_size_up = 10
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_up=step_size_up, mode='triangular')
        n_epochs = 4 * step_size_up

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        step = rint((max_batch_size - base_batch_size) / step_size_up)
        increasing_range = list(range(base_batch_size, max_batch_size, step))
        decreasing_range = list(range(max_batch_size, base_batch_size, -step))
        expected_batch_sizes = (increasing_range + decreasing_range) * 2

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_dataloader_batch_size_triangular2(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 100
        step_size_up = 10
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_up=step_size_up, mode='triangular2')
        n_epochs = 4 * step_size_up

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        step = rint((max_batch_size - base_batch_size) / step_size_up)
        increasing_range = list(range(base_batch_size, max_batch_size, step))
        decreasing_range = list(range(max_batch_size, base_batch_size, -step))
        increasing_range_2 = [base_batch_size]
        for _ in range(step_size_up - 1):
            increasing_range_2.append(increasing_range_2[-1] + step / 2.0)

        increasing_range_2[3] -= 1e-8  # Rounding errors ...
        increasing_range_2[9] += 1e-8  # Rounding errors ...
        # I don't know if this generalizes for other pairs of base batch size and upper batch size.
        # TODO: Find a solution for better numerical stability.

        decreasing_range_2 = [increasing_range_2[-1] + step / 2.0] + list(reversed(increasing_range_2))[
                                                                     :step_size_up - 1]
        increasing_range_2 = list(map(rint, increasing_range_2))
        decreasing_range_2 = list(map(rint, decreasing_range_2))
        expected_batch_sizes = increasing_range + decreasing_range + increasing_range_2 + decreasing_range_2

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_dataloader_batch_size_exp_range(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 100
        step_size_up = 10
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_up=step_size_up, mode='exp_range', gamma=gamma)
        n_epochs = 4 * step_size_up

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)

        def calculate_increasing_and_decreasing_range(base_batch_size, step_size_up, step, count):
            increasing_range = [base_batch_size]
            count += 1
            for i in range(1, step_size_up):
                increasing_range.append(base_batch_size + step * i * gamma ** count)
                count += 1
            decreasing_range = [base_batch_size + step * step_size_up * gamma ** count]
            count += 1
            for i in range(step_size_up - 1, 0, -1):
                decreasing_range.append(base_batch_size + step * i * gamma ** count)
                count += 1

            return increasing_range, decreasing_range, count

        step = rint((max_batch_size - base_batch_size) / step_size_up)
        count = 0

        increasing_range, decreasing_range, count = calculate_increasing_and_decreasing_range(
            base_batch_size, step_size_up, step, count)
        increasing_range_2, decreasing_range_2, count = calculate_increasing_and_decreasing_range(
            base_batch_size, step_size_up, step, count)

        increasing_range = list(map(rint, increasing_range))
        decreasing_range = list(map(rint, decreasing_range))
        increasing_range_2 = list(map(rint, increasing_range_2))
        decreasing_range_2 = list(map(rint, decreasing_range_2))
        expected_batch_sizes = increasing_range + decreasing_range + increasing_range_2 + decreasing_range_2

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

        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 100
        step_size_up = 10
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_up=step_size_up, mode='triangular2', gamma=gamma)
        n_epochs = 4 * step_size_up

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        plt.savefig("CyclicBS-triangular2.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='triangular2',
                                                      gamma=gamma, step_size_up=step_size_up)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        plt.savefig("CyclicLR-triangular2.png")
        plt.close()

    def test_graphic_exp_range(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        max_batch_size = 100
        step_size_up = 50
        gamma = 0.9
        scheduler = CyclicBS(dataloader, base_batch_size=base_batch_size, max_batch_size=max_batch_size,
                             step_size_up=step_size_up, mode='exp_range', gamma=gamma)
        n_epochs = 10 * step_size_up

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        plt.savefig("CyclicBS-exp_range.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, mode='exp_range',
                                                      gamma=gamma, step_size_up=step_size_up)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        plt.savefig("CyclicLR-exp_range.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

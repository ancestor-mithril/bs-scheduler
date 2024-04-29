import os
import unittest

from bs_scheduler import PolynomialBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest


class TestPolynomialBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def test_dataloader_lengths(self):
        base_batch_size = 10
        total_iters = 5
        power = 1.0
        n_epochs = 20

        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler = PolynomialBS(dataloader, total_iters=total_iters, power=power, verbose=False)

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [10, 12, 16, 24] + [48] * 16
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        total_iters = 10
        power = 1.0
        n_epochs = 10
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        scheduler = PolynomialBS(dataloader, total_iters=total_iters, power=power, max_batch_size=100, verbose=False)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [64, 71, 80, 91, 100, 100, 100, 100, 100, 100]

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        total_iters = 10
        power = 1.0
        scheduler = PolynomialBS(dataloader, total_iters=total_iters, power=power, verbose=False)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.power, power)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        total_iters = 10
        power = 1.0
        n_epochs = 10
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        scheduler = PolynomialBS(dataloader, total_iters=total_iters, power=power, verbose=False)

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/PolynomialBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.1)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/PolynomialLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

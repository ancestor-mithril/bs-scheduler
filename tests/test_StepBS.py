import os
import unittest

from bs_scheduler import StepBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip, rint


class TestStepBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, step_size, gamma, min_batch_size, max_batch_size):
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        for epoch in range(epochs):
            last_batch_size = expected_batch_sizes[-1]
            if epoch == 0 or epoch % step_size != 0:
                expected_batch_sizes.append(last_batch_size)
            else:
                expected_batch_sizes.append(clip(rint(last_batch_size * gamma), min_batch_size, max_batch_size))
        expected_batch_sizes.pop(0)  # Removing base batch size.
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        step_size = 50
        gamma = 1.1
        scheduler = StepBS(dataloader, step_size=step_size, gamma=gamma)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, step_size, gamma,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)

        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        step_size = 5
        gamma = 3.0
        scheduler = StepBS(dataloader, step_size=step_size, gamma=gamma, max_batch_size=5000, verbose=False)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, step_size, gamma,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        step_size = 5
        gamma = 3.0
        scheduler = StepBS(dataloader, step_size=step_size, gamma=gamma, max_batch_size=5000, verbose=False)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.step_size, step_size)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        step_size = 50
        gamma = 1.1
        scheduler = StepBS(dataloader, step_size=step_size, gamma=gamma)
        n_epochs = 300

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/StepBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.9)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/StepLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

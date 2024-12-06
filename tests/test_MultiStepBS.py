import os
import unittest

from bs_scheduler import MultiStepBS, CustomBatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip, rint, batched_dataset


class TestMultiStepBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    def test_create(self):
        dataloader = create_dataloader(batched_dataset(batch_size=self.base_batch_size), batch_size=None)
        kwargs = {
            'gamma': 1.1,
            'milestones': [70, 70, 80, 10, 50]
        }
        batch_size_manager = CustomBatchSizeManager(dataloader.dataset)
        self.create_scheduler(dataloader, MultiStepBS, batch_size_manager, **kwargs)

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, milestones, gamma, min_batch_size, max_batch_size):
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        for epoch in range(epochs):
            last_batch_size = expected_batch_sizes[-1]
            for _ in range(milestones.count(epoch)):
                last_batch_size *= gamma
            expected_batch_sizes.append(clip(rint(last_batch_size), min_batch_size, max_batch_size))
        expected_batch_sizes.pop(0)  # Removing base batch size
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        milestones = [70, 70, 80, 10, 50]
        gamma = 1.1
        scheduler = MultiStepBS(dataloader, milestones=milestones, gamma=gamma)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, milestones, gamma,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)

        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        milestones = [5, 7, 7, 9]
        gamma = 3.0
        scheduler = MultiStepBS(dataloader, milestones=milestones, gamma=gamma, max_batch_size=5000, verbose=False)
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, milestones, gamma,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        milestones = [5, 7, 7, 9]
        gamma = 3.0
        scheduler = MultiStepBS(dataloader, milestones=milestones, gamma=gamma, max_batch_size=5000, verbose=False)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(scheduler.gamma, gamma)

    def test_graphic(self):
        import matplotlib.pyplot as plt
        import torch
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        milestones = [5, 7, 7, 9]
        gamma = 3.0
        scheduler = MultiStepBS(dataloader, milestones=milestones, gamma=gamma, max_batch_size=5000, verbose=False)
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        plt.plot(batch_sizes)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/MultiStepBS.png")
        plt.close()

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=1.0 / gamma)
        learning_rates = []

        def get_lr(optimizer):
            for param_group in optimizer.param_groups:
                return param_group['lr']

        for _ in range(n_epochs):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()
        plt.plot(learning_rates)
        os.makedirs("images", exist_ok=True)
        plt.savefig("images/MultiStepLR.png")
        plt.close()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

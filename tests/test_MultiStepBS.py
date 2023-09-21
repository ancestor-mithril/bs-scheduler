import unittest

from bs_scheduler import MultiStepBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip


class TestMultiStepBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, milestones, gamma, min_batch_size, max_batch_size):
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        for epoch in range(epochs):
            last_batch_size = expected_batch_sizes[-1]
            for _ in range(milestones.count(epoch)):
                last_batch_size *= gamma
            expected_batch_sizes.append(clip(int(last_batch_size), min_batch_size, max_batch_size))
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
        milestones = [5, 10, 10, 12]
        gamma = 3.0
        scheduler = MultiStepBS(dataloader, milestones=milestones, gamma=gamma, max_batch_size=5000, verbose=False)
        n_epochs = 15

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, milestones, gamma,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)

        self.assertEqual(batch_sizes, expected_batch_sizes)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

import unittest

from bs_scheduler import MultiplicativeBS
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest, clip, rint


class TestMultiplicativeBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()

    @staticmethod
    def compute_expected_batch_sizes(epochs, base_batch_size, fn, min_batch_size, max_batch_size):
        expected_batch_sizes = [base_batch_size]  # Base batch size is added as a boundary condition.
        for epoch in range(epochs):
            batch_size = rint(expected_batch_sizes[-1] * fn(epoch))
            batch_size = clip(batch_size, min_batch_size, max_batch_size)
            expected_batch_sizes.append(batch_size)
        expected_batch_sizes.pop(0)  # Removing base batch size.
        return expected_batch_sizes

    def test_dataloader_lengths(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 1  # noqa: E731
        scheduler = MultiplicativeBS(dataloader, fn)
        n_epochs = 300

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, fn,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)

        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        fn = lambda epoch: epoch / 100 + 2  # noqa: E731
        scheduler = MultiplicativeBS(dataloader, fn, max_batch_size=5000, verbose=False)
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = self.compute_expected_batch_sizes(n_epochs, self.base_batch_size, fn,
                                                                 scheduler.min_batch_size, scheduler.max_batch_size)

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        fn = lambda epoch: 10 * epoch  # noqa: E731
        scheduler = MultiplicativeBS(dataloader, fn)

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        # TODO: Test that function objects are saved and can work again
        self.assertEqual(scheduler.bs_lambda(10), 100)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

import os
import unittest

from bs_scheduler import SequentialBS, ConstantBS, ExponentialBS, LinearBS, BatchSizeManager
from tests.test_utils import create_dataloader, simulate_n_epochs, fashion_mnist, \
    get_batch_sizes_across_epochs, BSTest


class TestSequentialBS(BSTest):
    def setUp(self):
        self.base_batch_size = 64
        self.dataset = fashion_mnist()
        # TODO: Test multiple dataloaders: dataloader with workers, dataloaders with samplers, with drop last and
        #  without drop last and so on.

    def test_dataloader_lengths(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        scheduler = SequentialBS(schedulers=[scheduler1, scheduler2], milestones=[5])
        n_epochs = 10

        epoch_lengths = simulate_n_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [100] * 4 + [10, 11, 12, 13, 15, 16]
        expected_lengths = self.compute_epoch_lengths(expected_batch_sizes, len(self.dataset), drop_last=False)
        self.assertEqual(epoch_lengths, expected_lengths)

    def test_dataloader_batch_size(self):
        base_batch_size = 10
        dataloader = create_dataloader(self.dataset, batch_size=base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4, max_batch_size=100)
        scheduler2 = ExponentialBS(dataloader, gamma=2, max_batch_size=100, verbose=False)
        scheduler = SequentialBS(schedulers=[scheduler1, scheduler2], milestones=[5])
        n_epochs = 10

        batch_sizes = get_batch_sizes_across_epochs(dataloader, scheduler, n_epochs)
        expected_batch_sizes = [100] * 4 + [10, 20, 40, 80, 100, 100]

        self.assertEqual(batch_sizes, expected_batch_sizes)

    def test_preconditions(self):
        dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
        scheduler1 = ConstantBS(dataloader, factor=10, milestone=4, max_batch_size=100)
        scheduler2 = ExponentialBS(dataloader, gamma=2, max_batch_size=100, verbose=False)
        scheduler3 = LinearBS(dataloader, max_batch_size=100, verbose=False)
        if len(os.getenv('PYTHONOPTIMIZE', '')) == 0:
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=scheduler1, milestones=[5])
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=[scheduler1], milestones=[5])
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=[scheduler1, scheduler2], milestones=5)
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=[scheduler1, scheduler2], milestones=[])
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=[scheduler1, scheduler2], milestones=[0])
            with self.assertRaises(AssertionError):
                SequentialBS(schedulers=[scheduler1, scheduler2, scheduler3], milestones=[10, 5])
            with self.assertRaises(ValueError):
                SequentialBS(schedulers=[scheduler1, scheduler2, scheduler3], milestones=[10])

        SequentialBS(schedulers=[scheduler1, scheduler2, scheduler3], milestones=[5, 10])
        with self.assertRaises(ValueError):
            other_dataloader = create_dataloader(self.dataset, batch_size=self.base_batch_size)
            other_scheduler = ConstantBS(other_dataloader, factor=10, milestone=4, max_batch_size=100)
            SequentialBS(schedulers=[other_scheduler, scheduler2, scheduler3], milestones=[5, 10])

        with self.assertRaises(ValueError):
            class OtherBatchManager(BatchSizeManager):
                def get_current_batch_size(self) -> int:
                    return 10

                def set_batch_size(self, new_bs: int):
                    pass

            other_scheduler = ConstantBS(dataloader, factor=10, milestone=4, max_batch_size=100,
                                         batch_size_manager=OtherBatchManager())
            SequentialBS(schedulers=[other_scheduler, scheduler2, scheduler3], milestones=[5, 10])

    def test_loading_and_unloading(self):
        dataloader = create_dataloader(self.dataset)
        factor = 10
        scheduler1 = ConstantBS(dataloader, factor=factor, milestone=4)
        scheduler2 = ExponentialBS(dataloader, gamma=2, verbose=False)
        scheduler = SequentialBS(schedulers=[scheduler1, scheduler2], milestones=[5])

        self.reloading_scheduler(scheduler)
        self.torch_save_and_load(scheduler)
        scheduler.step()
        self.assertEqual(len(scheduler.schedulers), 2)
        self.assertEqual(scheduler.schedulers[0].factor, factor)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

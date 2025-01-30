import math
import os
import tempfile
import unittest

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

from bs_scheduler import BSScheduler


class BatchedDataset(Dataset):
    def __init__(self, data: Tensor, batch_size: int):
        self.data = data
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.data) // self.batch_size

    def __getitem__(self, i: int) -> Tensor:
        return self.data[i * self.batch_size: (i + 1) * self.batch_size]

    def get_batch_size(self) -> int:
        return self.batch_size

    def change_batch_size(self, batch_size: int):
        self.batch_size = batch_size


def batched_dataset(size: int = 5000, batch_size: int = 100):
    return BatchedDataset(torch.rand(size, 128), batch_size=batch_size)


def fashion_mnist():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )


def create_dataloader(dataset, num_workers=0, batch_size=64, drop_last=False):
    shuffle = True  # shuffle or sampler
    batch_sampler = None  # Sampler or None
    sampler = None
    # Collate fn is default_convert when batch size and batch sampler are not defined, else default_collate
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler,
                      num_workers=num_workers, drop_last=drop_last, sampler=sampler)


def iterate(dataloader):
    inferred_len = len(dataloader)
    real_len = 0
    for _ in dataloader:
        real_len += 1
    return real_len, inferred_len


def simulate_n_epochs(dataloader, scheduler, epochs):
    lengths = []
    if isinstance(epochs, (tuple, list)):
        for d in epochs:
            lengths.append(len(dataloader))
            scheduler.step(**d)
    else:
        for _ in range(epochs):
            lengths.append(len(dataloader))
            scheduler.step()
    return lengths


def get_batch_size(dataloader):
    data = next(iter(dataloader))
    if isinstance(data, torch.Tensor):
        return len(data)
    if isinstance(data, (list, tuple)):
        return len(data[0])
    if isinstance(data, dict):
        return len(next(iter(data.values())))
    raise TypeError(f"Unknown type {type(data).__name__}")


def get_batch_sizes_across_epochs(dataloader, scheduler, epochs):
    batch_sizes = []
    if isinstance(epochs, (tuple, list)):
        for d in epochs:
            batch_sizes.append(get_batch_size(dataloader))
            scheduler.step(**d)
    else:
        for _ in range(epochs):
            batch_sizes.append(get_batch_size(dataloader))
            scheduler.step()
    return batch_sizes


def rint(x: float) -> int:
    """ Rounds to the nearest int and returns the value as int.
    """
    return int(round(x))


def clip(x, min_x, max_x):
    return min(max(x, min_x), max_x)


class BSTest(unittest.TestCase):

    @staticmethod
    def compute_epoch_lengths(batch_sizes, dataset_len, drop_last):
        if drop_last:
            return [dataset_len // bs for bs in batch_sizes]
        return [int(math.ceil(dataset_len / bs)) for bs in batch_sizes]

    @staticmethod
    def reloading_scheduler(scheduler: BSScheduler):
        state_dict = scheduler.state_dict()
        scheduler.load_state_dict(state_dict)

    @staticmethod
    def torch_save_and_load(scheduler: BSScheduler):
        state_dict = scheduler.state_dict()
        # on os.name == 'nt' we can't open a named temporary file twice
        tmp = tempfile.NamedTemporaryFile(delete=False)
        try:
            torch.save(state_dict, tmp.name)
            state_dict = torch.load(tmp.name, weights_only=False)
        finally:
            tmp.close()
            os.unlink(tmp.name)

        scheduler.load_state_dict(state_dict)

    def create_scheduler(self, dataloader, scheduler_class, batch_size_manager, **kwargs):
        scheduler_class(dataloader, **kwargs)
        with self.assertRaises(AssertionError):
            scheduler_class(None, **kwargs)
        scheduler_class(None, batch_size_manager=batch_size_manager, **kwargs)
        scheduler_class(dataloader, batch_size_manager=batch_size_manager, **kwargs)

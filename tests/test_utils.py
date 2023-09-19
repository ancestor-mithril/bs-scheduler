from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


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
    inferred_len = len(dataloader)  # TODO: review name
    real_len = 0
    for _ in dataloader:
        real_len += 1
    return real_len, inferred_len


def simulate_n_epochs(dataloader, scheduler, epochs):
    lengths = []
    for _ in range(epochs):
        lengths.append(len(dataloader))
        scheduler.step()
    return lengths

def get_batch_size(dataloader):
    return len(next(iter(dataloader))[0])


def get_batch_sizes_across_epochs(dataloader, scheduler, epochs):
    batch_sizes = []
    for _ in range(epochs):
        batch_sizes.append(get_batch_size(dataloader))
        scheduler.step()
    return batch_sizes

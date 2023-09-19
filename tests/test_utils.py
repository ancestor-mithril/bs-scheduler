from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def create_dataloader(num_workers=0, batch_size=64):
    dataset = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    shuffle = True  # shuffle or sampler
    batch_sampler = None  # Sampler or None
    # Collate fn is default_convert when batch size and batch sampler are not defined, else default_collate
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, batch_sampler=batch_sampler,
                      num_workers=num_workers)


def dataloader_len(dataloader):
    inferred_len = len(dataloader)  # TODO: review name
    real_len = 0
    for _ in dataloader:
        real_len += 1
    return real_len, inferred_len


def step_n_epochs(dataloader, scheduler, epochs):
    lengths = []
    for _ in range(epochs):
        lengths.append(len(dataloader))
        scheduler.step()
    return lengths

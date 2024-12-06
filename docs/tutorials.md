## Basic usage

Integrating a Batch Size Scheduler inside a PyTorch training script is simple:

```python
from torch.utils.data import DataLoader
from bs_scheduler import StepBS
# We use StepBS in this example, but we can use any BS Scheduler

# Define the Dataset and the DataLoader
dataset = ...
dataloader = DataLoader(dataset, batch_size=16)
scheduler = StepBS(dataloader, step_size=30, gamma=2)
# Activates every 30 epochs and doubles the batch size.

for _ in range(100):
    train(...)
    validate(...)
    scheduler.step()

# We expect the batch size to have the following values:
# epoch 0 - 29: 16
# epoch 30 - 59: 32
# epoch 60 - 89: 64
# epoch 90 - 99: 128
```

Full example:

```python
import timm
import torch.cuda
import torchvision.datasets
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from bs_scheduler import StepBS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
train_loader = DataLoader(
    torchvision.datasets.CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transforms,
    ),
    batch_size=100,
)
val_loader = DataLoader(
    torchvision.datasets.CIFAR10(root="../data", train=False, transform=transforms),
    batch_size=500,
)
scheduler = StepBS(train_loader, step_size=10)

model = timm.create_model("hf_hub:grodino/resnet18_cifar10", pretrained=False).to(
    device
)
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.001)


def train():
    correct = 0
    total = 0

    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct / total


@torch.inference_mode()
def val():
    correct = 0
    total = 0

    model.eval()
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return correct / total


def main():
    for epoch in range(100):
        train_accuracy = train()
        val_accuracy = val()

        scheduler.step()

        print(train_accuracy, val_accuracy)


if __name__ == "__main__":
    main()
```

## Integrating bs-scheduler with already batched data

Using already batched data disables the automated batching features of the PyTorch DataLoaders.
In this case, a `BatchSizeManager` must be implemented by the end user to enable Batch Size Schedulers to change the batch size.
The `CustomBatchSizeManager` can be used when the dataset class implements the `get_batch_size` and `change_batch_size` 
methods, similar to the example below:

```python
from bs_scheduler import StepBS
from bs_scheduler.batch_size_schedulers import CustomBatchSizeManager
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader



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


dataset = BatchedDataset(torch.rand(10000, 128), batch_size=100)
dataloader = DataLoader(dataset, batch_size=None)
scheduler = StepBS(dataloader,
                   step_size=2,
                   gamma=2.0,
                   batch_size_manager=CustomBatchSizeManager(dataset),
                   max_batch_size=10000)


for epoch in range(10):
    for batched_data in dataloader:
        pass
    print(f"There are {len(dataloader)} batches in epoch {epoch}.")
    scheduler.step()
```
Output:
```
There are 100 batches in epoch 0.
There are 100 batches in epoch 1.
There are 50 batches in epoch 2.
There are 50 batches in epoch 3.
There are 25 batches in epoch 4.
There are 25 batches in epoch 5.
There are 12 batches in epoch 6.
There are 12 batches in epoch 7.
There are 6 batches in epoch 8.
There are 6 batches in epoch 9.
```

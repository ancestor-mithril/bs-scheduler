## Basic usage

Integrating a Batch Size Scheduler inside a PyTorch training script is simple:

```python
from torch.utils.data import DataLoader
from bs_scheduler import StepBS  # We use StepBS in this example, but we can use any BS Scheduler

# Define the Dataset and the DataLoader
dataset = ...
dataloader = DataLoader(..., batch_size=16)
scheduler = StepBS(dataloader, step_size=30, gamma=2)  # Activates every 30 epochs and doubles the batch size.

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

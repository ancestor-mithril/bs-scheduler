import os
from typing import Type

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, StepLR, MultiStepLR, ConstantLR, ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

from bs_scheduler import BSScheduler, LinearBS, StepBS, MultiStepBS, ConstantBS, ExponentialBS


def get_batch_size(dataloader):
    data = next(iter(dataloader))
    if isinstance(data, torch.Tensor):
        return len(data)
    if isinstance(data, (list, tuple)):
        return len(data[0])
    if isinstance(data, dict):
        return len(next(iter(data.values())))
    raise TypeError(f"Unknown type {type(data).__name__}")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_dataloader(batch_size):
    dataset = TensorDataset(torch.ones(10000, 1, dtype=torch.bool))
    return DataLoader(dataset, batch_size=batch_size)


def create_optimizer(lr: float):
    model = torch.nn.Linear(10, 10)
    return torch.optim.SGD(model.parameters(), lr=lr)


def simulate_n_epochs_bs(n, scheduler, dataloader: DataLoader):
    batch_sizes = []
    if isinstance(n, (tuple, list)):
        for metric in n:
            batch_sizes.append(get_batch_size(dataloader))
            scheduler.step(**metric)
    else:
        for _ in range(n):
            batch_sizes.append(get_batch_size(dataloader))
            scheduler.step()
    return batch_sizes


def simulate_n_epochs_lr(n: int, scheduler, optimizer: Optimizer):
    learning_rates = []
    for _ in range(n):
        learning_rates.append(get_lr(optimizer))
        scheduler.step()

    return learning_rates


def make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, title):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Batch Size", color="tab:blue")
    line1, = ax1.plot(range(epochs), batch_sizes, label=label_1, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color="tab:red")
    line2, = ax2.plot(range(epochs), learning_rates, label=label_2, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.title(f"{label_1} and {label_2}")
    plt.tight_layout()
    plt.savefig(f"plots/{title}.png", bbox_inches="tight")


def plot_step_1():
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    step_size = 4
    gamma = 2
    linear_bs = StepBS(dataloader, step_size=step_size, gamma=gamma)
    linear_lr = StepLR(optimizer, step_size=step_size, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"StepBS({step_size}, {gamma})",
               f"StepLR({step_size}, {1.0 / gamma:.2f})", "Step_1")


def plot_step_2():
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    step_size = 20
    gamma = 3
    linear_bs = StepBS(dataloader, step_size=step_size, gamma=gamma)
    linear_lr = StepLR(optimizer, step_size=step_size, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"StepBS({step_size}, {gamma})",
               f"StepLR({step_size}, {1.0 / gamma:.2f})", "Step_2")


def plot_multi_step_1():
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestones = [5, 10, 12, 13]
    gamma = 2
    linear_bs = MultiStepBS(dataloader, milestones=milestones, gamma=gamma)
    linear_lr = MultiStepLR(optimizer, milestones=milestones, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"MultiStepBS({milestones}, {gamma})",
               f"MultiStepLR({milestones}, {1.0 / gamma:.2f})", "MultiStep_1")


def plot_multi_step_2():
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestones = [15, 40, 40, 50, 80]
    gamma = 1.5
    linear_bs = MultiStepBS(dataloader, milestones=milestones, gamma=gamma)
    linear_lr = MultiStepLR(optimizer, milestones=milestones, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"MultiStepBS({milestones}, {gamma})",
               f"MultiStepLR({milestones}, {1.0 / gamma:.2f})", "MultiStep_2")


def plot_constant():
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestone = 5
    factor = 5.0
    linear_bs = ConstantBS(dataloader, factor=factor, milestone=milestone)
    linear_lr = ConstantLR(optimizer, factor=1.0 / factor, total_iters=milestone)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"Constant({factor}, {milestone})",
               f"MultiStepLR({1.0 / factor:.2f}, {milestone})", "Constant")


def plot_linear():
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    start_factor = 1.0
    end_factor = 10.0
    linear_bs = LinearBS(dataloader, start_factor=start_factor, end_factor=end_factor, milestone=epochs)
    linear_lr = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0 / end_factor, total_iters=epochs)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"LinearBS({start_factor}, {end_factor})",
               f"LinearLR({start_factor}, {1.0 / end_factor:.2f})", "Linear")


def plot_exponential():
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    gamma = 1.05
    linear_bs = ExponentialBS(dataloader, gamma=gamma)
    linear_lr = ExponentialLR(optimizer, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, linear_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, linear_lr, optimizer)

    make_plots(epochs, batch_sizes, learning_rates, f"ExponentialBS({gamma})",
               f"ExponentialLR({1.0 / gamma:.2f})", "Exponential")


def plot_polynomial():
    pass


def plot_cosine():
    pass


def plot_plateau():
    pass


def plot_cyclic():
    pass


def plot_cosine_warm_restarts():
    pass


def plot_one_cycle():
    pass


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs("plots", exist_ok=True)

    plot_step_1()
    plot_step_2()
    plot_multi_step_1()
    plot_multi_step_2()
    plot_constant()
    plot_linear()
    plot_exponential()
    plot_polynomial()
    plot_cosine()
    plot_plateau()
    plot_cyclic()
    plot_cosine_warm_restarts()
    plot_one_cycle()


if __name__ == '__main__':
    main()

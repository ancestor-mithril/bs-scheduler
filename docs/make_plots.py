import os
from typing import Type

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LinearLR, StepLR, MultiStepLR, ConstantLR, ExponentialLR, PolynomialLR, \
    CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

from bs_scheduler import BSScheduler, LinearBS, StepBS, MultiStepBS, ConstantBS, ExponentialBS, PolynomialBS, \
    CosineAnnealingBS, IncreaseBSOnPlateau, CyclicBS, CosineAnnealingBSWithWarmRestarts, OneCycleBS


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


def simulate_n_epochs_lr(n, scheduler, optimizer: Optimizer):
    learning_rates = []
    if isinstance(n, (tuple, list)):
        for metric in n:
            learning_rates.append(get_lr(optimizer))
            scheduler.step(**metric)
    else:
        for _ in range(n):
            learning_rates.append(get_lr(optimizer))
            scheduler.step()

    return learning_rates


def make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path):
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
    plt.savefig(path, bbox_inches="tight")


def plot_step_1(fd):
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    step_size = 4
    gamma = 2

    fd.write(f"""
```
scheduler_bs = StepBS(dataloader, step_size={step_size}, gamma={gamma})
scheduler_lr = StepLR(optimizer, step_size={step_size}, gamma=1.0 / {gamma})
```\n
""")
    scheduler_bs = StepBS(dataloader, step_size=step_size, gamma=gamma)
    scheduler_lr = StepLR(optimizer, step_size=step_size, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Step_1.png"
    label_1 = f"StepBS({step_size}, {gamma})"
    label_2 = f"StepLR({step_size}, {1.0 / gamma:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_step_2(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    step_size = 20
    gamma = 3

    fd.write(f"""
```
scheduler_bs = StepBS(dataloader, step_size={step_size}, gamma={gamma})
scheduler_lr = StepLR(optimizer, step_size={step_size}, gamma=1.0 / {gamma})
```\n
""")
    scheduler_bs = StepBS(dataloader, step_size=step_size, gamma=gamma)
    scheduler_lr = StepLR(optimizer, step_size=step_size, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Step_2.png"
    label_1 = f"StepBS({step_size}, {gamma})"
    label_2 = f"StepLR({step_size}, {1.0 / gamma:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_multi_step_1(fd):
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestones = [5, 10, 12, 13]
    gamma = 2

    fd.write(f"""
```
scheduler_bs = MultiStepBS(dataloader, milestones={milestones}, gamma={gamma})
scheduler_lr = MultiStepLR(optimizer, milestones={milestones}, gamma=1.0 / {gamma})
```\n
""")
    scheduler_bs = MultiStepBS(dataloader, milestones=milestones, gamma=gamma)
    scheduler_lr = MultiStepLR(optimizer, milestones=milestones, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/MultiStep_1.png"
    label_1 = f"MultiStepBS({milestones}, {gamma})"
    label_2 = f"MultiStepLR({milestones}, {1.0 / gamma:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_multi_step_2(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestones = [15, 40, 40, 50, 80]
    gamma = 1.5

    fd.write(f"""
```
scheduler_bs = MultiStepBS(dataloader, milestones={milestones}, gamma={gamma})
scheduler_lr = MultiStepLR(optimizer, milestones={milestones}, gamma=1.0 / {gamma})
```\n
""")
    scheduler_bs = MultiStepBS(dataloader, milestones=milestones, gamma=gamma)
    scheduler_lr = MultiStepLR(optimizer, milestones=milestones, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/MultiStep_2.png"
    label_1 = f"MultiStepBS({milestones}, {gamma})"
    label_2 = f"MultiStepLR({milestones}, {1.0 / gamma:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_constant(fd):
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    milestone = 5
    factor = 5.0

    fd.write(f"""
```
scheduler_bs = ConstantBS(dataloader, factor={factor}, milestone={milestone})
scheduler_lr = ConstantLR(optimizer, factor=1.0 / {factor}, total_iters={milestone})
```\n
""")
    scheduler_bs = ConstantBS(dataloader, factor=factor, milestone=milestone)
    scheduler_lr = ConstantLR(optimizer, factor=1.0 / factor, total_iters=milestone)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Constant.png"
    label_1 = f"Constant({factor}, {milestone})"
    label_2 = f"MultiStepLR({1.0 / factor:.2f}, {milestone})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_linear(fd):
    epochs = 20
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    start_factor = 1.0
    end_factor = 10.0

    fd.write(f"""
```
scheduler_bs = LinearBS(dataloader, start_factor={start_factor}, end_factor={end_factor}, milestone={epochs})
scheduler_lr = LinearLR(optimizer, start_factor={start_factor}, end_factor=1.0 / {end_factor}, total_iters={epochs})
```\n
""")
    scheduler_bs = LinearBS(dataloader, start_factor=start_factor, end_factor=end_factor, milestone=epochs)
    scheduler_lr = LinearLR(optimizer, start_factor=start_factor, end_factor=1.0 / end_factor, total_iters=epochs)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Linear.png"
    label_1 = f"LinearBS({start_factor}, {end_factor})"
    label_2 = f"LinearLR({start_factor}, {1.0 / end_factor:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_exponential(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    gamma = 1.05

    fd.write(f"""
```
scheduler_bs = ExponentialBS(dataloader, gamma={gamma})
scheduler_lr = ExponentialLR(optimizer, gamma=1.0 / {gamma})
```\n
""")
    scheduler_bs = ExponentialBS(dataloader, gamma=gamma)
    scheduler_lr = ExponentialLR(optimizer, gamma=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Exponential.png"
    label_1 = f"ExponentialBS({gamma})"
    label_2 = f"ExponentialLR({1.0 / gamma:.2f})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_polynomial_1(fd):
    epochs = 50
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    power = 1.5

    fd.write(f"""
```
scheduler_bs = PolynomialBS(dataloader, total_iters={epochs}, power={power})
scheduler_lr = PolynomialLR(optimizer, total_iters={epochs}, power={power})
```\n
""")
    scheduler_bs = PolynomialBS(dataloader, total_iters=epochs, power=power)
    scheduler_lr = PolynomialLR(optimizer, total_iters=epochs, power=power)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Polynomial_1.png"
    label_1 = f"PolynomialBS({power})"
    label_2 = f"PolynomialLR({power})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_polynomial_2(fd):
    epochs = 50
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    power = 2

    fd.write(f"""
```
scheduler_bs = PolynomialBS(dataloader, total_iters={epochs}, power={power})
scheduler_lr = PolynomialLR(optimizer, total_iters={epochs}, power={power})
```\n
""")
    scheduler_bs = PolynomialBS(dataloader, total_iters=epochs, power=power)
    scheduler_lr = PolynomialLR(optimizer, total_iters=epochs, power=power)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Polynomial_2.png"
    label_1 = f"PolynomialBS({power})"
    label_2 = f"PolynomialLR({power})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cosine(fd):
    epochs = 50
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    total_iters = 10

    fd.write(f"""
```
scheduler_bs = CosineAnnealingBS(dataloader, total_iters={total_iters})
scheduler_lr = CosineAnnealingLR(optimizer, T_max={total_iters})
```\n
""")
    scheduler_bs = CosineAnnealingBS(dataloader, total_iters=total_iters)
    scheduler_lr = CosineAnnealingLR(optimizer, T_max=total_iters)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/CosineAnnealing.png"
    label_1 = f"CosineAnnealingBS({total_iters})"
    label_2 = f"CosineAnnealingLR({total_iters})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_plateau(fd):
    epochs = [{"metrics": 0.1}] * 50
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    gamma = 2.0

    fd.write(f"""
```
scheduler_bs = IncreaseBSOnPlateau(dataloader, factor={gamma})
scheduler_lr = ReduceLROnPlateau(optimizer, factor=1.0 / {gamma})
```\n
""")
    scheduler_bs = IncreaseBSOnPlateau(dataloader, factor=gamma)
    scheduler_lr = ReduceLROnPlateau(optimizer, factor=1.0 / gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Plateau.png"
    label_1 = f"IncreaseBSOnPlateau({gamma})"
    label_2 = f"ReduceLROnPlateau({1.0 / gamma:.2f})"

    make_plots(len(epochs), batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cyclic_1(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    base_batch_size = 64
    base_lr = 0.001
    max_lr = 0.01
    min_batch_size = 16
    mode = "triangular"
    step_size_down = 25

    fd.write(f"""
```
scheduler_bs = CyclicBS(dataloader, base_batch_size={base_batch_size}, min_batch_size={min_batch_size}, mode={mode}, step_size_down={step_size_down})
scheduler_lr = CyclicLR(optimizer, base_lr={base_lr}, max_lr={max_lr}, mode={mode}, step_size_up={step_size_down})
```\n
""")
    scheduler_bs = CyclicBS(dataloader, base_batch_size=base_batch_size, min_batch_size=min_batch_size, mode=mode,
                            step_size_down=step_size_down)
    scheduler_lr = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, step_size_up=step_size_down)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Cyclic_1.png"
    label_1 = f"CyclicBS({mode})"
    label_2 = f"CyclicLR({mode})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cyclic_2(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    base_batch_size = 64
    base_lr = 0.001
    max_lr = 0.01
    min_batch_size = 16
    mode = "triangular2"
    step_size_down = 25

    fd.write(f"""
```
scheduler_bs = CyclicBS(dataloader, base_batch_size={base_batch_size}, min_batch_size={min_batch_size}, mode={mode}, step_size_down={step_size_down})
scheduler_lr = CyclicLR(optimizer, base_lr={base_lr}, max_lr={max_lr}, mode={mode}, step_size_up={step_size_down})
```\n
""")
    scheduler_bs = CyclicBS(dataloader, base_batch_size=base_batch_size, min_batch_size=min_batch_size, mode=mode,
                            step_size_down=step_size_down)
    scheduler_lr = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, step_size_up=step_size_down)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Cyclic_2.png"
    label_1 = f"CyclicBS({mode})"
    label_2 = f"CyclicLR({mode})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cyclic_3(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    base_batch_size = 64
    base_lr = 0.001
    max_lr = 0.01
    min_batch_size = 16
    mode = "exp_range"
    gamma = 0.8
    step_size_down = 25

    fd.write(f"""
```
scheduler_bs = CyclicBS(dataloader, base_batch_size={base_batch_size}, min_batch_size={min_batch_size}, mode={mode}, step_size_down={step_size_down}, gamma={gamma})
scheduler_lr = CyclicLR(optimizer, base_lr={base_lr}, max_lr={max_lr}, mode={mode}, step_size_up={step_size_down}, gamma={gamma})
```\n
""")
    scheduler_bs = CyclicBS(dataloader, base_batch_size=base_batch_size, min_batch_size=min_batch_size, mode=mode,
                            step_size_down=step_size_down, gamma=gamma)
    scheduler_lr = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, mode=mode, step_size_up=step_size_down,
                            gamma=gamma)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/Cyclic_3.png"
    label_1 = f"CyclicBS({mode}, {gamma})"
    label_2 = f"CyclicLR({mode}, {gamma})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cosine_warm_restarts_1(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    t_0 = 40
    factor = 1

    fd.write(f"""
```
scheduler_bs = CosineAnnealingBSWithWarmRestarts(dataloader, t_0={t_0}, factor={factor})
scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0={t_0}, T_mult={factor})
```\n
""")
    scheduler_bs = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, factor=factor)
    scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=factor)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/CosineAnnnealingWarmRestarts_1.png"
    label_1 = f"CosineAnnealingBSWithWarmRestarts({t_0}, {factor})"
    label_2 = f"CosineAnnealingWarmRestarts({t_0}, {factor})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_cosine_warm_restarts_2(fd):
    epochs = 100
    dataloader = create_dataloader(16)
    optimizer = create_optimizer(0.01)
    t_0 = 30
    factor = 2

    fd.write(f"""
```
scheduler_bs = CosineAnnealingBSWithWarmRestarts(dataloader, t_0={t_0}, factor={factor})
scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0={t_0}, T_mult={factor})
```\n
""")
    scheduler_bs = CosineAnnealingBSWithWarmRestarts(dataloader, t_0=t_0, factor=factor)
    scheduler_lr = CosineAnnealingWarmRestarts(optimizer, T_0=t_0, T_mult=factor)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/CosineAnnnealingWarmRestarts_2.png"
    label_1 = f"CosineAnnealingBSWithWarmRestarts({t_0}, {factor})"
    label_2 = f"CosineAnnealingWarmRestarts({t_0}, {factor})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_one_cycle_1(fd):
    epochs = 100
    dataloader = create_dataloader(32)
    optimizer = create_optimizer(0.01)
    min_batch_size = 16
    max_batch_size = 64
    max_lr = 0.1
    div_factor = 2
    final_div_factor = 2
    total_steps = epochs
    decay_percentage = 0.3
    strategy = 'cos'

    fd.write(f"""
```
scheduler_bs = OneCycleBS(dataloader, total_steps={total_steps}, decay_percentage={decay_percentage}, strategy={strategy}, max_batch_size={max_batch_size}, min_batch_size={min_batch_size})
scheduler_lr = OneCycleLR(optimizer, total_steps={total_steps}, pct_start={decay_percentage}, anneal_strategy={strategy}, max_lr={max_lr}, div_factor={div_factor}, final_div_factor={final_div_factor})
```\n
""")
    scheduler_bs = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage, strategy=strategy,
                              max_batch_size=max_batch_size, min_batch_size=min_batch_size)
    scheduler_lr = OneCycleLR(optimizer, total_steps=total_steps, pct_start=decay_percentage, anneal_strategy=strategy,
                              max_lr=max_lr, div_factor=div_factor, final_div_factor=final_div_factor)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/OneCycle_1.png"
    label_1 = f"OneCycleBS({strategy})"
    label_2 = f"OneCycleLR({strategy})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def plot_one_cycle_2(fd):
    epochs = 100
    dataloader = create_dataloader(32)
    optimizer = create_optimizer(0.01)
    min_batch_size = 16
    max_batch_size = 64
    max_lr = 0.1
    div_factor = 10
    final_div_factor = 100
    total_steps = epochs
    decay_percentage = 0.3
    strategy = 'linear'

    fd.write(f"""
```
scheduler_bs = OneCycleBS(dataloader, total_steps={total_steps}, decay_percentage={decay_percentage}, strategy={strategy}, max_batch_size={max_batch_size}, min_batch_size={min_batch_size})
scheduler_lr = OneCycleLR(optimizer, total_steps={total_steps}, pct_start={decay_percentage}, anneal_strategy={strategy}, max_lr={max_lr}, div_factor={div_factor}, final_div_factor={final_div_factor})
```\n
""")
    scheduler_bs = OneCycleBS(dataloader, total_steps=total_steps, decay_percentage=decay_percentage, strategy=strategy,
                              max_batch_size=max_batch_size, min_batch_size=min_batch_size)
    scheduler_lr = OneCycleLR(optimizer, total_steps=total_steps, pct_start=decay_percentage, anneal_strategy=strategy,
                              max_lr=max_lr, div_factor=div_factor, final_div_factor=final_div_factor)

    batch_sizes = simulate_n_epochs_bs(epochs, scheduler_bs, dataloader)
    learning_rates = simulate_n_epochs_lr(epochs, scheduler_lr, optimizer)

    path = "./plots/OneCycle_2.png"
    label_1 = f"OneCycleBS({strategy})"
    label_2 = f"OneCycleLR({strategy})"

    make_plots(epochs, batch_sizes, learning_rates, label_1, label_2, path)
    fd.write(f"![{label_1} vs {label_2}]({path})\n")


def main():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    os.makedirs("plots", exist_ok=True)
    with open("plots.md", "w") as fd:
        fd.write("## Batch size policies vs Learning rate policies\n")

        fd.write("### Step\n")
        plot_step_1(fd)
        plot_step_2(fd)

        fd.write("### MultiStep\n")
        plot_multi_step_1(fd)
        plot_multi_step_2(fd)

        fd.write("### Constant\n")
        plot_constant(fd)

        fd.write("### Linear\n")
        plot_linear(fd)

        fd.write("### Exponential\n")
        plot_exponential(fd)

        fd.write("### Polynomial\n")
        plot_polynomial_1(fd)
        plot_polynomial_2(fd)

        fd.write("### CosineAnnealing\n")
        plot_cosine(fd)

        fd.write("### Plateau\n")
        plot_plateau(fd)

        fd.write("### Cyclic\n")
        plot_cyclic_1(fd)
        plot_cyclic_2(fd)
        plot_cyclic_3(fd)

        fd.write("### CosineAnnealingWarmRestarts\n")
        plot_cosine_warm_restarts_1(fd)
        plot_cosine_warm_restarts_2(fd)

        fd.write("### OneCycle\n")
        plot_one_cycle_1(fd)
        plot_one_cycle_2(fd)


if __name__ == '__main__':
    main()

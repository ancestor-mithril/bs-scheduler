# bs-scheduler

A Batch Size Scheduler library compatible with PyTorch DataLoaders.

*** 

## Documentation

* [API Reference](https://ancestor-mithril.github.io/bs-scheduler).

* [Examples](https://ancestor-mithril.github.io/bs-scheduler/tutorials).

<!--For Release Notes, see TODO. -->

***

## Why use a Batch Size Scheduler?

* Using a big batch size has several advantages:
  * Better hardware utilization.
  * Enhanced parallelism.
  * Faster training.
* However, using a big batch size from the start may lead to a generalization gap.
* Therefore, the solution is to gradually increase the batch size, similar to a learning rate decay policy.
* See [Don't Decay the Learning Rate, Increase the Batch Size](https://arxiv.org/abs/1711.00489).


## Available Schedulers

### Batch Size Schedulers

1. `LambdaBS` - sets the batch size to the base batch size times a given lambda.
2. `MultiplicativeBS` - sets the batch size to the current batch size times a given lambda.
3. `StepBS` - multiplies the batch size with a given factor at a given number of steps.
4. `MultiStepBS` - multiplies the batch size with a given factor each time a milestone is reached.
5. `ConstantBS` - multiplies the batch size by a given factor once and decreases it again to its base value after a
   given number of steps.
6. `LinearBS` - increases the batch size by a linearly changing multiplicative factor for a given number of steps.
7. `ExponentialBS` - increases the batch size by a given $\gamma$ each step.
8. `PolynomialBS` - increases the batch size using a polynomial function in a given number of steps.
9. `CosineAnnealingBS` - increases the batch size to a maximum batch size and decreases it again following a cyclic
   cosine curve.
10. `IncreaseBSOnPlateau` - increases the batch size each time a given metric has stopped improving for a given number
    of steps.
11. `CyclicBS` - cycles the batch size between two boundaries with a constant frequency, while also scaling the
    distance between boundaries.
12. `CosineAnnealingBSWithWarmRestarts` - increases the batch size to a maximum batch size following a cosine curve,
    then restarts while also scaling the number of iterations until the next restart.
13. `OneCycleBS` - decreases the batch size to a minimum batch size then increases it to a given maximum batch size,
    following a linear or cosine annealing strategy.
14. `SequentialBS` - calls a list of schedulers sequentially given a list of milestone points which reflect which
    scheduler should be called when.
15. `ChainedBSScheduler` - chains a list of batch size schedulers and calls them together each step.

<!--

## Quick Start

TODO.

-->

## Installation

Please install [PyTorch](https://github.com/pytorch/pytorch) first before installing this repository.

```
pip install bs-scheduler
```

## Licensing

The library is licensed under the [BSD-3-Clause license](LICENSE).

## Citation

To be added...

<!--Citation: TODO. -->

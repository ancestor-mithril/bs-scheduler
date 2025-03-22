# bs-scheduler

A Batch Size Scheduler library compatible with PyTorch DataLoaders.

***

### Batch Size Schedulers

1. [LambdaBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.LambdaBS) - sets the batch size to the base batch size times a given lambda.
2. [MultiplicativeBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.MultiplicativeBS) - sets the batch size to the current batch size times a given lambda.
3. [StepBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.StepBS) - multiplies the batch size with a given factor at a given number of steps.
4. [MultiStepBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.MultiStepBS) - multiplies the batch size with a given factor each time a milestone is reached.
5. [ConstantBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.ConstantBS) - multiplies the batch size by a given factor once and decreases it again to its base value after a
   given number of steps.
6. [LinearBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.LinearBS) - increases the batch size by a linearly changing multiplicative factor for a given number of steps.
7. [ExponentialBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.ExponentialBS) - increases the batch size by a given $\gamma$ each step.
8. [PolynomialBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.PolynomialBS) - increases the batch size using a polynomial function in a given number of steps.
9. [CosineAnnealingBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.CosineAnnealingBS) - increases the batch size to a maximum batch size and decreases it again following a cyclic
   cosine curve.
10. [IncreaseBSOnPlateau](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.IncreaseBSOnPlateau) - increases the batch size each time a given metric has stopped improving for a given number
    of steps.
11. [CyclicBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.CyclicBS) - cycles the batch size between two boundaries with a constant frequency, while also scaling the
    distance between boundaries.
12. [CosineAnnealingBSWithWarmRestarts](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.CosineAnnealingBSWithWarmRestarts) - increases the batch size to a maximum batch size following a cosine curve,
    then restarts while also scaling the number of iterations until the next restart.
13. [OneCycleBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.OneCycleBS) - decreases the batch size to a minimum batch size then increases it to a given maximum batch size,
    following a linear or cosine annealing strategy.
14. [SequentialBS](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.SequentialBS) - calls a list of schedulers sequentially given a list of milestone points which reflect which
    scheduler should be called when.
15. [ChainedBSScheduler](https://ancestor-mithril.github.io/bs-scheduler/reference/#bs_scheduler.ChainedBSScheduler) - chains a list of batch size schedulers and calls them together each step.

Check the [plots](https://ancestor-mithril.github.io/bs-scheduler/plots) for a visual comparison between Batch Size Schedulers and Learning Rate Schedulers.

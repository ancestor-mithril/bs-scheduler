# Inspired from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html.
import inspect
import math
import types
from bisect import bisect_right
from collections import Counter
from functools import partial
from typing import Callable, Sequence, Tuple, Optional

import torch
from torch.utils.data import DataLoader

__all__ = ['LambdaBS', 'MultiplicativeBS', 'StepBS', 'MultiStepBS', 'ConstantBS', 'LinearBS', 'ExponentialBS',
           'SequentialBS', 'PolynomialBS', 'CosineAnnealingBS', 'ChainedBSScheduler', 'IncreaseBSOnPlateau', 'CyclicBS',
           'CosineAnnealingBSWithWarmRestarts', 'OneCycleBS', 'BSScheduler']

from .batch_size_manager import BatchSizeManager, DefaultBatchSizeManager, CustomBatchSizeManager
from .utils import check_isinstance, clip, rint


class BSScheduler:
    def __init__(self, dataloader: Optional[DataLoader], batch_size_manager: Optional[BatchSizeManager],
                 max_batch_size: Optional[int], min_batch_size: int, verbose: bool):
        dataloader is None or check_isinstance(dataloader, DataLoader)
        self.dataloader: Optional[DataLoader] = dataloader
        self.verbose: bool = verbose

        assert max_batch_size is None or isinstance(max_batch_size, int)
        assert isinstance(min_batch_size, int)
        if max_batch_size is None:
            if dataloader is not None and hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, '__len__'):
                max_batch_size = len(dataloader.dataset) + 1
            else:
                max_batch_size = 1
        else:
            if max_batch_size < 0:
                raise ValueError(f"Maximum batch size must be greater than 0, but is {max_batch_size}.")
        self.max_batch_size: int = max_batch_size

        if min_batch_size < 0:
            raise ValueError(f"Minimum batch size must be greater than 0, but is {min_batch_size}.")
        if min_batch_size > self.max_batch_size:
            raise ValueError(f"Minimum batch size must be smaller than or equal to the maximum batch size "
                             f"({max_batch_size}), but is {min_batch_size}.")
        self.min_batch_size: int = min_batch_size

        if batch_size_manager is None:
            assert self.dataloader is not None, "batch_size_manager must be provided if dataloader is None"
            if self.dataloader.batch_sampler is not None:
                batch_size_manager = DefaultBatchSizeManager(self.dataloader)
            else:
                # We require the client to implement a "change_batch_size" method and a "get_batch_size" method for
                # their dataset.
                batch_size_manager = CustomBatchSizeManager(self.dataloader.dataset)
        else:
            check_isinstance(batch_size_manager, BatchSizeManager)
        self.batch_size_manager: BatchSizeManager = batch_size_manager

        self.last_epoch: int = -1
        if not hasattr(self.batch_size_manager, '_base_batch_size'):
            self.batch_size_manager._base_batch_size = self.batch_size
        self._last_bs: int = self.batch_size_manager._base_batch_size
        self._finished: bool = False

        self._init_get_new_bs()

        # Doing the zero-th step.
        self.step()
        # The initial step may make the scheduler to finish during initialization. So we reinitialize self._finished.
        self._finished = False

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader.
        """
        return {key: value for key, value in self.__dict__.items() if
                key not in ('dataloader', '_internal_get_new_bs')}

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self.set_batch_size(self.last_bs)  # Setting the batch size to the last computed batch size.
        self._init_get_new_bs()

    def set_batch_size(self, new_bs: int):
        """ Forwards the call for setting the new batch size to the batch size manager. If the dataloader batch_size
        member variable is not None, it also modifies it to reflect the change in batch size.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        if self.dataloader is not None and self.dataloader.batch_size is not None:
            # We can't directly do `dataloader.batch_size = new_bs` because the dataloader raises an error if we change
            # the batch size after initialization. But we are still hacking around it.
            self.dataloader.__dict__['batch_size'] = new_bs
        self.batch_size_manager.set_batch_size(new_bs)

    @property
    def batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`. """
        return self.batch_size_manager.get_current_batch_size()

    @property
    def finished(self) -> bool:
        """ Returns True if the scheduler has already finished its job or has exceeded the minimum or maximum batch
        size. Otherwise, returns False.
        """
        return self._finished

    @property
    def last_bs(self) -> int:
        """ Returns the last computed batch size by current scheduler. If called before the first call to :meth:`step`
        returns the base batch size.
        """
        return self._last_bs

    def get_new_bs(self) -> int:
        """ Computes the next batch size. Should not be called explicitly in client code, but it doesn't really matter
        if the client does so. Some batch size schedulers use the keyword arguments.
        """
        raise NotImplementedError

    def _init_get_new_bs(self):
        # Setting the correct get_new_bs() dispatch function.
        if inspect.getfullargspec(self.get_new_bs).varkw is None:
            self._internal_get_new_bs = self._internal_bare_dispatch
        else:
            self._internal_get_new_bs = self._internal_kwargs_dispatch

    def _internal_bare_dispatch(self, **kwargs) -> int:
        return self.get_new_bs()

    def _internal_kwargs_dispatch(self, **kwargs) -> int:
        return self.get_new_bs(**kwargs)

    def print_bs(self, new_bs):
        if self.verbose:
            print(f'Adjusting batch size to {new_bs}.')

    def step(self, **kwargs):
        # Changing the batch size does not impact batch sizes loaded by workers before the change.
        if self.finished:
            return  # Stops doing work if already finished.

        self.last_epoch += 1
        new_bs = self._internal_get_new_bs(**kwargs)
        if not self.min_batch_size <= new_bs <= self.max_batch_size:
            self._finished = True
            new_bs = clip(new_bs, min_x=self.min_batch_size, max_x=self.max_batch_size)
        if new_bs != self.batch_size:
            self.set_batch_size(new_bs)
            self.print_bs(new_bs)
        self._last_bs = new_bs


class LambdaBS(BSScheduler):
    """ Sets the batch size to the initial batch size times a given function. Unlike torch.optim.lr_scheduler.LambdaLR,
    there is a single batch size for a given dataloader so only one function should be passed as a parameter.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        bs_lambda (Callable[[int], float]): A function which computes a multiplicative factor given an integer
            parameter epoch.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> func = lambda epoch: 1.05 ** epoch
        >>> scheduler = LambdaBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], bs_lambda: Callable[[int], float],
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert callable(bs_lambda)
        self.bs_lambda: Callable[[int], float] = bs_lambda
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The batch size lambda
        function will only be saved if they are callable objects and not if they are functions or lambdas.
        """
        state_dict = super().state_dict()
        state_dict['bs_lambda'] = None
        if not isinstance(self.bs_lambda, types.FunctionType):
            state_dict['bs_lambda'] = self.bs_lambda.__dict__.copy()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        bs_lambda = state_dict.pop('bs_lambda')
        super().load_state_dict(state_dict)
        if bs_lambda is not None:
            self.bs_lambda.__dict__.update(bs_lambda)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. It is calculated as the initial value of the batch size
        times the factor returned by `bs_lambda`.
        """
        return rint(self.batch_size_manager._base_batch_size * self.bs_lambda(self.last_epoch))


class MultiplicativeBS(BSScheduler):
    """ Multiply the batch size by a factor given in the specified function. Unlike
    torch.optim.lr_scheduler.MultiplicativeLR, there is a single batch size for a given dataloader so only one function
    should be passed as a parameter.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        bs_lambda (Callable[[int], float]): A function which computes a multiplicative factor given an integer
            parameter epoch.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> func = lambda epoch: 1.05
        >>> scheduler = MultiplicativeBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], bs_lambda: Callable[[int], float],
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert callable(bs_lambda)
        self.bs_lambda: Callable[[int], float] = bs_lambda
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The batch size lambda
        function will only be saved if they are callable objects and not if they are functions or lambdas.
        """
        state_dict = super().state_dict()
        state_dict['bs_lambda'] = None
        if not isinstance(self.bs_lambda, types.FunctionType):
            state_dict['bs_lambda'] = self.bs_lambda.__dict__.copy()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        bs_lambda = state_dict.pop('bs_lambda')
        super().load_state_dict(state_dict)
        if bs_lambda is not None:
            self.bs_lambda.__dict__.update(bs_lambda)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. It is calculated as the current value of the batch size
        times the factor returned by `bs_lambda`.
        """
        return rint(self.batch_size * self.bs_lambda(self.last_epoch))


class StepBS(BSScheduler):
    """ Multiplies the batch size by gamma every step_size epochs.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        step_size (int): Period of batch size growth.
        gamma (float): Multiplicative factor of batch size growth. Default: 2.0.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if epoch < 30
        >>> # bs = 20 if 30 <= epoch < 60
        >>> # bs = 40 if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepBS(dataloader, step_size=30, gamma=2.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], step_size: int, gamma: float = 2.0,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(step_size, int) and step_size > 0
        assert gamma > 0.0
        # Gamma is expected to be greater than 1, but we do not forbid batch size decay.
        self.step_size: int = step_size
        self.gamma: float = gamma
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. It returns the current batch size times gamma each
        step_size epochs, otherwise it returns the current batch size.
        """
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.batch_size
        return rint(self.batch_size * self.gamma)


class MultiStepBS(BSScheduler):
    """ Multiplies the batch size by gamma once the number of epochs reaches one of the milestones.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        milestones (Sequence[int]): Sequence of epoch indices.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if epoch < 30
        >>> # bs = 20 if 25 <= epoch < 80
        >>> # bs = 40 if 80 <= epoch
        >>> scheduler = MultiStepBS(dataloader, milestones=[25, 80], gamma=2.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], milestones: Sequence[int], gamma: float = 2.0,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(milestones, (tuple, list))
        assert len(milestones) > 0 and all([x > 0 and isinstance(x, int) for x in milestones])
        assert gamma > 0.0
        # Gamma is expected to be greater than 1, but we do not forbid batch size decay.
        # We do not require milestones to be sorted. However, sorted looks better.
        self.milestones: Counter[int, int] = Counter(milestones)
        self.gamma: float = gamma
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    @property
    def finished(self) -> bool:
        """ Returns True if the scheduler has already finished its job or has exceeded the minimum or maximum batch
        size. Otherwise, returns False.
        """
        if not self._finished:
            # Should we cache max(self.milestones)?
            self._finished = self.last_epoch > max(self.milestones)
        return self._finished

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. It returns the current batch size times gamma each epoch a
        milestone is reached, otherwise it returns the current batch size. Beware that in the event of multiple
        milestones with the same value, the current batch size is multiplied with gamma multiple times.
        """
        if self.last_epoch not in self.milestones:
            return self.batch_size
        return rint(self.batch_size * self.gamma ** self.milestones[self.last_epoch])


class ConstantBS(BSScheduler):
    """ Increases the batch size by a constant multiplicative factor until the number of epochs reaches a pre-defined
    milestone. The batch size is multiplied by the constant factor during initialization and is multiplied again with
    the inverse of the constant factor when the milestone is reached.
    If the constant factor makes the batch size increase the image out of bounds, the constant factor is changed
    automatically such that the batch size remains within bounds.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        factor (float): The number we multiply the batch size until the milestone.
        milestone (int): The number of steps that the scheduler increases the learning rate. Default: 5.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 50 if epoch == 0
        >>> # bs = 50 if epoch == 1
        >>> # bs = 50 if epoch == 2
        >>> # bs = 10 if epoch >= 3
        >>> scheduler = ConstantBS(dataloader, factor=5, milestone=3)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], factor: float, milestone: int = 5,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(milestone, int) and milestone > 0
        assert factor > 0.0
        # Factor is expected to be greater than 1.0, as this should be a warmup process.
        self.factor: float = factor
        self.milestone: int = milestone
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. The value of the batch size is changed once at
        initialization, when the batch size is multiplied with the given factor, and twice when the milestone is
        reached and the batch size is multiplied with the inverse of the given factor. The factor is adjusted during
        initialization such that it does not return a batch size out of bounds.
        """
        if self.last_epoch == 0:
            max_factor = self.max_batch_size / self.batch_size
            min_factor = self.min_batch_size / self.batch_size
            if self.factor > max_factor:
                self.factor = max_factor
            elif self.factor < min_factor:
                self.factor = min_factor
            return rint(self.batch_size * self.factor)

        if self.last_epoch != self.milestone:
            return self.batch_size

        self._finished = True  # My job is done.
        return rint(self.batch_size * (1.0 / self.factor))


class LinearBS(BSScheduler):
    """ Increases the batch size by a linearly changing small multiplicative factor until the number of epochs reaches
    a pre-defined milestone.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        start_factor (float): The number we multiply the batch size in the first epoch. The multiplication factor
            changes towards end_factor in the following epochs. Default: 3.0.
        end_factor (float): The number we multiply the batch size at the end of the linear changing process.
                Default: 1.0.
        milestone (int): The number of steps that the scheduler increases the learning rate. Default: 5.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 60 if epoch == 0
        >>> # bs = 50 if epoch == 1
        >>> # bs = 40 if epoch == 2
        >>> # bs = 30 if epoch == 3
        >>> # bs = 20 if epoch == 4
        >>> # bs = 10 if epoch >= 5
        >>> scheduler = LinearBS(dataloader, start_factor=6.0, end_factor=1.0, milestone=5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], start_factor: float = 3.0, end_factor: float = 1.0,
                 milestone: int = 5,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(milestone, int) and milestone > 0
        assert start_factor > 0.0 and end_factor > 0.0
        # Both start_factor and end_factor are expected to be greater than 1.0, with start_factor > end_factor, as this
        # should be a warmup process. But we do not forbid any other sound combinations.
        self.start_factor: float = start_factor
        self.end_factor: float = end_factor
        self.milestone: int = milestone
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. The current batch size is multiplied by the linear changing
        factor, starting from start_factor to end_factor. After the milestone is reached, the batch size is not changed
        anymore.
        """
        if self.last_epoch > self.milestone:
            self._finished = True  # My job is done.
            return self.batch_size

        if self.last_epoch == 0:
            return rint(self.batch_size * self.start_factor)

        value_range = self.end_factor - self.start_factor
        return rint(self.batch_size * (
                1.0 + value_range / (self.milestone * self.start_factor + (self.last_epoch - 1) * value_range)))


class ExponentialBS(BSScheduler):
    """ Increases the batch size by a gamma every epoch.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        gamma (float): Multiplicative factor of batch size growth.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if epoch == 0
        >>> # bs = 11 if epoch == 1
        >>> # bs = 12 if epoch == 2
        >>> # bs = 13 if epoch == 3
        >>> # ...
        >>> scheduler = ExponentialBS(dataloader, gamma=1.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], gamma: float,
                 batch_size_manager: Optional[BatchSizeManager] = None,
                 max_batch_size: Optional[int] = None, min_batch_size: int = 1, verbose: bool = False):
        assert gamma > 0.0
        # Gamma is expected to be greater than 1.0 for batch size growth. It can be lower than 1.0 for batch size decay.
        self.gamma: float = gamma
        self.float_bs: Optional[float] = None
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. The current batch size is multiplied by gamma each epoch
        except the first one.
        """
        if self.last_epoch == 0:
            return self.batch_size

        if self.float_bs is None or rint(self.float_bs) != self.batch_size:
            # Using rint instead of int because otherwise we will increas the BS faster
            self.float_bs = self.batch_size

        self.float_bs *= self.gamma
        return rint(self.float_bs)


class SequentialBS(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.SequentialLR. Receives a sequence of schedulers and calls them sequentially
    given the milestone points that reflect which scheduler is supposed to be called at a fiven epoch

    Args:
        schedulers (Sequence[BSScheduler]): Sequence of batch size schedulers. We expect the first scheduler to have
            been initialized first.
        milestones (Sequence[int]): Sequence of integers that reflects the milestone points. Must be sorted in a
            non-descending order.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 100 if epoch == 0
        >>> # bs = 100 if epoch == 1
        >>> # bs = 100 if epoch == 2
        >>> # bs = 100 if epoch == 3
        >>> # bs = 10 if epoch == 4
        >>> # bs = 11 if epoch == 5
        >>> # ...
        >>> scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        >>> scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        >>> scheduler = SequentialBS(dataloader, schedulers=[scheduler1, scheduler2], milestones=[5])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, schedulers: Sequence[BSScheduler], milestones=Sequence[int]):

        assert isinstance(schedulers, (tuple, list)) and len(schedulers) >= 2 and all(
            [isinstance(x, BSScheduler) for x in schedulers])
        assert isinstance(milestones, (tuple, list)) and len(milestones) >= 1 and all(
            [isinstance(x, int) for x in milestones]) and milestones[0] > 0
        assert all([milestones[i] >= milestones[i - 1] for i in range(1, len(milestones))]), \
            f"Milestones must be sorted, are {milestones}"

        if len(milestones) != len(schedulers) - 1:
            raise ValueError(f"SequentialBS expects the number of schedulers provided to be one more than the number "
                             f"of milestone points, but got {len(schedulers)} and the number of milestones is "
                             f"{len(milestones)}")

        super().__init__(schedulers[0].dataloader, schedulers[0].batch_size_manager, schedulers[0].max_batch_size,
                         schedulers[0].min_batch_size, verbose=False)

        for i in range(len(schedulers)):
            if schedulers[i].dataloader != self.dataloader:
                raise ValueError(f"SequentialBS expects all schedulers to belong to the same dataloader, but got "
                                 f"scheduler at index {i} to be different than the scheduler at index 0.")
            if not isinstance(schedulers[i].batch_size_manager, type(self.batch_size_manager)):
                raise ValueError(f"SequentialBS expects all schedulers to have the same batch size manager, but got "
                                 f"scheduler at index {i} to have a different batch size manager. Expected type of "
                                 f"batch size manager: {type(self.batch_size_manager).__name__}, got: "
                                 f"{type(schedulers[i].batch_size_manager).__name__}.")

            if schedulers[i].max_batch_size > self.max_batch_size:
                self.max_batch_size = schedulers[i].max_batch_size
            if schedulers[i].min_batch_size < self.min_batch_size:
                self.min_batch_size = schedulers[i].min_batch_size

            # Undoing the steps done by the schedulers.
            schedulers[i]._last_bs = self.batch_size_manager._base_batch_size
            schedulers[i].last_epoch -= 1

        self.set_batch_size(self.batch_size_manager._base_batch_size)  # Set the batch size back to initial value.

        self.schedulers: Tuple[BSScheduler, ...] = tuple(schedulers)
        self.milestones: Tuple[int, ...] = tuple(milestones)
        # Do the initial step again, but only for the first scheduler.
        self.schedulers[0].step()

    @property
    def finished(self) -> bool:
        """ Returns True if all the schedulers have already finished their job or have exceeded the minimum or maximum
        batch size. Otherwise, returns False.
        """
        if not self._finished:
            # The last milestone was reached and the last scheduler is finished.
            self._finished = self.last_epoch > self.milestones[-1] and self.schedulers[-1].finished
        return self._finished

    def step(self, **kwargs):
        """ Performs the step method for each scheduler until a milestone point is reached and a new scheduler is to be
        used. The new scheduler is used as if it is called for the first time.

        Args:
            **kwargs: All kwargs are passed to each scheduler.
        """
        self.last_epoch += 1  # We still increase last_epoch, even though the scheduler has finished its job. It should
        # not really matter.
        if self.last_epoch == 0 or self.finished:
            return
        i = bisect_right(self.milestones, self.last_epoch)
        scheduler = self.schedulers[i]
        if i > 0 and self.milestones[i - 1] == self.last_epoch:
            scheduler.last_epoch = 0
        if not scheduler.finished:
            scheduler.step(**kwargs)
            self._last_bs = scheduler.last_bs

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The wrapped scheduler
        states will also be saved.
        """
        state_dict = super().state_dict()
        state_dict['schedulers'] = [None] * len(self.schedulers)

        for i, s in enumerate(self.schedulers):
            state_dict['schedulers'][i] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        schedulers = state_dict.pop('schedulers')
        self.__dict__.update(state_dict)

        state_dict['schedulers'] = schedulers
        for i, s in enumerate(schedulers):
            self.schedulers[i].load_state_dict(s)

        self.set_batch_size(self.last_bs)  # Setting the batch size to the last computed batch size.


class PolynomialBS(BSScheduler):
    """ Increases the batch size using a polynomial function in the given total_iters. Unlike
    torch.optim.lr_scheduler.PolynomialLR whose polynomial factor decays from 1.0 to 0.5 ** power, in this case the
    polynomial factor decays from 1.5 ** power to 1.0.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        total_iters (int): The number of steps that the scheduler increases the batch size.
        power (float): The power of the polynomial. Default: 1.0.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if epoch == 0
        >>> # bs = 10 * 1.25 if epoch == 1
        >>> # bs = 12 * 1.33 if epoch == 2
        >>> # bs = 16 * 1.50 if epoch == 3
        >>> # bs = 24 * 2.00 if epoch == 4
        >>> # bs = 48 if epoch >= 5
        >>> scheduler = PolynomialBS(dataloader, total_iters=5, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], total_iters: int, power: float = 1.0,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(total_iters, int) and total_iters > 1

        self.total_iters: int = total_iters
        self.power: float = power
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. From epoch 1 to total_iters - 1, the current batch size is
        multiplied by an increasing polynomial factor.
        """
        if self.last_epoch == 0 or self.last_epoch >= self.total_iters:
            self._finished = self.last_epoch >= self.total_iters
            return self.batch_size

        remaining_steps = self.total_iters - self.last_epoch
        factor = 2.0 - ((1.0 - remaining_steps / self.total_iters) / (
                1.0 - (remaining_steps - 1) / self.total_iters)) ** self.power
        return rint(self.batch_size * factor)


class CosineAnnealingBS(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.CosineAnnealingLR which implements the cosine annealing part of
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. For batch size, we perform reverse annealing and instead
    of decreasing the batch size to min_batch_size we increase it to max_batch_size.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        total_iters (int): The number of steps that the scheduler increases the batch size.
        base_batch_size (Optional[int]): The base batch size. If None, the base batch size will be retrieved from
            the dataloader. Default: None.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if epoch % 10 == 0
        >>> # bs = 19 if epoch % 10 == 1
        >>> # bs = 41 if epoch % 10 == 2
        >>> # bs = 69 if epoch % 10 == 3
        >>> # bs = 91 if epoch % 10 == 4
        >>> # bs = 100 if epoch % 10 == 5
        >>> # bs = 91 if epoch % 10 == 6
        >>> # bs = 67 if epoch % 10 == 7
        >>> # bs = 37 if epoch % 10 == 8
        >>> # bs = 13 if epoch % 10 == 9
        >>> scheduler = CosineAnnealingBS(dataloader, total_iters=5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], total_iters: int, base_batch_size: Optional[int] = None,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(total_iters, int) and total_iters > 1
        assert base_batch_size is None or (isinstance(base_batch_size, int) and base_batch_size >= min_batch_size)

        self.total_iters: int = total_iters
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)
        self.base_batch_size: int = self.batch_size_manager._base_batch_size if base_batch_size is None else base_batch_size
        assert self.max_batch_size > self.base_batch_size
        self._float_batch_size: float = self.base_batch_size

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. Increases the batch size from base batch size to maximum
        batch size following a cyclic cosine curve. The implementation is equivalent to
        torch.optim.lr_scheduler.CosineAnnealingLR.get_lr() and instead of `eta_min` we use `self.max_batch_size` and
        we clip the values to be within bounds.
        """
        if self.last_epoch == 0:
            return self.batch_size

        if self.last_epoch == 1 and self.base_batch_size == self.batch_size:
            new_bs = self.max_batch_size + (self.base_batch_size - self.max_batch_size) * (
                    1 + math.cos(self.last_epoch * math.pi / self.total_iters)) / 2
        elif (self.last_epoch - 1 - self.total_iters) % (2 * self.total_iters) == 0:
            new_bs = self.batch_size + (self.base_batch_size - self.max_batch_size) * (
                    1 - math.cos(math.pi / self.total_iters)) / 2
        else:
            new_bs = (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / (
                    1 + math.cos(math.pi * (self.last_epoch - 1) / self.total_iters)) * (
                             self._float_batch_size - self.max_batch_size) + self.max_batch_size

        self._float_batch_size = new_bs
        return clip(rint(new_bs), min_x=self.base_batch_size, max_x=self.max_batch_size)


class ChainedBSScheduler(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.ChainedScheduler.
    Chains a list of batch size schedulers. It takes the list of batch size schedulers and performs consucutive
    step() functions belonging to them by just one call

    Args:
        schedulers (Sequence[BSScheduler]): List of chained schedulers.

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 100 if epoch == 0
        >>> # bs = 110 if epoch == 1
        >>> # bs = 121 if epoch == 2
        >>> # bs = 133 if epoch == 3
        >>> # bs = 14 if epoch == 4
        >>> # bs = 15 if epoch == 5
        >>> # bs = 16 if epoch == 6
        >>> # bs = 18 if epoch == 7
        >>> # ...
        >>> scheduler1 = ConstantBS(dataloader, factor=10, milestone=4)
        >>> scheduler2 = ExponentialBS(dataloader, gamma=1.1)
        >>> scheduler = ChainedBSScheduler([scheduler1, scheduler2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, schedulers: Sequence[BSScheduler]):
        assert isinstance(schedulers, (tuple, list)) and len(schedulers) > 1 and all(
            [isinstance(x, BSScheduler) for x in schedulers])

        dataloader: Optional[DataLoader] = schedulers[0].dataloader
        batch_size_manger: BatchSizeManager = schedulers[0].batch_size_manager
        for i in range(1, len(schedulers)):
            if schedulers[i].dataloader != dataloader:
                raise ValueError(f"ChainedBSScheduler expects all schedulers to belong to the same dataloader, but got "
                                 f"scheduler at index {i} to be different than the scheduler at index 0.")
            if not isinstance(schedulers[i].batch_size_manager, type(batch_size_manger)):
                raise ValueError(
                    f"ChainedBSScheduler expects all schedulers to have the same batch size manager, but got "
                    f"scheduler at index {i} to have a different batch size manager. Expected type of "
                    f"batch size manager: {type(batch_size_manger).__name__}, got: "
                    f"{type(schedulers[i].batch_size_manager).__name__}.")
            # We do not require equality for min_batch_size and max_batch_size, but maybe we should.

        self.dataloader: Optional[DataLoader] = dataloader
        self.batch_size_manager: BatchSizeManager = batch_size_manger
        self.schedulers: Tuple[BSScheduler, ...] = tuple(schedulers)
        self._last_bs: int = self.schedulers[-1].last_bs
        self.max_batch_size: int = max([x.max_batch_size for x in self.schedulers])
        self.min_batch_size: int = min([x.min_batch_size for x in self.schedulers])
        self._finished: bool = False
        # self.verbose: bool = False
        # self.last_epoch: int = 0
        self._init_get_new_bs()

    def step(self, **kwargs):
        """ Executes the step() function for all schedulers in order.

        Args:
            **kwargs: All kwargs arguments are passed to each scheduler.
        """
        for scheduler in self.schedulers:
            scheduler.step(**kwargs)
        self._last_bs = self.schedulers[-1].last_bs

    @property
    def finished(self) -> bool:
        """ Returns True if all the schedulers have already finished their job or have exceeded the minimum or maximum
        batch size. Otherwise, returns False.
        """
        if not self._finished:
            self._finished = all([x.finished for x in self.schedulers])
        return self._finished

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The wrapped scheduler
        states will also be saved.
        """
        state_dict = super().state_dict()
        state_dict['schedulers'] = [None] * len(self.schedulers)

        for i, s in enumerate(self.schedulers):
            state_dict['schedulers'][i] = s.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        schedulers = state_dict.pop('schedulers')
        self.__dict__.update(state_dict)

        state_dict['schedulers'] = schedulers
        for i, s in enumerate(schedulers):
            self.schedulers[i].load_state_dict(s)

        self.set_batch_size(self.last_bs)  # Setting the batch size to the last computed batch size.


class IncreaseBSOnPlateau(BSScheduler):
    """ The inverse of torch.optim.lr_scheduler.ReduceLROnPlateau.
    Increases the batch size when a metric has stopped improving. Models often benefit from increasing the batch size
    by a factor once the learning stagnates. This scheduler receives a metric value and if no improvement is seen for a
    given number of epochs, the batch size is increased.
    The step() function needs to receive the metric value using the `metrics` keyword argument.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        mode (str): One of `min`, `max`. In `min` mode, the batch size will be increased when the metric value has
            stopped decreasing; in `max` mode, the batch size will be increased when the metric value has stopped
            increasing. Default: 'min'.
        factor (float): Factor by which the batch size will be increased. Default: 2.0.
        patience (int): Number of epochs with no improvement after which the batch size will be increased. Default: 10.
        threshold (float): Threshold for measuring the new metric value, to only focus on significant changes.
            Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode, dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode. In `abs` mode, dynamic_threshold = best + threshold in 'max'
            mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming normal operation after the batch size has been reduced.
            Default: 0.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> scheduler = IncreaseBSOnPlateau(dataloader)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     scheduler.step(metric=val_loss)
    """

    def __init__(self, dataloader: Optional[DataLoader], mode: str = 'min', factor: float = 2.0, patience: int = 10,
                 threshold: float = 1e-4, threshold_mode: str = 'rel', cooldown: int = 0,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)
        assert isinstance(factor, (int, float)) and factor != 1.0 and factor >= 0.0
        # Factor is expected to be greater than 1, but we do not forbid batch size decay.
        assert isinstance(patience, int) and patience >= 0
        assert isinstance(threshold, (int, float)) and threshold > 0.0
        assert isinstance(cooldown, int) and cooldown >= 0

        self.mode: str = mode
        self.factor: float = float(factor)
        self.patience: int = patience
        self.threshold: float = float(threshold)
        self.threshold_mode: str = threshold_mode
        self.cooldown: int = cooldown
        self.cooldown_counter: int = 0
        self.mode_worse: float = torch.inf if mode == 'min' else -torch.inf
        self.best: float = self.mode_worse
        self.num_bad_epochs: int = 0

        self.last_epoch = 0  # setting last epoch to 0

        self._init_is_better(mode, threshold_mode)
        self._reset()

    @property
    def in_cooldown(self) -> bool:
        """ Returns True if scheduler is in cooldown, False otherwise.
        """
        return self.cooldown_counter > 0

    def _reset(self):
        """ Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    @staticmethod
    def is_better_min_rel(a: float, best: float, threshold: float) -> bool:
        return a < best * (1.0 - threshold)

    @staticmethod
    def is_better_min_abs(a: float, best: float, threshold: float) -> bool:
        return a < best - threshold

    @staticmethod
    def is_better_max_rel(a: float, best: float, threshold: float) -> bool:
        return a > best * (1.0 + threshold)

    @staticmethod
    def is_better_max_abs(a: float, best: float, threshold: float) -> bool:
        return a > best + threshold

    def _init_is_better(self, mode: str, threshold_mode: str):
        if mode not in ('min', 'max'):
            raise ValueError(f'Mode {mode} is unknown!')
        if threshold_mode not in ('rel', 'abs'):
            raise ValueError(f'Threshold mode {mode} is unknown!')

        if mode == 'min' and threshold_mode == 'rel':
            self.is_better = self.is_better_min_rel
        elif mode == 'min' and threshold_mode == 'abs':
            self.is_better = self.is_better_min_abs
        elif mode == 'max' and threshold_mode == 'rel':
            self.is_better = self.is_better_max_rel
        else:  # mode == 'min' and threshold_mode == 'abs':
            self.is_better = self.is_better_max_abs

    def get_new_bs(self, **kwargs) -> int:
        """ Returns the next batch size as an :class:`int`. Receives a metric and increases the batch size by a give
        factor if the metric has not been improved for `patience` epochs. After increasing the batch size, the
        scheduler goes through a cooldown period in which bad epochs are ignored.

        Args:
            **kwargs: All keyword arguments except 'metric' are ignored. The keyword 'metric' must be passed to the
                step() function, otherwise a TypeError would be raised.
        """
        if self.last_epoch == 0:  # Don't do anything at initialization.
            return self.batch_size

        metric = kwargs.pop('metrics', None)
        if metric is None:
            raise TypeError("IncreaseBSOnPlateau requires passing a 'metrics' keyword argument in the step() function.")

        current = float(metric)
        if self.is_better(current, self.best, self.threshold):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown.

        if self.num_bad_epochs > self.patience:
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            return rint(self.batch_size * self.factor)

        return self.batch_size

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self._init_is_better(self.mode, self.threshold_mode)


class CyclicBS(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.CyclicLR. Sets the batch size according to a cyclical batch size policy,
    inspired from the cyclical learning rate policy (CLR). The policy cycles the batch size between two boundaries with
    a constant frequency, similar to a reversed cycle from the method detailed in the paper `Cyclical Learning Rates
    for Training Neural Networks`_. The distance between the two boundaries can be scaled on a per-iteration or
    per-cycle basis.

    Cyclical batch size policy changes the batch size after every batch. The step() function should be called after a
    batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\\gamma^{\\text{cycle iterations}}` at each cycle
        iteration.

    This implementation was adapted from `pytorch/pytorch`_ which was adapted from the github repo: `bckenstler/CLR`_.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        base_batch_size (Optional[int]): Initial batch size which is the lower boundery in the cycle. If None, the
            base batch size will be retrieved from the dataloader. Default: None.
        step_size_down (int): Number of training iterations in the decreasing half of a cycle. Default: 2000.
        step_size_up (Optional[int]): Number of training iterations in the increasing half of a cycle. If
            step_size_down is None, it is set to step_size_down. Default: None.
        mode (str): One of `triangular`, `triangular2`, `exp_range`. Values correspond to the policies detailed above.
            If scale_fn is not None, this argument is ignored. Default: 'triangular'.
        gamma (float): Constant in the 'exp_range' scaling function: gamma ** (cycle iterations). Default: 1.0.
        scale_fn (Optional[Callable[[int], float]]): Custom scaling policy defined by a single argument lambda
            function, where 0 <= scale_fn(x) <= 1 for all x >= 0. If specified, then 'mode' is ignored. Default: None.
        scale_mode (str): One of `cycle`, `iterations`. Defines whether scale_fn is evaluated on cycle number of cycle
            iterations (training iterations since the start of the cycle). When scale_fn is None, scale_mode is
            automatically set to 'iterations' if mode is 'exp_range' and 'cycle' otherwhise. Default: 'cycle'.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper batch size boundary in the cycle. Functionally, it defines the cycle
            amplitude (upper_batch_size_bound - base_batch_size). The batch size at any cycle is the sum of
            base_batch_size and some scaling of the amplitude; therefore, upper_batch_size_bound may not actually be
            reached depending on scaling function. If None, max_batch_size is set to
            `len(self.dataloader.dataset) if available else 0 + 1`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
        >>> dataloader = ...
        >>> scheduler = CyclicBS(dataloader)
        >>> for epoch in range(100):
        >>>     for batch in dataloader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _pytorch/pytorch: https://github.com/pytorch/pytorch
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self, dataloader: Optional[DataLoader], base_batch_size: Optional[int] = None,
                 step_size_down: int = 2000, step_size_up: Optional[int] = None, mode: str = 'triangular',
                 gamma: float = 1.0, scale_fn: Optional[Callable[[int], float]] = None, scale_mode: str = 'cycle',
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert base_batch_size is None or (isinstance(base_batch_size, int) and base_batch_size >= min_batch_size)
        assert isinstance(step_size_down, int) and step_size_down > 0
        assert step_size_up is None or (isinstance(step_size_up, int) and step_size_up > 0)
        assert isinstance(gamma, (int, float)) and gamma > 0.0
        assert scale_fn is None or callable(scale_fn)
        assert scale_mode in ('cycle', 'iterations')

        if mode not in ('triangular', 'triangular2', 'exp_range') and scale_fn is None:
            raise ValueError("CyclicBS requires either a valid mode or passing a custom scale_fn.")
        self.mode: str = mode

        if step_size_up is None:
            step_size_up = step_size_down
        self.total_size: float = float(step_size_down + step_size_up)
        self.step_ratio: float = step_size_down / self.total_size
        self.gamma: float = float(gamma)

        self._scale_fn_custom: Optional[Callable[[int], float]] = scale_fn
        self.scale_mode: str = scale_mode
        self._init_scale_fn()

        self.base_batch_size: Optional[int] = base_batch_size
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)
        self.base_batch_size: int = self.batch_size_manager._base_batch_size if base_batch_size is None else base_batch_size
        assert self.min_batch_size < self.base_batch_size

    def _init_scale_fn(self):
        if self._scale_fn_custom is not None:
            self.scale_fn = self._scale_fn_custom
        elif self.mode == 'triangular':
            self.scale_fn = self._triangular_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'triangular2':
            self.scale_fn = self._triangular2_scale_fn
            self.scale_mode = 'cycle'
        elif self.mode == 'exp_range':
            self.scale_fn = partial(self._exp_range_scale_fn, self.gamma)
            self.scale_mode = 'iterations'

    @staticmethod
    def _triangular_scale_fn(x: int) -> float:
        return 1.0

    @staticmethod
    def _triangular2_scale_fn(x: int) -> float:
        return 1.0 / (2.0 ** (x - 1))

    @staticmethod
    def _exp_range_scale_fn(gamma: float, x: int) -> float:
        return gamma ** x

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. The value of the batch size cycles from base_batch_size to
        max_batch_size and back, while being scaled at each iteration.
        """
        if self.last_epoch == 0:  # Return base batch size or current batch size at initialization.
            return self.base_batch_size if self.base_batch_size is not None else self.batch_size

        ratio = self.last_epoch / self.total_size
        cycle = math.floor(1 + ratio)
        x = 1.0 + ratio - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        base_height = (self.base_batch_size - self.min_batch_size) * scale_factor
        if self.scale_mode == 'cycle':
            base_height *= self.scale_fn(cycle)
        else:
            base_height *= self.scale_fn(self.last_epoch)

        return rint(self.base_batch_size - base_height)

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The wrapped scheduler
        states will also be saved.
        """
        state_dict = super().state_dict()
        state_dict.pop('scale_fn')
        if self._scale_fn_custom is not None:
            state_dict.pop('_scale_fn_custom')
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self._init_scale_fn()


class CosineAnnealingBSWithWarmRestarts(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.CosineAnnealingWarmRestarts which implements `SGDR: Stochastic Gradient
    Descent with Warm Restarts`_. Unlike torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, which decreases the
    learning rate for :math:`t_{i}` iterations and then restarts, we increase the batch size from base_batch_size to
    max_batch_size in :math:`t_{i} + 1` iterations, then the batch size is restarted.

    This scheduler can be used after every batch.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        t_0 (int): The number of iterations for the first restart is t_0 + 1.
        base_batch_size (Optional[int]): The base batch size. If None, the base batch size will be retrieved from
            the dataloader. Default: None.
        factor (int): The factor with which :math:`t_{i}` is increased after a restart. Default: 1.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    .. _SGDR\\: Stochastic Gradient Descent with Warm Restarts: https://arxiv.org/abs/1608.03983

    Examples:
        >>> dataloader = ...
        >>> # Assuming the base batch size is 10.
        >>> # bs = 10 if last_epoch % 6 == 0
        >>> # bs = 19 if last_epoch % 6 == 1
        >>> # bs = 41 if last_epoch % 6 == 2
        >>> # bs = 69 if last_epoch % 6 == 3
        >>> # bs = 91 if last_epoch % 6 == 4
        >>> # bs = 100 if last_epoch % 6 == 5
        >>> scheduler = CosineAnnealingBSWithWarmRestarts(dataloader, 10)
        >>> for epoch in range(100):
        >>>     for batch in dataloader:
        >>>         train_batch(...)
        >>>         scheduler.step()
    """

    def __init__(self, dataloader: Optional[DataLoader], t_0: int, base_batch_size: Optional[int] = None,
                 factor: int = 1,
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(t_0, int) and t_0 > 0
        assert isinstance(factor, int) and factor > 0
        assert base_batch_size is None or (isinstance(base_batch_size, int) and base_batch_size >= min_batch_size)

        self.t_0: int = t_0
        self.t_i: int = t_0
        self.t_cur: int = 0
        self.factor: int = factor
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)
        self.base_batch_size: int = self.batch_size_manager._base_batch_size if base_batch_size is None else base_batch_size
        assert self.max_batch_size > self.base_batch_size

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. Increases the batch size from base batch size to maximum
        batch and restarts. The implementation is similar to
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, but instead of `eta_min` we use max_batch_size, and we
        increase the batch size instead of decreasing the learning rate. We clip the values to always remain within
        bound.
        """
        if self.last_epoch == 0:  # Don't do anything at initialization.
            return self.batch_size

        self.t_cur += 1
        if self.t_cur > self.t_i:  # > so that we reach max_batch_size
            self.t_cur -= self.t_i + 1  # + 1 so that we go back to base_batch_size
            self.t_i *= self.factor

        new_bs = self.base_batch_size + (self.max_batch_size - self.base_batch_size) * (
                1 + math.cos(math.pi + math.pi * self.t_cur / self.t_i)) / 2
        return clip(rint(new_bs), min_x=self.base_batch_size, max_x=self.max_batch_size)


class OneCycleBS(BSScheduler):
    """ Similar to torch.optim.lr_scheduler.OneCycleLR. Sets the batch size according to the one cycle batch size
    policy, inspired from the 1cycle learning rate policy. The one cycle batch size policy decreases the batch size
    from the base_batch_size to some minimum batch size and that it increases it to some maximum batch size bigger than
    the base_batch_size.
    This policy is inspired from the policy described in the paper `Super-Convergence: Very Fast Training of Neural
    Networks Using Large Learning Rates`_. It only uses two phases (base -> min, min -> max) instead of the three
    phases described in the paper (base -> min, min -> base, base -> max).

    The once cycle batch size policy changes the batch size after every batch. The step() function should be called
    after a batch has been used for training. But it may also be called after every epoch and the total_steps should be
    adjusted accordingly.

    This scheduler is not chainable.

    Args:
        dataloader (Optional[Dataloader]): Wrapped dataloader.
        total_steps (int): The total number of steps in the cycle.
        decay_percentage (float): The fraction of the cycle spend decreasing the batch size. 1 - decay_percentage will
            be spent increasing the batch size. Default: 0.3.
        base_batch_size (Optional[int]): The base batch size. If None, the base batch size will be retrieved from
            the dataloader. Default: None.
        strategy (str): One of `cos`, `linear`. Specifies the strategy used for annealing the batch size, 'cos' for
            cosine annealing, 'linear' for linear annealing. Default: 'cos'.
        batch_size_manager (Optional[BatchSizeManager]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Optional[int]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None, max_batch_size is set to `len(self.dataloader.dataset) if available else 0 + 1`.
            Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Examples:
            >>> dataloader = ...
            >>> scheduler = OneCycleBS(dataloader, total_steps=1000)
            >>> for epoch in range(100):
            >>>     for batch in dataloader:
            >>>         train_batch(...)
            >>>         scheduler.step()

    .. _Super-Convergence\\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(self, dataloader: Optional[DataLoader], total_steps: int, decay_percentage: float = 0.3,
                 base_batch_size: Optional[int] = None, strategy: str = 'cos',
                 batch_size_manager: Optional[BatchSizeManager] = None, max_batch_size: Optional[int] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert isinstance(total_steps, int)
        assert isinstance(decay_percentage, float) and 0 < decay_percentage < 1
        assert rint(total_steps * decay_percentage) > 0 and total_steps - rint(total_steps * decay_percentage) > 0
        assert base_batch_size is None or (isinstance(base_batch_size, int) and base_batch_size > min_batch_size)
        assert strategy in ('cos', 'linear')

        self.end_step_1: int = rint(total_steps * decay_percentage)
        self.end_step_2: int = total_steps - self.end_step_1

        self.strategy: str = strategy
        if strategy == 'cos':
            self.anneal_fn = self._annealing_cos
        else:
            self.anneal_fn = self._annealing_linear

        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)
        self.base_batch_size: int = self.batch_size_manager._base_batch_size if base_batch_size is None else base_batch_size
        assert self.max_batch_size > self.base_batch_size

    @staticmethod
    def _annealing_cos(start, end, percentage):
        """ Cosine annealing from start to end as percentage goes from 0.0 to 1.0.
        """
        return end + (start - end) / 2.0 * (1 + math.cos(math.pi * percentage))

    @staticmethod
    def _annealing_linear(start, end, percentage):
        """ Linear annealing from start to end as percentage goes from 0.0 to 1.0.
        """
        return (end - start) * percentage + start

    def get_new_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`. Increases the batch size from base batch size to maximum
        batch and restarts. The implementation is similar to
        torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, but instead of `eta_min` we use max_batch_size, and we
        increase the batch size instead of decreasing the learning rate. We clip the values to always remain within
        bound.
        """
        if self.last_epoch == 0:  # Don't do anything at initialization.
            return self.batch_size

        if self.last_epoch <= self.end_step_1:
            # Phase 1
            percentage = self.last_epoch / self.end_step_1
            new_bs = self.anneal_fn(self.base_batch_size, self.min_batch_size, percentage)
        else:
            # Phase 2
            percentage = (self.last_epoch - self.end_step_1) / self.end_step_2
            new_bs = self.anneal_fn(self.min_batch_size, self.max_batch_size, percentage)

            if percentage == 1.0:
                self._finished = True

        return clip(rint(new_bs), min_x=self.min_batch_size, max_x=self.max_batch_size)

# Inspired from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html.
import types
from typing import Callable, Union, List
from collections import Counter

from torch.utils.data import DataLoader, Dataset

__all__ = ['LambdaBS', 'MultiplicativeBS', 'StepBS', 'MultiStepBS', 'ConstantBS', 'BSScheduler', 'BatchSizeManager']


def check_isinstance(x, instance: type):
    if not isinstance(x, instance):
        raise TypeError(f"{type(x).__name__} is not a {x.__name__}.")


class BatchSizeManager:
    """ Base class for all batch size managers, used for getting and setting the batch size. It is not mandatory to
    inherit from this, but users must implement :meth:`get_current_batch_size` and :meth:`set_batch_size`.
    """

    def get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`.
        """
        raise NotImplementedError

    def set_batch_size(self, new_bs: int):
        """ Sets the new value of the batch size.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        raise NotImplementedError


class DefaultBatchSizeManager(BatchSizeManager):
    """ The default batch size manager used when the dataloader has a batch sampler. The batch sampler controls the
    batch size used by the dataloader, and it can be queried and changed. Changes are reflected in the number of samples
    given to the dataloader. See
    https://github.com/pytorch/pytorch/blob/772e104dfdfd70c74cbc9600cfc946dc7c378f68/torch/utils/data/sampler.py#L241.
    """

    def __init__(self, dataloader: DataLoader):
        check_isinstance(dataloader, DataLoader)
        if dataloader.batch_sampler is None:
            raise ValueError(f"Dataloader must have a batch sampler.")
        self.dataloader: DataLoader = dataloader

    def get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`, which is the owned by the batch
        sampler.
        """
        return self.dataloader.batch_sampler.batch_size

    def set_batch_size(self, new_bs: int):
        """ Sets the new value of the batch size, which is owned by the batch sampler.

        (!) Setting dataloader.batch_size raises a :class:`ValueError`. For now, we do not modify it, but we may need
        to.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        self.dataloader.batch_sampler.batch_size = new_bs

        # TODO: changing self.dataloader.batch_size raises an error. Maybe change the __setattr__ temporarily using
        #  a weak ref or sth


class CustomBatchSizeManager(BatchSizeManager):
    """ Custom batch size manager, used when the dataloader does not use a batch sampler. In this case, the batch size
    is controlled by the dataset wrapped by the dataloader, so this class expects the dataset to provide a getter and
    a setter for the batch size, named :meth:`get_batch_size` and :meth:`change_batch_size` respectively.
    """

    def __init__(self, dataset: Dataset):
        check_isinstance(dataset, Dataset)
        if not hasattr(dataset, "change_batch_size"):
            raise KeyError("Because the dataloader does not have a batch sampler, the dataset owns and controls the "
                           "batch size. In order to change the batch size after dataloader creation we require our "
                           "users to implement a Callable[[int],None] method named `change_batch_size` in their "
                           "dataset which changes the batch size. Please see TODO for examples.")
        if not hasattr(dataset, "get_batch_size"):
            raise KeyError("We require our users to implement a Callable[[], int] method named `get_batch_size` in "
                           "their dataset which returns the current batch size. Please see TODO for examples. ")
        self.dataset = dataset

    def get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataset as an :class:`int`.

        In this case, the dataset controls the batch size, so we require our users to implement a
        :class:`Callable[[], int]` method named :meth:`get_batch_size` in their dataset which returns the current value
        of the batch size.
        """
        return self.dataset.get_batch_size()

    def set_batch_size(self, new_bs: int):
        """ Sets the new value of the batch size.

        In this case, the dataset controls the batch size, so we require our users to implement a
        :class:`Callable[[int],None]` method named :meth:`change_batch_size` in their dataset which modifies the batch
        size to the given value.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        self.dataset.change_batch_size(new_bs)


class BSScheduler:
    def __init__(self, dataloader: DataLoader, batch_size_manager: Union[BatchSizeManager, None],
                 max_batch_size: Union[int, None], min_batch_size: int, verbose: bool):
        try:
            # Should we allow our users to use us with dataloader == None and just use the batch size managers they
            # provide us with?
            check_isinstance(dataloader, DataLoader)
        except TypeError:
            print("If you really need this feature, please open an issue at "
                  "https://github.com/ancestor-mithril/bs_scheduler/issues and describe your use case.")
            raise
        self.dataloader: DataLoader = dataloader
        self.verbose: bool = verbose

        if max_batch_size is None:
            self.max_batch_size: int = len(self.dataloader.dataset)
        else:
            if max_batch_size < 0:
                raise ValueError(f"Maximum batch size must be greater than 0, but is {max_batch_size}.")
            self.max_batch_size: int = min(len(self.dataloader.dataset), max_batch_size)

        if min_batch_size < 0:
            raise ValueError(f"Minimum batch size must be greater than 0, but is {min_batch_size}.")
        if min_batch_size > self.max_batch_size:
            raise ValueError(f"Minimum batch size must be smaller than or equal to the maximum batch size "
                             f"({max_batch_size}), but is {min_batch_size}.")
        self.min_batch_size: int = min_batch_size

        if batch_size_manager is not None:
            self.batch_size_manager: BatchSizeManager = batch_size_manager
        elif self.dataloader.batch_sampler is not None:
            self.batch_size_manager: BatchSizeManager = DefaultBatchSizeManager(self.dataloader)
        else:
            # We require the client to implement a "change_batch_size" method and a "get_batch_size" method for their
            # dataset.
            self.batch_size_manager: BatchSizeManager = CustomBatchSizeManager(self.dataloader.dataset)

        # Taking over the batch size manager methods for easier batch size getting&setting.
        self.get_current_batch_size = self.batch_size_manager.get_current_batch_size
        self.set_batch_size = self.batch_size_manager.set_batch_size

        # See https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html for "with_counter".
        self.last_epoch: int = -1
        self.base_bs: int = self.get_current_batch_size()
        self._last_bs: int = self.base_bs
        self.step()

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'dataloader'}

    def load_state_dict(self, state_dict: dict):
        """ Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        # TODO: Test training, saving, loading and resuming scheduler. Ensure that the batch size is set correctly.
        self.set_batch_size(self.get_last_bs())  # Setting the batch size to the last computed batch size.

    def get_last_bs(self) -> int:
        """ Returns the last computed batch size by current scheduler. If called before the first call to :meth:`step`
        returns the base batch size.
        """
        return self._last_bs

    def get_bs(self) -> int:
        """ Computes the next batch size. Should not be called explicitly in client code, but it doesn't really matter
        if the client does so.
        """
        raise NotImplementedError

    def print_bs(self, new_bs):
        if self.verbose:
            print(f'Adjusting batch size to {new_bs}')

    def step(self):
        # TODO: Documentation
        # TODO: Check how the dataloader behaves if we change the batch size mid epoch. Write a guideline for this.
        #  Changing the batch size does not impact batch sizes loaded by workers before the change.
        # TODO: Check if changing the batch size needs locking. Because of multiprocessing. Normally it should not.
        self.last_epoch += 1
        new_bs = max(min(self.get_bs(), self.max_batch_size), self.min_batch_size)  # Clip new_bs. Clearer than if elif.
        self.set_batch_size(new_bs)
        self.print_bs(new_bs)
        self._last_bs = new_bs


class LambdaBS(BSScheduler):
    """ Sets the batch size to the initial batch size times a given function. Unlike torch.optim.lr_scheduler.LambdaLR,
    there is a single batch size for a given dataloader so only one function should be passed as a parameter.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        bs_lambda (Callable[[int], float]): A function which computes a multiplicative factor given an integer
            parameter epoch.
        batch_size_manager (Union[BatchSizeManager, None]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greater than the lenght of the dataset wrapped by the dataloader, max_batch_size
            is set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
        >>> dataloader = ...
        >>> func = lambda epoch: 1.05 ** epoch
        >>> scheduler = LambdaBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: DataLoader, bs_lambda: Callable[[int], int],
                 batch_size_manager: Union[BatchSizeManager, None] = None, max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.bs_lambda: Callable[[int], int] = bs_lambda
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The batch size lambda
        function will only be saved if they are callable objects and not if they are functions or lambdas.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('dataloader', 'bs_lambda')}
        state_dict['bs_lambda'] = None
        if not isinstance(self.bs_lambda, types.FunctionType):
            self.bs_lambda.__dict__.copy()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        bs_lambda = state_dict.pop('bs_lambda')
        self.__dict__.update(state_dict)
        self.set_batch_size(self.get_last_bs())  # Setting the batch size to the last computed batch size.
        if bs_lambda is not None:
            self.bs_lambda.__dict__.update(bs_lambda)

    def get_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`.

        It is calculated as the initial value of the batch size times the factor returned by `bs_lambda`.
        """
        return int(self.base_bs * self.bs_lambda(self.last_epoch))


class MultiplicativeBS(BSScheduler):
    """ Multiply the batch size by a factor given in the specified function. Unlike
    torch.optim.lr_scheduler.MultiplicativeLR, there is a single batch size for a given dataloader so only one function
    should be passed as a parameter.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        bs_lambda: (Callable[[int], float]): A function which computes a multiplicative factor given an integer
            parameter epoch.
        batch_size_manager (Union[BatchSizeManager, None]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greater than the lenght of the dataset wrapped by the dataloader, max_batch_size
            is set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
        >>> dataloader = ...
        >>> func = lambda epoch: 1.05
        >>> scheduler = MultiplicativeBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: DataLoader, bs_lambda: Callable[[int], int],
                 batch_size_manager: Union[BatchSizeManager, None] = None, max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.bs_lambda: Callable[[int], int] = bs_lambda
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def state_dict(self) -> dict:
        """ Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader. The batch size lambda
        function will only be saved if they are callable objects and not if they are functions or lambdas.
        """
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('dataloader', 'bs_lambda')}
        state_dict['bs_lambda'] = None
        if not isinstance(self.bs_lambda, types.FunctionType):
            self.bs_lambda.__dict__.copy()
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        bs_lambda = state_dict.pop('bs_lambda')
        self.__dict__.update(state_dict)
        self.set_batch_size(self.get_last_bs())  # Setting the batch size to the last computed batch size.
        if bs_lambda is not None:
            self.bs_lambda.__dict__.update(bs_lambda)

    def get_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`.

        It is calculated as the current value of the batch size times the factor returned by `bs_lambda`.
        """
        return int(self.get_current_batch_size() * self.bs_lambda(self.last_epoch))


class StepBS(BSScheduler):
    """ Multiplies the batch size by gamma every step_size epochs.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        step_size (int): Period of batch size growth.
        gamma (float): Multiplicative factor of batch size growth. Default: 2.0.
        batch_size_manager (Union[BatchSizeManager, None]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greater than the lenght of the dataset wrapped by the dataloader, max_batch_size
            is set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
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

    def __init__(self, dataloader: DataLoader, step_size: int, gamma: float = 2.0,
                 batch_size_manager: Union[BatchSizeManager, None] = None, max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.step_size: int = step_size
        self.gamma: float = gamma
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`.

        It returns the current batch size times gamma each step_size epochs, otherwise it returns the current batch
        size.
        """
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return self.get_current_batch_size()
        return int(self.get_current_batch_size() * self.gamma)


class MultiStepBS(BSScheduler):
    """ Multiplies the batch size by gamma once the number of epochs reaches one of the milestones.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        milestones (List[int]): List of epoch indices. Must be sorted in non-desceding order.
        gamma (float): Multiplicative factor of batch size growth. Default: 2.0.
        batch_size_manager (Union[BatchSizeManager, None]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greater than the lenght of the dataset wrapped by the dataloader, max_batch_size
            is set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
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

    def __init__(self, dataloader: DataLoader, milestones: List[int], gamma: float = 2.0,
                 batch_size_manager: Union[BatchSizeManager, None] = None, max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        assert all(milestones[i] <= milestones[i + 1] for i in range(len(milestones) - 1)), "Milestones must be sorted."
        self.milestones: Counter[int, int] = Counter(milestones)
        self.gamma: float = gamma
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`.

        It returns the current batch size times gamma each epoch a milestone is reached, otherwise it returns the
        current batch size. Beware that in the event of multiple milestones with the same value, the current batch size
        is multiplied with gamma multiple times.
        """
        if self.last_epoch not in self.milestones:
            return self.get_current_batch_size()
        return int(self.get_current_batch_size() * self.gamma ** self.milestones[self.last_epoch])


class ConstantBS(BSScheduler):
    """ Increases the batch size by a constant factor until the number of epochs reaches a pre-defined milestone.
    The batch size is multiplied by the constant factor during initialization and is multiplied again with the inverse
    of the constant factor when the milestone is reached.
    If the constant factor makes the batch size increase the image out of bounds, the constant factor is changed
    automatically such that the batch size remains within bounds.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        factor (float): The number we multiply the batch size until the milestone.
        milestone (int): The number of steps that the scheduler increases the learning rate. Default: 5.
        batch_size_manager (Union[BatchSizeManager, None]): If not None, a custom class which manages the batch size,
            which provides a getter and setter for the batch size. Default: None.
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greater than the lenght of the dataset wrapped by the dataloader, max_batch_size
            is set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
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

    def __init__(self, dataloader: DataLoader, factor: float, milestone: int = 5,
                 batch_size_manager: Union[BatchSizeManager, None] = None, max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.factor: float = factor
        self.milestone: int = milestone
        super().__init__(dataloader, batch_size_manager, max_batch_size, min_batch_size, verbose)

    def get_bs(self) -> int:
        """ Returns the next batch size as an :class:`int`.

        The value of the batch size is changed once at initialization, when the batch size is multiplied with the given
        factor, and twice when the milestone is reached and the batch size is multiplied with the inverse of the given
        factor. The factor is adjusted during initialization such that it does not return a batch size out of bounds.
        """
        current_batch_size = self.get_current_batch_size()

        if self.last_epoch == 0:
            max_factor = self.max_batch_size / current_batch_size
            min_factor = self.min_batch_size / current_batch_size
            if self.factor > max_factor:
                self.factor = max_factor
            elif self.factor < min_factor:
                self.factor = min_factor
            return int(current_batch_size * self.factor)

        if self.last_epoch != self.milestone:
            return current_batch_size

        return int(current_batch_size * (1.0 / self.factor))

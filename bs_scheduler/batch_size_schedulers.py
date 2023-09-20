# Inspired from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html.
import types
from typing import Callable, Union

from torch.utils.data import DataLoader

__all__ = ['LambdaBS', 'MultiplicativeBS']


class BSScheduler:
    def __init__(self, dataloader: DataLoader, max_batch_size: Union[int, None], min_batch_size: int, verbose: bool):
        # TODO: Finalize interface and documentation
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"{type(dataloader).__name__} is not a Dataloader")
        self.dataloader: DataLoader = dataloader
        self.verbose: bool = verbose

        self.max_batch_size: int = len(self.dataloader.dataset)
        if max_batch_size is not None:
            self.max_batch_size = min(self.max_batch_size, max_batch_size)
        assert min_batch_size > 0, f"Minimum batch size must be greater than 0, is {min_batch_size}"
        assert min_batch_size <= self.max_batch_size, f"Minimum batch size ({min_batch_size}) must be smaller or equal " \
                                                      f"than maximum batch size ({self.max_batch_size})"
        self.min_batch_size: int = min_batch_size

        if self.dataloader.batch_sampler is not None:
            self.set_batch_size = self.batch_sampler_set_batch_size
            self.get_current_batch_size = self.batch_sampler_get_current_batch_size
        else:
            # We require the client to implement a "change_batch_size" method and a "get_batch_size" method for their
            # dataset.
            if not hasattr(self.dataloader.dataset, "change_batch_size"):
                # TODO: Validate the error name
                raise KeyError("The wrapped dataloader does not have a batch sampler because the dataset controls the "
                               "batch size. To change the batch size after dataloader creation we require our users to "
                               "implement a Callable[[int],None] method named `change_batch_size` in their dataset "
                               "which would change the batch size. Please see TODO for examples.")
            if not hasattr(self.dataloader.dataset, "get_batch_size"):
                # TODO: Validate the error name
                raise KeyError("We require our users to implement a Callable[[], int] method named `get_batch_size` in "
                               "their dataset which returns the current batch size. Please see TODO for examples. ")

        # See https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html for "with_counter".
        self.last_epoch: int = -1
        self.base_bs: int = self.get_current_batch_size()
        self._last_bs: int = self.base_bs
        self.step()

    def get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`.

        If the dataloader does not have a batch sampler, namely when the dataset itself controls the batch size, we
        require our users to implement a :class:`Callable[[], int]` method named :meth:`get_batch_size` in their dataset
        which returns the current value of the batch size. Otherwise, this method is overriden by
        :meth:`batch_sampler_get_current_batch_size`.
        """
        return self.dataloader.dataset.get_batch_size()

    def batch_sampler_get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`.

        Overrides :meth:`get_current_batch_size` when the dataloader has a batch sampler which controls the batch size.
        """
        return self.dataloader.batch_sampler.batch_size

    def set_batch_size(self, new_bs: int):
        """ Sets the new value of the batch size. If the dataloader does not have a batch sampler, namely when the
        dataset itself controls the batch size, we require our users to implement a :class:`Callable[[int],None]` method
        named :meth:`change_batch_size` in their dataset which modifies the batch size to the given value. Otherwise,
        this method is overriden by :meth:`batch_sampler_set_batch_size`.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        self.dataloader.dataset.change_batch_size(new_bs)

    def batch_sampler_set_batch_size(self, new_bs: int):
        """ Overrides :meth:`set_batch_size` when the dataloader has a batch sampler which controls the batch size.

        (!) Setting dataloader.batch_size raises a :class:`ValueError`. For now, we do not modify it, but we may need
        to.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        self.dataloader.batch_sampler.batch_size = new_bs

        # TODO: Read this:
        # NOTE [ IterableDataset and __len__ ]
        #
        # For `IterableDataset`, `__len__` could be inaccurate when one naively
        # does multi-processing data loading, since the samples will be duplicated.
        # However, no real use case should be actually using that behavior, so
        # it should count as a user error. We should generally trust user
        # code to do the proper thing (e.g., configure each replica differently
        # in `__iter__`), and give us the correct `__len__` if they choose to
        # implement it (this will still throw if the dataset does not implement
        # a `__len__`).
        #
        # To provide a further warning, we track if `__len__` was called on the
        # `DataLoader`, save the returned value in `self._len_called`, and warn
        # if the iterator ends up yielding more than this number of samples.

        # Cannot statically verify that dataset is Sized

        # TODO: changing self.dataloader.batch_size raises an error. Maybe change the __setattr__ temporarily using
        #  a weak ref or sth

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which is not the dataloader.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'dataloader'}

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
        self.set_batch_size(self.get_last_bs())  # Setting the batch size to the last computed batch size.

    def get_last_bs(self) -> int:
        """ Return last computed batch size by current scheduler. If called before the first call to :meth:`step`
        returns the base batch size.
        """
        return self._last_bs

    def get_bs(self) -> int:
        """ Computes the next batch size. Should not be called explicitly in client code. """
        raise NotImplementedError

    def print_bs(self, new_bs):
        if self.verbose:
            print(f'Adjusting batch size to {new_bs}')

    def step(self):
        # TODO: Documentation + implementation
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
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greate than the lenght of the dataset wrapped by the dataloader, max_batch_size is
             set to `len(self.dataloader.dataset)`. Default: None.
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

    def __init__(self, dataloader: DataLoader, bs_lambda: Callable[[int], int], max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.bs_lambda = bs_lambda
        super().__init__(dataloader, max_batch_size, min_batch_size, verbose)

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
        max_batch_size (Union[int, None]): Upper limit for the batch size so that a batch of size max_batch_size fits
            in the memory. If None or greate than the lenght of the dataset wrapped by the dataloader, max_batch_size is
             set to `len(self.dataloader.dataset)`. Default: None.
        min_batch_size (int): Lower limit for the batch size which must be greater than 0. Default: 1.
        verbose (bool): If ``True``, prints a message to stdout for each update. Default: ``False``.

    Example:
        >>> dataloader = ...
        >>> func = lambda epoch: 1.05
        >>> scheduler = LambdaBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, dataloader: DataLoader, bs_lambda: Callable[[int], int], max_batch_size: Union[int, None] = None,
                 min_batch_size: int = 1, verbose: bool = False):
        self.bs_lambda = bs_lambda
        super().__init__(dataloader, max_batch_size, min_batch_size, verbose)

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

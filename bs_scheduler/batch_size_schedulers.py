# Inspired from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html.
from typing import Callable

from torch.utils.data import DataLoader

__all__ = ['LambdaBS']


class BSScheduler:
    def __init__(self, dataloader):
        # TODO: Finalize interface and documentation
        if not isinstance(dataloader, DataLoader):
            raise TypeError(f"{type(dataloader).__name__} is not a Dataloader")
        self.dataloader = dataloader

        self._last_bs = None
        # See https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html for "with_counter".

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'dataloader'}

    def load_state_dict(self, state_dict: dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_bs(self) -> int:
        """ Return last computed batch size by current scheduler. If called before the first call to :meth:`step`
        returns None.
        """
        # TODO: Check if it better to return dataloader.batch_size
        return self._last_bs

    def _get_bs(self) -> int:
        """ Computes the next batch size. Should not be called explicitly in client code. """
        raise NotImplementedError

    def step(self):
        # TODO: Documentation + implementation
        pass


class LambdaBS(BSScheduler):
    """ Sets the batch size to the initial batch size times a given function. Unlike torch.optim.lr_scheduler.LambdaLR,
    there is a single batch size for a given dataloader so only one function should be given.

    Args:
        dataloader (DataLoader): Wrapped dataloader.
        bs_lambda: (Callable[[int], int]): A function which computes a multiplicative factor given an integer parameter
            epoch.

    Example:
        >>> dataloader = ...
        >>> func = lambda epoch: int(100 * 1.05 ** epoch)
        >>> scheduler = LambdaBS(dataloader, bs_lambda=func)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(self, dataloader: DataLoader, bs_lambda: Callable[[int], int]):
        pass

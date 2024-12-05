from torch.utils.data import DataLoader, Dataset

from .utils import check_isinstance


class BatchSizeManager:
    """ Base class for all batch size managers, used for getting and setting the batch size.
    Users must implement :meth:`get_current_batch_size` and :meth:`set_batch_size`.
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
            raise ValueError("Dataloader must have a batch sampler.")
        self.dataloader: DataLoader = dataloader

    def get_current_batch_size(self) -> int:
        """ Returns the current batch size used by the dataloader as an :class:`int`. The batch size member variable is
        owned by the batch sampler.
        """
        return self.dataloader.batch_sampler.batch_size

    def set_batch_size(self, new_bs: int):
        """ Sets the new value of the batch size, which is owned by the batch sampler.

        Args:
            new_bs (int): The new batch sizes that needs to be set.
        """
        self.dataloader.batch_sampler.batch_size = new_bs


class CustomBatchSizeManager(BatchSizeManager):
    """ Custom batch size manager, used when the dataloader does not use a batch sampler. In this case, the batch size
    is controlled by the dataset wrapped by the dataloader, so this class expects the dataset to provide a getter and
    a setter for the batch size, named :meth:`get_batch_size` and :meth:`change_batch_size` respectively.
    """

    def __init__(self, dataset: Dataset):
        check_isinstance(dataset, Dataset)
        if not hasattr(dataset, 'change_batch_size'):
            raise KeyError("Because the dataloader does not have a batch sampler, the dataset owns and controls the "
                           "batch size. In order to change the batch size after dataloader creation we require our "
                           "users to implement a Callable[[int],None] method named `change_batch_size` in their "
                           "dataset which changes the batch size. Please see "
                           "https://ancestor-mithril.github.io/bs-scheduler/tutorials/ for examples.")
        if not hasattr(dataset, 'get_batch_size'):
            raise KeyError("We require our users to implement a Callable[[], int] method named `get_batch_size` in "
                           "their dataset which returns the current batch size. Please see "
                           "https://ancestor-mithril.github.io/bs-scheduler/tutorials/ for examples. ")
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

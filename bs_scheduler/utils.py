def check_isinstance(x, instance: type):
    if not isinstance(x, instance):
        raise TypeError(f"{type(x).__name__} is not a {instance.__name__}.")


def rint(x: float) -> int:
    """ Rounds to the nearest int and returns the value as int.
    """
    return int(round(x))


def clip(x: int, min_x: int, max_x: int) -> int:
    """ Clips x to [min, max] interval.
    """
    if x < min_x:
        return min_x
    if x > max_x:
        return max_x
    return x

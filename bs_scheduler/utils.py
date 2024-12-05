def check_isinstance(x, instance: type):
    if not isinstance(x, instance):
        raise TypeError(f"{type(x).__name__} is not a {instance.__name__}.")

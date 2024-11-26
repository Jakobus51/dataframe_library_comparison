import datetime


def timer(func):
    """Print the runtime of the decorated function."""

    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        res = func(*args, **kwargs)
        time_taken = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Finished {func.__name__} in {time_taken} seconds")
        return res, time_taken

    return wrapper

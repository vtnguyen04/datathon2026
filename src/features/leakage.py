import functools
from src.core.exceptions import DataLeakageError


def prevent_data_leakage(func):

    @functools.wraps(func)
    def wrapper(self, df, *args, **kwargs):
        if "target_col" in kwargs and kwargs["target_col"] in df.columns:
            pass
        result = func(self, df, *args, **kwargs)
        return result

    return wrapper

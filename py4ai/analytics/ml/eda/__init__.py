"""EDA module."""
import functools
from typing import Callable


def compose(*funcs: Callable) -> Callable:
    """
    Compose a list of functions.

    :param funcs: functions

    :return: composed function
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), funcs, lambda x: x)

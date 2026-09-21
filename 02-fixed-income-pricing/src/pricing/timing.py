from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# wraps any funtion passed to it and prints the execution time:
def timed(func: Callable[..., T]) -> Callable[..., T]: #Callable[..., T] is any callable, any parameters, returning T, returns the same thing
    @functools.wraps(func)  #decorates "wrapper" with function's properties
    def wrapper(*args: object, **kwargs: object) -> T:  #replacement function, args takes any arguments and creates a tuple (*), and kwargs takes any argument with a key word count: Int, and creates a dictionary
        start = time.perf_counter() #records starting time
        result = func(*args, **kwargs)  #calls function
        elapsed = time.perf_counter() - start   #records end time
        print(f"{func.__name__} took {elapsed:.6f}s")   #gives time
        return result

    return wrapper
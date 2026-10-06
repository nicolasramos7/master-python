"""Sitting 1, part 2: exponential backoff with jitter, as an iterator and a decorator.

Two pieces, and they fit together:

1. `BackoffDelays` -- a class that implements the **iterator protocol** by hand.
   Iterating it yields how long to sleep before each retry:

       for i in range(attempts - 1):          # attempts - 1 sleeps between attempts
           cap = min(max_delay, base_delay * 2**i)
           yield rng.uniform(0, cap)           # "full jitter"

   Why jitter: if 50 clients all fail at the same instant and all back off by
   exactly 0.1s, they all retry at the same instant and fail again together.
   Randomising the wait spreads them out. (AWS's architecture blog calls this
   variant "full jitter"; it is what most client libraries do.)

   The protocol is two methods: `__iter__` returns `self`, and `__next__`
   returns the next value or raises `StopIteration`. That's all a `for` loop
   needs. The catch: an iterator is **single-use**. Once exhausted it stays
   exhausted -- the second `for` loop over it does nothing, silently.

2. `retry(...)` -- a decorator *factory* for **async** functions. Project 2 had
   you write sync decorators; this one wraps a coroutine function:

       @retry(attempts=5, base_delay=0.01, max_delay=0.5, retry_on=(TransientError,))
       async def fetch_bars(...) -> Payload: ...

   Call the wrapped function; if it raises one of `retry_on`, log a WARNING,
   `await sleep(delay)` and call it again. Anything not in `retry_on` propagates
   at once. When the delays run out, re-raise the last error.

   `sleep` is a parameter so tests can pass a fake that records the delays
   instead of waiting. That is how you test code that involves time: inject
   the clock, don't wait for it.

**The traps, both silent:**

- `time.sleep` inside an `async def` blocks the *whole event loop* -- every
  other request stops too. Always `await asyncio.sleep`.
- Build the `BackoffDelays` inside the wrapper, per call. Build it once, at
  decoration time, and the first call that retries exhausts it; every later
  call gets zero retries. Nothing errors -- retries just quietly stop.

The Python in it: `__iter__`/`__next__`, `StopIteration`, closures,
`functools.wraps`, `ParamSpec`, `async def` / `await`, the `logging` module.
"""

# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import asyncio
import functools
import logging
import random
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from typing import Any, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

Sleep = Callable[[float], Awaitable[None]]
"""Anything you can `await sleep(seconds)`. Normally `asyncio.sleep`."""


class BackoffDelays:
    """Yields `attempts - 1` full-jitter delays, then stops. Single-use, like every iterator."""

    def __init__(
        self,
        attempts: int,
        base_delay: float,
        max_delay: float,
        rng: random.Random | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError(f"attempts must be >= 1, got {attempts}")
        self.attempts = attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.rng = rng if rng is not None else random.Random()
        self._i = 0

    def __iter__(self) -> Iterator[float]:
        return self

    def __next__(self) -> float:
        if self._i >= self.attempts - 1:
            raise StopIteration
        cap = min(self.max_delay, self.base_delay * 2**self._i)
        self._i += 1
        return self.rng.uniform(0.0, cap)



def retry(
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    retry_on: tuple[type[Exception], ...],
    sleep: Sleep = asyncio.sleep,
    rng: random.Random | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Coroutine[Any, Any, T]]]:
    """Retry an async function on the given exceptions, with full-jitter backoff.

    Args:
        attempts: Total calls, including the first. `attempts=1` means no retry.
        base_delay, max_delay: See `BackoffDelays`.
        retry_on: Exception types worth retrying. Everything else propagates.
        sleep: Awaited between attempts. Tests pass a fake.
        rng: For reproducible jitter in tests.

    Every retry logs one WARNING naming the function, the attempt number, the
    delay, and the error.
    """

    def decorate(fn: Callable[P, Awaitable[T]]) -> Callable[P, Coroutine[Any, Any, T]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delays = BackoffDelays(attempts, base_delay, max_delay, rng)
            attempt = 1
            while True:
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    delay = next(delays, None)
                    if delay is None:
                        raise
                    logger.warning(
                        "%s failed (attempt %d/%d), retrying in %.3fs: %s",
                        fn.__name__,
                        attempt,
                        attempts,
                        delay,
                        exc,
                    )
                    await sleep(delay)
                    attempt += 1

        return wrapper
    return decorate
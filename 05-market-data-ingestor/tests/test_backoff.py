"""Sitting 1, step 3. An iterator by hand, then an async decorator built on it.

Nothing here waits for real time. `FakeSleep` records every delay the
decorator asks for and returns at once -- inject the clock, don't wait for it.
"""

import asyncio
import logging
import random
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from ingest.backoff import BackoffDelays, retry
from ingest.errors import PermanentError, TransientError


class FakeSleep:
    """Stands in for `asyncio.sleep`. Awaitable, instant, and it takes notes."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


class Flaky:
    """An async callable that raises `errors[i]` on call i, then returns "ok"."""

    __name__ = "flaky"
    """Functions have a __name__; instances don't. Your log line will want one."""

    def __init__(self, *errors: Exception) -> None:
        self.errors = list(errors)
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def retrying(
    fn: Flaky, sleep: FakeSleep, attempts: int = 5
) -> Callable[[], Coroutine[Any, Any, str]]:
    """`fn`, wrapped in the retry policy these tests share."""
    decorate = retry(
        attempts=attempts, base_delay=0.1, max_delay=1.0, retry_on=(TransientError,), sleep=sleep
    )
    return decorate(fn)


# --- BackoffDelays: the iterator protocol ---------------------------------


def test_yields_one_fewer_delay_than_attempts() -> None:
    assert len(list(BackoffDelays(5, 0.1, 10.0))) == 4
    assert list(BackoffDelays(1, 0.1, 10.0)) == []


def test_each_delay_is_inside_its_exponential_envelope() -> None:
    """Delay i is uniform on [0, min(max_delay, base_delay * 2**i)]."""
    for seed in range(200):
        delays = list(BackoffDelays(6, 0.1, 1.0, random.Random(seed)))
        caps = [0.1, 0.2, 0.4, 0.8, 1.0]
        assert all(0.0 <= d <= cap for d, cap in zip(delays, caps, strict=True))


def test_delays_are_jittered_not_fixed() -> None:
    """Fixed delays mean every failed client retries at the same instant."""
    firsts = {next(BackoffDelays(3, 1.0, 10.0, random.Random(seed))) for seed in range(50)}
    assert len(firsts) == 50


def test_same_seed_same_delays() -> None:
    a = list(BackoffDelays(6, 0.1, 1.0, random.Random(7)))
    b = list(BackoffDelays(6, 0.1, 1.0, random.Random(7)))
    assert a == b


def test_is_its_own_iterator_and_is_single_use() -> None:
    """`__iter__` returns self. Once exhausted, it stays exhausted -- silently."""
    delays = BackoffDelays(3, 0.1, 1.0)
    assert iter(delays) is delays
    assert len(list(delays)) == 2
    assert list(delays) == []
    with pytest.raises(StopIteration):
        next(delays)


def test_rejects_zero_attempts() -> None:
    with pytest.raises(ValueError):
        BackoffDelays(0, 0.1, 1.0)


# --- retry: the decorator ---------------------------------------------------


def test_retries_until_success() -> None:
    sleep = FakeSleep()
    flaky = Flaky(TransientError("503"), TransientError("503"))
    wrapped = retrying(flaky, sleep, attempts=5)
    assert asyncio.run(wrapped()) == "ok"
    assert flaky.calls == 3
    assert len(sleep.delays) == 2


def test_gives_up_and_reraises_the_last_error() -> None:
    sleep = FakeSleep()
    flaky = Flaky(*(TransientError(f"attempt {i}") for i in range(1, 10)))
    wrapped = retrying(flaky, sleep, attempts=4)
    with pytest.raises(TransientError, match="attempt 4"):
        asyncio.run(wrapped())
    assert flaky.calls == 4
    assert len(sleep.delays) == 3


def test_does_not_retry_errors_outside_retry_on() -> None:
    """A 404 will be a 404 next time too. Retrying it only wastes the vendor's patience."""
    sleep = FakeSleep()
    flaky = Flaky(PermanentError("404"))
    wrapped = retrying(flaky, sleep, attempts=5)
    with pytest.raises(PermanentError):
        asyncio.run(wrapped())
    assert flaky.calls == 1
    assert sleep.delays == []


def test_every_call_gets_its_own_retries() -> None:
    """The trap: build BackoffDelays once, at decoration time, and the first call
    that retries uses them up. The second call then gets no retries at all."""
    sleep = FakeSleep()
    flaky = Flaky()
    wrapped = retrying(flaky, sleep, attempts=3)
    for _ in range(3):
        flaky.errors = [TransientError("503"), TransientError("503")]
        assert asyncio.run(wrapped()) == "ok"
    assert flaky.calls == 9


def test_passes_arguments_and_return_value_through() -> None:
    @retry(attempts=2, base_delay=0.1, max_delay=1.0, retry_on=(TransientError,), sleep=FakeSleep())
    async def add(a: int, *, b: int) -> int:
        return a + b

    assert asyncio.run(add(2, b=3)) == 5


def test_preserves_the_wrapped_functions_identity() -> None:
    """functools.wraps. Without it every decorated function is called "wrapper"
    in logs, tracebacks and help()."""

    @retry(attempts=2, base_delay=0.1, max_delay=1.0, retry_on=(TransientError,))
    async def fetch_something() -> None:
        """Docstring survives."""

    assert fetch_something.__name__ == "fetch_something"
    assert fetch_something.__doc__ == "Docstring survives."


def test_logs_one_warning_per_retry(caplog: pytest.LogCaptureFixture) -> None:
    sleep = FakeSleep()
    flaky = Flaky(TransientError("HTTP 503"), TransientError("HTTP 429"))
    wrapped = retrying(flaky, sleep, attempts=5)
    with caplog.at_level(logging.WARNING, logger="ingest.backoff"):
        asyncio.run(wrapped())
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert "HTTP 503" in warnings[0].getMessage()
    assert "HTTP 429" in warnings[1].getMessage()

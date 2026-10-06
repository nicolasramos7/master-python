"""Sitting 2, step 4. Benchmarks you can trust -- tested on what's stable, not on noise.

Timing tests are only safe when the effect is huge compared with the noise.
"Concurrent I/O is several times faster" is; "processes beat threads by 1.8x"
is not, on a laptop running other things. So the CPU benchmark is tested on
its *answers*, and its timings go in your README instead.
"""

import re
import time

import pytest

from ingest.bench import (
    METHODS,
    Method,
    best_of,
    cpu_prices,
    fake_tickers,
    io_benchmark,
    price_call,
)

# --- helpers ------------------------------------------------------------------------


def test_fake_tickers_are_distinct_valid_symbols() -> None:
    assert fake_tickers(3) == ["AAA", "AAB", "AAC"]
    fifty = fake_tickers(50)
    assert len(set(fifty)) == 50
    assert all(re.fullmatch(r"[A-Z]{3}", t) for t in fifty)


def test_best_of_keeps_the_fastest_run() -> None:
    """Noise only ever adds time, so the minimum is the best estimate."""
    durations = iter([0.06, 0.01, 0.04])
    fastest = best_of(lambda: time.sleep(next(durations)), repeats=3)
    assert 0.01 <= fastest < 0.03


# --- I/O-bound ------------------------------------------------------------------------


def test_concurrent_ingest_is_several_times_faster() -> None:
    """6 tickers, 50ms each: ~0.3s one at a time, ~0.05s all at once."""
    (row,) = io_benchmark((6,), concurrency=6, latency=0.05, repeats=2)
    assert row.tickers == 6
    assert row.speedup > 2.5


def test_speedup_flattens_at_the_concurrency_limit() -> None:
    """8 tickers through a semaphore of 2: never more than ~2x, however many tickers."""
    (row,) = io_benchmark((8,), concurrency=2, latency=0.05, repeats=2)
    assert 1.4 < row.speedup < 2.3


# --- CPU-bound -------------------------------------------------------------------------


def test_price_call_is_the_project_3_price() -> None:
    """Black-Scholes says 8.916. 100k paths has a standard error of ~0.045."""
    assert price_call(seed=0, n_paths=100_000) == pytest.approx(8.916, abs=4 * 0.045)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_gets_exactly_the_same_prices(method: Method) -> None:
    """Parallelism must not change the answer. Each task owns its seed, so it can't."""
    seeds = [0, 1, 2]
    expected = [price_call(seed, 2_000) for seed in seeds]
    assert cpu_prices(method, seeds, 2_000) == expected

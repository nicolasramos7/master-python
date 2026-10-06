"""Sitting 2, part 4: two benchmarks that answer "when does asyncio help?"

**I/O-bound.** Ingest N tickers against the fake vendor twice: with
`concurrency=1` (one request at a time -- sequential, through the same code)
and with `concurrency=C`. Each request spends most of its life waiting on
`latency`. Waiting is exactly what `asyncio` overlaps, so the speedup should
grow with N... until N reaches C. After that the semaphore is the bottleneck
and the speedup flattens at about C. Explaining that flattening is the point.

**CPU-bound.** Price the Project 3 option by pure-Python Monte Carlo, several
times over with different seeds, four ways: a plain loop, `asyncio.gather`,
a thread pool, and a process pool. Expect:

    sequential   1.0x
    asyncio      ~1x   a coroutine that never awaits never yields; gather runs them one by one
    threads      ~1x   the GIL: one thread runs Python bytecode at a time
    processes    ~cores  separate interpreters, separate GILs

And every way must return *identical* prices -- parallelism must not change
the answer. That is the test.

**Measuring honestly** (the Project 3 lesson again): time with
`time.perf_counter`, repeat each measurement, and keep the **minimum**. Noise
on a laptop -- another app, a GC pause, a CPU frequency change -- only ever
makes a run slower, never faster, so the minimum is the best estimate of the
code's own cost.

**Process-pool traps.** On macOS (and on Linux from Python 3.14) worker
processes are *spawned*: each one starts a fresh interpreter and imports your
module to find the function. So the function must be a top-level `def` in an
importable module -- not a lambda, not a nested function -- and the entry
point must sit behind `if __name__ == "__main__":`, or every worker re-runs
the program. `__main__.py` already has the guard. Leave it.

The Python in it: `time.perf_counter`, `functools.partial`,
`itertools.product` + `itertools.islice`, `concurrent.futures`
(`ThreadPoolExecutor`, `ProcessPoolExecutor`), `Literal` + `match` +
`assert_never` (Project 2), `tempfile.TemporaryDirectory`.
"""

# ruff: noqa: F401
# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import asyncio
import functools
import itertools
import math
import random
import string
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, assert_never

from .client import make_client
from .fake_api import API_KEY, FakeMarketAPI
from .jobs import make_jobs
from .run import run

Method = Literal["sequential", "asyncio", "threads", "processes"]
METHODS: tuple[Method, ...] = ("sequential", "asyncio", "threads", "processes")

WINDOW = (date(2024, 1, 1), date(2024, 1, 31))
"""One month per ticker: small files, so the benchmark measures waiting, not writing."""


@dataclass(frozen=True)
class IoRow:
    tickers: int
    sequential: float
    concurrent: float

    @property
    def speedup(self) -> float:
        return self.sequential / self.concurrent


@dataclass(frozen=True)
class CpuRow:
    method: Method
    seconds: float
    speedup: float
    """sequential seconds / this method's seconds."""


def fake_tickers(n: int) -> list[str]:
    """n distinct valid symbols: "AAA", "AAB", "AAC", ... No list of 50 typed by hand."""
    letters = itertools.product(string.ascii_uppercase, repeat=3)
    return ["".join(chars) for chars in itertools.islice(letters, n)]


def best_of(fn: Callable[[], object], repeats: int = 3) -> float:
    """Call `fn` `repeats` times; return the fastest wall time, in seconds.

    Time each call with `time.perf_counter()` before and after. The first call
    also pays for imports and cold caches -- one more reason to keep the min.
    """
    times = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        fn()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    return min(times)


async def _ingest(tickers: Sequence[str], concurrency: int, latency: float, out: Path) -> None:
    api = FakeMarketAPI(latency=latency)
    async with make_client(api, API_KEY) as client:
        await run(make_jobs(tickers, *WINDOW), client=client, out=out, concurrency=concurrency)


def ingest_once(tickers: Sequence[str], concurrency: int, latency: float) -> None:
    """One full ingest into a throwaway directory, so nothing is ever skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(_ingest(tickers, concurrency, latency, Path(tmp)))


def io_benchmark(
    ticker_counts: Sequence[int] = (5, 20, 50),
    *,
    concurrency: int = 10,
    latency: float = 0.05,
    repeats: int = 3,
) -> list[IoRow]:
    """Sequential vs concurrent ingest time, one row per ticker count.

    For each n: `fake_tickers(n)`, then `best_of` on `ingest_once` with
    concurrency 1 and with `concurrency`. `best_of` wants a zero-argument
    callable -- build one with `functools.partial(ingest_once, tickers, 1, latency)`.

    Why not `lambda: ingest_once(tickers, 1, latency)`? Inside a loop, a lambda
    captures the *variable* `tickers`, not its current value; call it after the
    loop moves on and it sees the new one. Here it's called immediately so it
    would work -- but ruff's B023 flags it, and `partial` binds the value now.
    """
    rows = []
    for n in ticker_counts:
        tickers = fake_tickers(n)
        sequential = best_of(functools.partial(ingest_once, tickers, 1, latency), repeats)
        concurrent = best_of(functools.partial(ingest_once, tickers, concurrency, latency), repeats)
        rows.append(IoRow(n, sequential, concurrent))
    return rows


def price_call(seed: int, n_paths: int) -> float:
    """Project 3's base-case call (S=K=100, r=2%, sigma=20%, T=1) by pure-Python Monte Carlo.

    Deliberately slow -- a Python loop, no NumPy -- because NumPy releases the
    GIL inside its C loops and would muddy the thread result. Given.
    """
    s, k, r, sigma, t = 100.0, 100.0, 0.02, 0.2, 1.0
    drift = (r - sigma**2 / 2) * t
    vol = sigma * math.sqrt(t)
    rng = random.Random(seed)
    total = 0.0
    for _ in range(n_paths):
        total += max(s * math.exp(drift + vol * rng.gauss(0.0, 1.0)) - k, 0.0)
    return math.exp(-r * t) * total / n_paths


def cpu_prices(method: Method, seeds: Sequence[int], n_paths: int) -> list[float]:
    """`price_call(seed, n_paths)` for every seed, in order, run the given way.

    `match method:` with one `case` per Method, and `case _: assert_never(method)`
    so mypy tells you if a fifth method is ever added and not handled here.

        "sequential"  a list comprehension
        "asyncio"     an `async def` that just returns price_call(...), gathered
                      over the seeds, inside one `asyncio.run`. Module-level
                      helpers, please -- you'll want to test them.
        "threads"     `with ThreadPoolExecutor(max_workers=len(seeds)) as pool:`
                      then `list(pool.map(fn, seeds))`
        "processes"   the same with ProcessPoolExecutor. `fn` must be picklable:
                      `functools.partial(price_call, n_paths=n_paths)` is;
                      a lambda is not.
    """
    work = functools.partial(price_call, n_paths=n_paths)
    match method:
        case "sequential":
            return [work(seed) for seed in seeds]
        case "asyncio":
            return asyncio.run(_gather_prices(seeds, n_paths))
        case "threads":
            with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
                return list(pool.map(work, seeds))
        case "processes":
            with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
                return list(pool.map(work, seeds))
        case _:
            assert_never(method)
        


def cpu_benchmark(n_tasks: int = 4, n_paths: int = 300_000, repeats: int = 3) -> list[CpuRow]:
    """Time `cpu_prices` each way, one row per method, speedups relative to sequential."""
    seeds = list(range(n_tasks))
    seconds = {
        method: best_of(functools.partial(cpu_prices, method, seeds, n_paths), repeats)
        for method in METHODS
    }
    return [CpuRow(m, s, seconds["sequential"] / s) for m, s in seconds.items()]

async def _price_async(seed: int, n_paths: int) -> float:
    return price_call(seed, n_paths)

async def _gather_prices(seeds: Sequence[int], n_paths: int) -> list[float]:
    return list(await asyncio.gather(*(_price_async(s, n_paths) for s in seeds)))

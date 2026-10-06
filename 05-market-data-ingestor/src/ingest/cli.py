"""Sitting 2, part 3: the command line. The only module that prints.

    python -m ingest fetch --tickers AAPL,MSFT,SPY --start 2015-01-01 --out data/
    python -m ingest bench

Two subcommands with `argparse` sub-parsers. `parse_args` turns `argv` into a
frozen dataclass -- `FetchArgs` or `BenchArgs` -- so nothing past this module
ever touches an untyped `argparse.Namespace`. `main` dispatches on it with
`match` (Project 2's Enum + match, with classes this time).

Validate at the boundary, with argparse's `type=` hook. A `type=` function
takes the raw string and returns the parsed value, or raises
`argparse.ArgumentTypeError` -- argparse turns that into a clean usage error
and exit code 2, not a traceback:

    --tickers "aapl, msft,,AAPL"  ->  ("AAPL", "MSFT")   strip, upper-case, drop blanks, dedupe
    --start / --end 2015-01-01    ->  date(2015, 1, 1)   date.fromisoformat
    --concurrency 8               ->  8                  must be >= 1
    --out data/                   ->  Path("data")

A check across two arguments (start after end) can't live in a `type=`
function; call `parser.error(...)` after parsing -- same exit code 2.

Exit codes are part of the interface. Scripts and CI read them:

    0  every job done (fetched now or already on disk)
    1  the run finished but some jobs failed -- rerun to retry just those
    2  usage or config error (bad argument, missing MARKETDATA_API_KEY)

Logging goes to stderr, configured once here with `logging.basicConfig`; the
report goes to stdout. So `python -m ingest fetch ... > report.txt` captures
the report and still shows progress. `-v` turns on DEBUG.

`make_api` and the `format_*` functions are given. `make_api` is a function,
not a constant, so the tests can monkeypatch in a zero-latency vendor.

The Python in it: `argparse` (sub-parsers, `type=`, `default=`,
`ArgumentTypeError`, `parser.error`), `match` on dataclasses, `sys.stderr`,
`asyncio.run` as the one bridge from sync to async, exit codes.
"""

# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .bench import CpuRow, IoRow, cpu_benchmark, io_benchmark
from .client import api_key_from_env, make_client
from .errors import ConfigError
from .fake_api import LAST_DATE, FakeMarketAPI
from .jobs import make_jobs
from .run import Report, run


@dataclass(frozen=True)
class FetchArgs:
    tickers: tuple[str, ...]
    start: date
    end: date
    out: Path
    concurrency: int
    verbose: bool


@dataclass(frozen=True)
class BenchArgs:
    quick: bool
    """Smaller sizes, one repeat: a smoke test of the benchmark, not a measurement."""


def make_api() -> FakeMarketAPI:
    """The vendor the CLI talks to: a little slow, a little flaky, rate-limited. Given."""
    return FakeMarketAPI(latency=0.05, fail_rate=0.05, max_concurrent=8)


def parse_tickers(raw: str) -> tuple[str, ...]:
    """ "aapl, msft,,AAPL" -> ("AAPL", "MSFT"). No tickers at all -> ArgumentTypeError."""
    raw_split = raw.split(",")
    clean = []
    for split in raw_split:
        s = split.strip()
        s = s.upper()
        if s == "" or s in clean:
            continue
        clean.append(s)
    if len(clean) == 0:
        raise argparse.ArgumentTypeError("no tickers given")
    return tuple(clean)


def positive_int(raw: str) -> int:
    """ "8" -> 8. Anything below 1, or not an integer -> ArgumentTypeError."""
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {raw!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> FetchArgs | BenchArgs:
    """Parse argv (default: sys.argv[1:]). Exits with code 2 on any usage error.

    fetch: --tickers (required), --start (required), --end (default LAST_DATE),
           --out (default "data"), --concurrency (default 8), -v/--verbose.
    bench: --quick.

    Shape: one `ArgumentParser`, `parser.add_subparsers(dest="command",
    required=True)`, then `sub.add_parser("fetch")` and `sub.add_parser("bench")`,
    each with its own `add_argument` calls. `type=` does the per-argument
    validation (`parse_tickers`, `date.fromisoformat`, `Path`, `positive_int`).
    After `parse_args(argv)`, check start <= end with `fetch_parser.error(...)`,
    and build the dataclass from the Namespace.
    """
    parser = argparse.ArgumentParser(prog="python -m ingest")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch")
    fetch.add_argument("--tickers", type=parse_tickers, required=True)
    fetch.add_argument("--start", type=date.fromisoformat, required=True)
    fetch.add_argument("--end", type=date.fromisoformat, default=LAST_DATE)
    fetch.add_argument("--out", type=Path, default=Path("data"))
    fetch.add_argument("--concurrency", type=positive_int, default=8)
    fetch.add_argument("-v", "--verbose", action="store_true")

    bench = sub.add_parser("bench")
    bench.add_argument("--quick", action="store_true")

    ns = parser.parse_args(argv)
    if ns.command == "bench":
        return BenchArgs(quick=ns.quick)
    if ns.start > ns.end:
        fetch.error(f"--start {ns.start} is after --end {ns.end}")
    return FetchArgs(
        tickers=ns.tickers,
        start=ns.start,
        end=ns.end,
        out=ns.out,
        concurrency=ns.concurrency,
        verbose=ns.verbose,
    )


def format_report(report: Report) -> str:
    """The summary printed after a fetch. Given."""
    lines = [
        f"fetched {report.fetched}, skipped {report.skipped}, "
        f"failed {len(report.failed)} in {report.seconds:.2f}s"
    ]
    lines += [f"  FAILED {job.ticker} {job.year}: {error}" for job, error in report.failed]
    return "\n".join(lines)


def format_io_table(rows: Sequence[IoRow]) -> str:
    """Given."""
    lines = [f"{'tickers':>8}{'sequential':>12}{'concurrent':>12}{'speedup':>9}"]
    lines += [
        f"{r.tickers:>8}{r.sequential:>11.3f}s{r.concurrent:>11.3f}s{r.speedup:>8.1f}x"
        for r in rows
    ]
    return "\n".join(lines)


def format_cpu_table(rows: Sequence[CpuRow]) -> str:
    """Given."""
    lines = [f"{'method':<12}{'seconds':>9}{'speedup':>9}"]
    lines += [f"{r.method:<12}{r.seconds:>8.3f}s{r.speedup:>8.1f}x" for r in rows]
    return "\n".join(lines)


async def fetch(args: FetchArgs, api: FakeMarketAPI, api_key: str) -> Report:
    """Open one client for the whole run and hand every job to `run`.

    `async with make_client(api, api_key) as client:` -- then `await run(...)` with
    `make_jobs(args.tickers, args.start, args.end)`.
    """
    async with make_client(api, api_key) as client:
        return await run(make_jobs(args.tickers, args.start, args.end), 
                         client=client, out=args.out, concurrency=args.concurrency)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, configure logging, dispatch. Returns the exit code; never calls sys.exit.

    1. `args = parse_args(argv)`.
    2. `logging.basicConfig(level=..., format="%(levelname)-7s %(name)s: %(message)s",
       stream=sys.stderr)` -- DEBUG if fetch -v, else INFO. Then quiet httpx, which
       logs every request at INFO: `logging.getLogger("httpx").setLevel(logging.WARNING)`.
    3. `match args:`
       - `case FetchArgs():` read the key (ConfigError -> print to stderr, return 2),
         `asyncio.run(fetch(args, make_api(), key))`, print `format_report`, and
         return 1 if anything failed, else 0.
       - `case BenchArgs(quick=quick):` sizes are ticker counts (2, 4), 1 repeat and
         20_000 paths when quick; (5, 20, 50), 3 repeats and 300_000 paths otherwise.
         Print both tables with a one-line title each (include `os.cpu_count()`
         above the CPU table). Return 0.
    """
    args = parse_args(argv)
    
    verbose = isinstance(args, FetchArgs) and args.verbose

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    match args:
        case FetchArgs():
            try:
                api_key = api_key_from_env()
            except ConfigError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            report = asyncio.run(fetch(args, make_api(), api_key))
            print(format_report(report))
            return 1 if report.failed else 0
        case BenchArgs(quick=quick):
            counts, repeats, n_paths = ((2, 4), 1, 20_000) if quick else ((5, 20, 50), 3, 300_000)
            print("I/O-bound: ingest, sequential vs concurrency=10")
            print(format_io_table(io_benchmark(counts, repeats=repeats)))
            print(f"\nCPU-bound: 4 Monte Carlo pricings, {os.cpu_count()} cores")
            print(format_cpu_table(cpu_benchmark(n_paths=n_paths, repeats=repeats)))
            return 0
            


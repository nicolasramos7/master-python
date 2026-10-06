"""Sitting 1, part 1: the work list, as generators. No I/O, no async -- just iteration.

A run of the ingestor is a list of *jobs*: one per (ticker, calendar year).
Each job fetches one window of bars and writes exactly one file:

    out/AAPL/2015.parquet      <- Job("AAPL", 2015-01-01, 2015-12-31)
    out/AAPL/2016.parquet
    out/SAP.DE/2015.parquet    ...

One file per job is the whole resumability design. The file existing *is* the
record that the job is done -- no state file, no database, nothing to get out
of sync. (That only works if a file can never exist half-written. That is
`storage.py`'s job.)

Three functions, each a generator exercise:

    year_windows(start, end)   ->  (2015-03-10, 2015-12-31), (2016-01-01, 2016-12-31), ...
    make_jobs(tickers, s, e)   ->  every (ticker, window) pair, via itertools.product
    pending(jobs, out)         ->  only the jobs whose file does not exist yet

The Python in it: `yield`, generator expressions, `itertools.product`,
`itertools.filterfalse`, `pathlib.Path` and its `/` operator.

**The generator trap.** A function containing `yield` runs *none* of its body
when called -- not even the first line. So `if start > end: raise ValueError`
inside a generator does not raise when the caller makes the mistake; it raises
later, wherever the generator is first iterated, possibly in a different
function entirely. `year_windows` must raise at *call* time. The fix is a plain
function that validates, then returns an inner generator.
"""

# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, order=True)
class Job:
    """One request, one file. Frozen and ordered, so jobs can go in sets and sort."""

    ticker: str
    start: date
    end: date

    @property
    def year(self) -> int:
        return self.start.year

    def path(self, out: Path) -> Path:
        """Where this job's file lives: `out / ticker / "<year>.parquet"`."""
        return out / self.ticker / f"{self.year}.parquet"


def year_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Split [start, end] into calendar-year windows, both ends inclusive.

    The first window starts at `start` and the last ends at `end`; every window
    in between is a full Jan 1 - Dec 31.

    Raises:
        ValueError: if start > end -- at CALL time, not on first iteration.
    """
    if start > end:
        raise ValueError(f"Start date {start} is after the end date {end}")

    def it():
        for year in range(start.year, end.year + 1):
            window_start = max(start, date(year, 1, 1))  # the later of the two dates
            window_end = min(end, date(year, 12, 31))  # the earlier of the two dates
            yield window_start, window_end

    return it()


def make_jobs(tickers: Iterable[str], start: date, end: date) -> Iterator[Job]:
    """Every (ticker, year window) pair, tickers in the given order, years ascending.

    Duplicate tickers produce one set of jobs, not two. (`dict.fromkeys` dedupes
    and keeps order; `set` dedupes and scrambles it.)
    """
    ts = []
    for t in tickers:
        if t not in ts:
            ts.append(t)

    windows = list(year_windows(start, end))

    pairs = itertools.product(ts, windows)

    def it():
        for t, (s, e) in pairs:
            yield Job(t, s, e)

    return it()


def pending(jobs: Iterable[Job], out: Path) -> Iterator[Job]:
    """The jobs whose file does not exist yet. Lazy: checks each file as it is reached."""

    def it():
        for j in jobs:
            if not j.path(out).exists():
                yield j

    return it()

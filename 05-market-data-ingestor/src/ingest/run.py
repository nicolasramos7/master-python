"""Sitting 2, part 2: the whole ingest, concurrent, resumable, idempotent.

    jobs --pending--> todo --gather(Semaphore)--> fetch -> to_frame -> atomic write
                        |
                        +-- already on disk: skipped, never requested

`run` takes the full job list, drops the ones already done (`pending`), and
runs the rest concurrently with `asyncio.gather`. Three things make it correct:

1. **A `Semaphore` caps concurrency.** `gather` on 50 coroutines starts all 50
   at once. The vendor allows `max_concurrent` in flight and answers the rest
   with 429. `async with sem:` around the fetch keeps at most `concurrency`
   requests open. (The retry decorator would eventually push 50 through, but
   hammering a vendor into 429s is how you get your key revoked.)

2. **`return_exceptions=True`, then sort what comes back.** Without it the
   first failed job makes `gather` raise and you lose every other result --
   and the other tasks keep running unobserved in the background. With it,
   `gather` returns a list where each item is either a result or the exception
   that job raised. Then:

       IngestError   -> expected failure: record (job, str(error)) in the report
       anything else -> a bug in your code: RAISE it. `return_exceptions=True`
                        catches TypeErrors too, and a bug written down as
                        "AAPL 2015 failed" is a bug nobody will ever look at.

3. **The write happens inside the job.** Each job fetches, converts, and writes
   its own file before it counts as done. Kill the run at any point and every
   file on disk is complete (atomic rename), so the next run's `pending`
   resumes exactly where this one stopped.

Time the run with `time.perf_counter()` -- a monotonic clock meant for
measuring intervals. (`time.time()` is wall-clock and can jump when the system
clock is adjusted.)

The Python in it: `asyncio.gather`, `asyncio.Semaphore`, `async with`, nested
`async def`, `isinstance` on exceptions, `time.perf_counter`, logging.
"""

# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from .bars import to_frame
from .client import fetch_bars
from .errors import IngestError
from .jobs import Job, pending
from .storage import atomic_write_parquet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Report:
    """What one run did. `fetched + skipped + len(failed)` == number of jobs."""

    fetched: int
    """Jobs requested and written this run."""
    skipped: int
    """Jobs whose file already existed. Never requested."""
    failed: tuple[tuple[Job, str], ...]
    """(job, error message) for every job that raised an IngestError."""
    seconds: float
    """Wall time for the run, from `time.perf_counter`."""


async def ingest_one(client: httpx.AsyncClient, job: Job, out: Path) -> None:
    """Fetch one job, convert it, write its file atomically. Logs one INFO line."""
    payload = await fetch_bars(client, job)
    frame = to_frame(payload)
    atomic_write_parquet(frame, job.path(out))
    logger.info("wrote %s (%d bars)", job.path(out), len(frame))



async def run(
    jobs: Iterable[Job],
    *,
    client: httpx.AsyncClient,
    out: Path,
    concurrency: int,
) -> Report:
    """Run every job not already on disk, at most `concurrency` at a time.

    Raises:
        Whatever non-IngestError a job raised. Those are bugs; they propagate.
    """
    start_time = time.perf_counter()
    jobs_list = list(jobs)
    to_do = list(pending(jobs_list, out))

    sem = asyncio.Semaphore(concurrency)

    async def limited(job: Job) -> None:
        async with sem:
            await ingest_one(client, job, out)
    results = await asyncio.gather( *(limited(job) for job in to_do), return_exceptions=True)

    failed: list[tuple[Job, str]] = []

    for job, result in zip(to_do, results, strict=True):
        if isinstance(result, IngestError):
            failed.append((job,str(result)))
        elif isinstance(result, BaseException):
            raise result

    return Report(
        fetched=len(to_do) - len(failed),
        skipped=len(jobs_list) - len(to_do),
        failed=tuple(failed),
        seconds=time.perf_counter() - start_time,
    )
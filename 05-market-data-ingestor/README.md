# Project 5 — Async Market Data Ingestor

**Framing question:** your nightly download died at 3am, halfway through 50
tickers. You rerun it. How do you *prove* nothing was fetched twice, nothing
was lost, and the files on disk are byte-for-byte what a clean run would have
written?

This is the generators-and-asyncio project. As in Projects 3 and 4, the domain
is kept thin: one endpoint, one kind of record (a daily OHLCV bar), and a fake
vendor that runs inside `httpx`, so there's no sign-up, no rate-limited free
tier, and no network. What you're learning is **iteration (generators, the
iterator protocol, `itertools`), `async`/`await` with real HTTP, and writing
files so that a crash can't corrupt them.**

## What you build

| Module | What it does | The Python in it |
|---|---|---|
| `fake_api.py` | **Given.** The vendor: latency, 429s, 503s, outages, a stock split, two timezones | — |
| `errors.py` | **Given.** `TransientError` vs `PermanentError` | — |
| `jobs.py` | The work list: one job per (ticker, year) | `yield`, lazy errors, `itertools.product`, `pathlib` |
| `backoff.py` | Exponential backoff with jitter, as a decorator | `__iter__`/`__next__` by hand, async decorator, `ParamSpec`, `logging` |
| `bars.py` | Vendor JSON → typed DataFrame | `TypedDict`, `cast`, `tz_localize`/`tz_convert`, `groupby().pct_change()` |
| `storage.py` | Atomic Parquet writes, read-back, fingerprints | temp-file-then-rename, `try`/re-raise, `hashlib`, `rglob` |
| `client.py` | One HTTP request, status → exception | `async def`, `httpx.AsyncClient`, exception chaining, `os.environ` |
| `run.py` | The whole ingest: concurrent, resumable, idempotent | `asyncio.gather`, `Semaphore`, `return_exceptions`, `perf_counter` |
| `bench.py` | Sequential vs async, and the GIL | `concurrent.futures`, `functools.partial`, `match` + `assert_never` |
| `cli.py` | `python -m ingest fetch …` / `bench` | `argparse` sub-parsers, `type=`, exit codes — the only module that prints |

## Development cycle — two sittings

The tests are the spec: read the test file for a module *before* you write it.
Each checkpoint means that test file is fully green.

### Sitting 1 (~3h) — iteration, and files that survive a crash

No network, and barely an event loop.

| Step | Module | Checkpoint | Time |
|---|---|---|---|
| 1 | Read `fake_api.py`'s docstring, `errors.py`, and `tests/` top to bottom. Don't code yet. | — | 15m |
| 2 | `jobs.py` | `uv run pytest tests/test_jobs.py` green (9) | 35m |
| 3 | `backoff.py`: `BackoffDelays` first, then `retry` | `tests/test_backoff.py` green (13) | 60m |
| 4 | `bars.py` | `tests/test_bars.py` green (10) | 40m |
| 5 | `storage.py` | `tests/test_storage.py` green (11) | 25m |
| 6 | Tooling loop, commit. | | 5m |

Until `retry` works, importing `client.py` fails (it applies your decorator at
import time), so `test_client`, `test_run`, `test_cli` and `test_bench` show up
as **collection errors**. That's expected — `pytest` is configured to keep
going and run everything else. They're sitting 2.

### Sitting 2 (~3h) — async, end to end

| Step | Module | Checkpoint | Time |
|---|---|---|---|
| 1 | `client.py`: `api_key_from_env`, `fetch_bars` | `tests/test_client.py` green (12) | 30m |
| 2 | `run.py`: `ingest_one`, `run` | `tests/test_run.py` green (10) | 50m |
| 3 | `cli.py`: `parse_tickers`, `positive_int`, `parse_args`, `fetch`, `main` | `tests/test_cli.py` green (13); try it for real (below) | 40m |
| 4 | `bench.py`, then `uv run python -m ingest bench` | whole suite green (87) | 30m |
| 5 | Fill in Findings, Caveats, Python I used. Learning Log row. Squash-merge. | | 25m |

**If you're over time on sitting 2:** cut the CPU half of `bench.py` to its
`sequential` and `processes` branches. Not the tests, not the README.

## Running it

```bash
uv sync
uv run pytest                      # 87 tests; 86 red until you implement
uv run mypy
uv run ruff check .

export MARKETDATA_API_KEY=demo     # the fake vendor's key. Never in code.
uv run python -m ingest fetch --tickers AAPL,MSFT,SPY,SAP.DE --start 2015-01-01 --out data/
uv run python -m ingest bench      # ~30s
```

### Try it for real

Once `cli.py` works, do these by hand. Each one is a claim the tests make,
seen from the outside.

```bash
# 1. Idempotency: rerun, and diff every byte.
shasum -a 256 data/*/*.parquet > before.txt
uv run python -m ingest fetch --tickers AAPL,MSFT,SPY,SAP.DE --start 2015-01-01 --out data/
shasum -a 256 data/*/*.parquet | diff before.txt - && echo identical

# 2. Resume: one request at a time so it's slow enough to kill. Ctrl-C halfway.
rm -rf data/
uv run python -m ingest fetch --tickers AAPL,MSFT,SPY,SAP.DE --start 2005-01-01 --concurrency 1
uv run python -m ingest fetch --tickers AAPL,MSFT,SPY,SAP.DE --start 2005-01-01   # "skipped" = work saved

# 3. Too wide: the CLI's vendor allows 8 in flight. Ask for 30.
uv run python -m ingest fetch --tickers AAPL,MSFT,SPY,SAP.DE,IBM,KO --start 2005-01-01 --concurrency 30 --out wide/
```

Watch the 429 warnings in (3). Do all the jobs make it? Rerun and see.

## Verification hooks

1. **Idempotency.** Run twice: the second run makes *zero* requests (ask the
   vendor — `api.requests` is its log) and `snapshot()` of the output is
   unchanged. Two runs from scratch, at different concurrency, are
   byte-identical. And `read_all` has zero duplicate `(ticker, date)` pairs.
2. **Resumability.** The vendor goes down after 2 of 6 jobs. The rerun makes
   exactly **4** requests — one per missing file — and the result is
   byte-identical to a run that never failed.
3. **Concurrency payoff.** `bench`: speedup grows with tickers until it hits the
   semaphore limit, then flattens. Beside it, the CPU table: `asyncio` and
   threads do nothing for a CPU-bound Python loop; processes scale with cores.
   Two benchmarks you ran yourself are the answer to "when does asyncio help?"

## The traps

All silent. All asserted in the tests, so you'll walk into them rather than
read past them.

**A generator's errors are lazy too.** `year_windows(2020, 2019)` must raise
*when called*. A function with `yield` in it doesn't run a single line until
it's iterated — including the `raise`.

**An iterator is single-use.** Build the `BackoffDelays` once at decoration
time and the first call that retries exhausts it. Every later call gets zero
retries, and nothing says so.

**Naive timestamps.** `"2015-01-02T16:00:00"` is New York time; the payload
says so in a separate field. 16:00 New York is 21:00 UTC in January and 20:00
UTC in July. "Add five hours" is right half the year.

**Adjusted vs unadjusted close.** A 7-for-1 split is an 86% one-day crash in
`close` and nothing at all in `adj_close`. `suspicious_moves` finds it — if it
groups by ticker. Without `groupby`, the boundary between two stacked tickers
is a "move" too.

**A half-written file that looks finished.** The resume logic trusts any file
that exists. Write straight to the final path, crash mid-write, and that year
is skipped forever. Temp file, then atomic rename.

**`return_exceptions=True` swallows your bugs too.** Record `IngestError`s as
failed jobs; re-raise everything else. A `TypeError` logged as "AAPL 2015
failed" is a bug nobody will ever look at.

**Blocking the loop.** A `time.sleep`, or a synchronous HTTP client, inside
`async def` quietly turns concurrent code sequential.
`test_requests_actually_overlap` is the test that notices.

**Secrets in logs.** The key goes in a header, never a URL query string — httpx
logs every request URL at INFO. `test_the_key_never_reaches_the_logs` checks
every line at every level.

## Findings

_Fill in after sitting 2._

- Paste both `bench` tables, with your machine's core count.
- Why does the I/O speedup flatten — and why does it flatten *below* 10x? (Hint:
  what does each job do after its response arrives, and where does that run?)
- Why do `asyncio` and threads both land near 1.0x on the CPU table, for two
  different reasons? What would change with NumPy instead of a Python loop?
- What happened in "try it for real" (3)?

## Caveats

_Report the number, then attack it. Some candidates:_

- The vendor is fake. `asyncio.sleep` latency is the best case for async; a real
  network adds variance, TLS setup, and connection limits.
- The current year's file is "done" as soon as it exists, so a daily run never
  picks up new bars. A real ingester needs to refetch the open partition.
- `os.replace` is atomic, but not durable: without `fsync`, a power cut can still
  lose a renamed file. Crash-safe and power-safe are different claims.
- Parsing and Parquet writes run on the event loop, one at a time.
- The fake has no holidays, and ignores `Retry-After`. Real vendors have both.

## Python I used

_What surprised you, what broke, what you'd do differently. The bugs that cost
time are the ones that didn't raise._

## Scope

Cut from the roadmap's version to keep this to two sittings: a real data
vendor (the fake stands in, as Project 4's synthetic loan book did), `--dry-run`,
and structured JSON logging. In order, if revisited:

1. **Honor `Retry-After`** on 429 instead of guessing a backoff.
2. **Refetch the open year**, so a daily cron run stays current.
3. **`--dry-run`**: print the pending jobs, request nothing. `pending` already
   does the work.
4. **`asyncio.to_thread`** for the Parquet write — then rerun the benchmark.
5. **A real vendor adapter** behind the same `fetch_bars` signature.

## Findings

Measured on an 8-core Mac, `uv run python -m ingest bench`, best of 3.

**I/O-bound: ingest, sequential vs concurrency=10**

| tickers | sequential | concurrent | speedup |
|---:|---:|---:|---:|
| 5 | 0.351s | 0.135s | 2.6x |
| 20 | 1.391s | 0.370s | 3.8x |
| 50 | 3.556s | 0.915s | 3.9x |

**CPU-bound: 4 Monte Carlo pricings (300,000 paths each)**

| method | seconds | speedup |
|---|---:|---:|
| sequential | 0.938s | 1.0x |
| asyncio | 0.916s | 1.0x |
| threads | 0.944s | 1.0x |
| processes | 1.367s | 0.7x |

- **The I/O speedup flattens at about 4x, not 10x.** Sequential costs ~71ms per
  job at every size, and only 50ms of that is the vendor's latency. The other
  ~20ms is parsing the JSON and writing the Parquet file, which runs on the
  event loop one job at a time. Concurrency overlaps the waiting but not that
  work: 50 jobs concurrently take 0.915s, about 18ms each, which is almost
  exactly the non-waiting part. So the ceiling is roughly 71 / 20 ≈ 3.5-4x,
  and the semaphore limit of 10 is never the bottleneck.
- **asyncio and threads both land on 1.0x, for different reasons.** A coroutine
  that never awaits never gives up the loop, so `gather` runs the pricings one
  after another. Threads do run "at once", but the GIL lets only one of them
  execute Python bytecode at a time.
- **Processes were slower here (0.7x), not faster.** Each pricing takes ~0.23s.
  Starting a worker process means a fresh interpreter that re-imports the
  package, and at this task size that startup most likely costs more than the
  parallelism saves. Processes are the only method that *can* use several
  cores, but they have a fixed cost that small tasks don't repay.
- **When does asyncio help?** When the time is spent waiting. It turned 3.6s of
  ingest into 0.9s and did nothing for a CPU loop.

## Caveats

- The process result is one task size on one machine. I'd expect it to flip
  above 1x with more paths per task; that needs measuring, not assuming.
- The vendor is fake. `asyncio.sleep` latency is the best case for async; a real
  network adds variance, TLS setup and connection limits.
- The current year's file counts as done as soon as it exists, so a daily run
  never picks up new bars. A real ingester has to refetch the open partition.
- `Path.replace` is atomic but not durable: without `fsync`, a power cut can
  still lose a renamed file. Crash-safe and power-safe are different claims.
- Parsing and Parquet writes run on the event loop. That is the ~20ms per job
  that caps the speedup; `asyncio.to_thread` for the write is the next thing
  to try, then rerun the benchmark.
- The fake has no holidays and ignores `Retry-After`. Real vendors have both.

## Python I used

- **Generators are lazy, including their errors.** A `raise` inside a function
  with `yield` doesn't run until the first `next()`. Validation goes in a plain
  function that returns an inner generator.
- **An iterator is single-use.** `BackoffDelays` has to be built inside the
  wrapper, per call. `next(it, None)` gives a default instead of `StopIteration`.
- **Forgetting `await` doesn't raise.** `payload = fetch_bars(...)` hands back a
  coroutine object, and the failure shows up somewhere else.
- **Passing a function vs calling it.** `best_of(ingest_once(...))` runs it once
  and passes `None`. `functools.partial(ingest_once, ...)` passes something
  callable. Same idea in `pool.map(work, seeds)`.
- **Built-ins vs methods.** `min(times)`, `len(x)`, `s in items` for plain
  Python; `.min()` is NumPy/pandas. I wrote `times.min()` and `clean.contains(s)`.
- **pandas and `Path` methods return new objects.** `frame.assign(...)` without
  using the result does nothing.
- **`return_exceptions=True` catches bugs too.** Record `IngestError`, re-raise
  everything else.
- **Exit codes and streams are the interface.** Report on stdout, logs and
  errors on stderr, `main` returns 0 / 1 / 2.
- **What cost time:** the bugs that didn't raise. A wrong `fetched` count and a
  list where a tuple was expected both ran fine and were only caught by tests.
# master-python

Project-based Python roadmap for fintech and data science. Nine projects, each ~6 hours, each shipping something that works and is tested.

Every project has a **verification hook**: one quantity computed two independent ways that must agree. A single implementation is a guess.

## Conventions

Each project lives in its own numbered directory with `src/`, `tests/`, a `pyproject.toml`, and a README covering the framing question, findings, caveats, and Python used.

Tooling is the same everywhere: `uv` for environments, `pytest` for tests, `mypy --strict` for types, `ruff` for linting. One branch per project, squash-merged to `main`.

```bash
cd NN-project-name
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```

## Learning Log

| # | Project | Date | New concepts | Verification hook | What I learned |
|---|---------|------|--------------|-------------------|----------------|
| 1 | [Double-Entry Ledger](01-double-entry-ledger/) | 2026-09-05 | Frozen dataclasses, `__post_init__`, dunders, `Decimal` vs `float`, custom exception hierarchy, circular imports, validate-then-mutate | `balance()` (sum of postings) must equal the final running total from `statement()`, and both must equal a hand-computed value | Float money fails intermittently: 0.1 + 0.1 - 0.2 passed while 0.1 + 0.2 - 0.3 failed. The bugs that cost time do not raise. |
| 2 | [Fixed Income Pricing Engine](02-fixed-income-pricing) | 2026-09-21 | Protocol vs ABC, Enum + match + assert_never, context managers (@contextmanager/yield), custom decorators, lru_cache, dataclasses.replace, layered imports | A coupon bond priced as a sum of its cashflows must equal the sum of the same cashflows priced as separate zero-coupon bonds | Verifying by self-inversion (price→ytm→price) beats verifying against theory: the round-trip passed tightly while "YTM should equal 5%" failed, because my own ACT/365.25 day count means a 5% bond doesn't price to exact par. One day-count shortcut surfaced in three places. |
| 3 | [03-options-pricing](03-options-pricing/) | 2026-09-24 | NumPy broadcasting and slicing, `numpy.random.Generator`, chunked accumulators, keyword-only parameters | Put-call parity; binomial → Black-Scholes at 5,000 steps; Monte Carlo within 4 standard errors; the 1/√N slope | Every bug was silent: `(r + σ²)/2` instead of `r + σ²/2` cost 0.1%, and only the known-value test caught it — the self-consistency tests all agreed on the wrong number. The tree converges as 1/n and Monte Carlo as 1/√n, so matching a 10,000-step tree would take ~10^10 paths. |
| 4 | Credit Risk Model | 2026-09-28 | pandas cleaning, sklearn `fit`/`transform`, `Pipeline`, `ColumnTransformer`, broadcasting, bootstrap | Leaky model (AUC 1.00) beats the oracle (0.76); honest model (0.75) can't | `Series.map` silently NaNs anything not in the dict — validate with `.isin` first |
| 5 | [Market Data Ingestor](05-market-data-ingestor/) | 2026-10-06 | Generators, iterator protocol, itertools, async/await, gather + Semaphore, async decorator, argparse sub-parsers, atomic writes | A rerun makes zero requests and the output is byte-identical. Async gave 3.9x on I/O, capped by per-job parsing and writing on the event loop; on a CPU loop asyncio and threads gave 1.0x | Processes were slower (0.7x) at this task size: worker startup outweighed the gain |

See [python-fintech-roadmap.md](python-fintech-roadmap.md) for full project specs.
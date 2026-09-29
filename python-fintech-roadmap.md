# Python Mastery Roadmap — Fintech & Data Science Track

**Baseline assumed:** Kaggle Basic Python + Pandas. You know syntax, control flow, functions, and `read_csv` / `groupby` / `merge`.

**Format:** 9 projects, each scoped to ~6 hours of focused work (two sittings of ~3 hours). Each has a **Core** (ship this or the project doesn't count) and a **Stretch** (only if Core landed early).

| # | Project | Theme |
|---|---|---|
| 1 | Double-Entry Ledger | OOP, money representation, testing |
| 2 | Fixed Income Pricing Engine | Protocols vs ABCs, decorators, context managers |
| 3 | Options Pricing: Three Ways, One Answer | NumPy vectorization, numerical methods |
| 4 | Credit Risk Model, Built Wrong Then Right | scikit-learn pipelines, leakage discipline |
| 5 | Async Market Data Ingestor | Generators, async I/O, idempotency |
| 6 | Returns, Risk & the Look-Ahead Trap | pandas at depth, risk metrics |
| 7 | Event-Driven Backtester | Architecture, testing hard things |
| 8 | The Reproducible Pipeline | Layered storage, data contracts, CI |
| 9 | Capstone: The Zetamac Performance Lab | Everything, on your own data |

---

## Principles carried over from the reference repo

1. **Verification against a known answer.** Every project has a **verification hook**: a quantity computed two independent ways that must agree.
2. **Silent bugs are the curriculum.** Every project has a **planted trap**: a specific silent wrongness it's designed to walk you into.
3. **The Caveats section.** Report the number, then attack it. Every README gets one.
4. **Compute separated from display.** Functions return values; `__main__.py` prints.

## Repository conventions

```
NN-project-name/
├── README.md              # framing question, running it, findings, caveats, Python used
├── pyproject.toml
├── src/package_name/
└── tests/test_*.py
```

- One branch per project (`NN-project-name`), squash-merged into `main`. Commit bodies explain *why*, including alternatives rejected. Nothing is pushed until you say so.
- Tooling loop before every commit: `uv run pytest`, `uv run mypy src`, `uv run ruff check .`
- After each project: fill in its README, add a row to the root Learning Log, squash-merge, branch for the next one.

---

## Project 1 — Double-Entry Ledger

**Domain:** Core banking / payments infrastructure.

**Core Python skills:** `dataclass` (including `frozen=True`), `decimal.Decimal` and integer minor units, custom exception hierarchies, `__repr__` / `__eq__` / `__hash__`, `@property`, **pytest** (fixtures, `parametrize`, `raises`), type hints, `mypy`, `ruff`, `uv`/`pyproject.toml`.

**What you build:** An account and transaction system where a transaction is a set of *postings* that must sum to zero: `Account`, `Posting`, `Transaction`, `Ledger`. Deposits, withdrawals, transfers, and a statement generator. Balances are derived by replaying the posting log, never stored and mutated.

**Verification hook:** (a) the sum of postings for an account must equal (b) the running balance from replaying the log. Globally, `sum(all postings) == 0` after every operation.

**Planted trap:** Write the first version with `float` balances. Deposit `0.1` a hundred times and assert the balance is `10.00`. Watch it fail, then redo it with `Decimal` and with integer cents, and write up which you'd choose and why.

**Second trap:** `withdraw` must reject invalid amounts, not clamp them. Clamping hands back a confident wrong answer.

**Stretch:** a `hypothesis` property test showing the zero-sum invariant holds for any random sequence of valid operations.

---

## Project 2 — Fixed Income Pricing Engine

**Domain:** Bond math, valuation, quant fundamentals.

**Core Python skills:** `ABC` vs `typing.Protocol`, polymorphism, composition over inheritance, custom decorators with `functools.wraps`, `lru_cache`, `contextlib.contextmanager`, `Enum`, operator dunders.

**What you build:** An `Instrument` interface with `price(market)`, implemented by `ZeroCouponBond`, `CouponBond`, `Annuity`, and `CashPosition`. A `MarketSnapshot` holds a valuation date and discount curve. A `Portfolio` prices heterogeneous instruments through one interface. Also YTM by root-finding, Macaulay/modified duration, and convexity. `with market.as_of(date):` freezes the valuation date inside the block, which prevents look-ahead by design.

**Verification hook:**
- A coupon bond priced whole equals its cashflows priced as separate zero-coupon bonds.
- Analytic modified duration matches bump-and-reprice.
- Price → YTM → price round-trips.

**Planted trap:** Δ too small in bump-and-reprice means floating-point cancellation; Δ too large means convexity contaminates the result. Plot the U-shaped error curve.

**Stretch:** an `FXForward` and multi-currency `Portfolio.price()`.

---

## Project 3 — Options Pricing: Three Ways, One Answer

**Domain:** Derivatives pricing.

**Core Python skills:** NumPy broadcasting and vectorization, `numpy.random.Generator`, variance reduction, chunked accumulators, keyword-only parameters, profiling with `timeit`.

**What you build:** Price European calls and puts three ways: Black-Scholes closed form, a vectorized Cox-Ross-Rubinstein binomial tree, and vectorized Monte Carlo with no Python loop over paths. Then the Greeks, analytic and by finite difference (reusing Project 2's bump-and-reprice), plus a convergence study.

**Verification hook:** put-call parity `C - P == S - K·e^(-rT)` for all three methods; binomial → Black-Scholes as steps grow; Monte Carlo within 4 standard errors, with error shrinking as 1/√N.

**Planted trap:** Measure MC error as `abs(mean(estimates) - BS)` and overshoots cancel undershoots. Fix it with `mean(abs(...))` and confirm the 1/√N slope. **Second trap:** 10M paths × 252 steps of `float64` is 20 GB, so accumulate terminal values or chunk.

**Why it matters:** A numerical answer without an error estimate isn't an answer. Known-value tests catch what self-consistency tests miss.

---

## Project 4 — Credit Risk Model, Built Wrong Then Right

**Domain:** Credit risk / fraud, the two largest DS functions in fintech.

**Core Python skills:** pandas cleaning, scikit-learn `Pipeline` and `ColumnTransformer`, a custom transformer against the `fit`/`transform` API, time-based splits, class imbalance, calibration, bootstrap.

**What you build:** A default-prediction model built **twice**. Version 1 leaks: it scales before splitting, uses random folds on time-ordered data, and includes post-outcome features. Version 2 is honest: fitting happens inside the pipeline, the split is by time, and post-outcome features are dropped. The deliverable is the writeup of the gap.

**Verification hook:** A base-rate baseline must be beaten. Leakage is proven when the leaky model beats even an oracle built on the true risk.

**Planted trap:** The default 0.5 threshold on a rare-positive dataset never predicts the positive class. Pick a threshold from an explicit cost assumption. Also: `Series.map` silently produces NaN for unmapped values, so validate first.

**Caveats section:** Bootstrap a confidence interval on the model-vs-baseline AUC gap, and check calibration with a reliability curve.

---

## Project 5 — Async Market Data Ingestor

**Domain:** Market data plumbing, which is genuinely what a junior engineer does in month one at a fintech.

**Core Python skills:** generators and `yield`, the iterator protocol, `itertools`, `async` / `await`, `asyncio.gather`, `asyncio.Semaphore` for rate limiting, `httpx.AsyncClient`, exponential backoff with jitter (as a decorator, a callback to Project 2), `argparse` or `typer`, the `logging` module, `pathlib`, environment-based secrets, `time.perf_counter` benchmarking.

**What you build:** A CLI that pulls daily OHLCV bars for a list of tickers from a free public API, writes them to partitioned Parquet, and is **resumable and idempotent**. Run it twice and nothing changes. Kill it halfway and a rerun completes the job without re-fetching what it already has.

`python -m ingest --tickers AAPL,MSFT,SPY --start 2015-01-01 --out data/`

**Verification hook:**
- **Idempotency:** run twice, diff the output. It should be byte-identical with zero duplicate `(ticker, date)` pairs. Assert this in a test.
- **Concurrency payoff:** benchmark sequential vs async across 5 / 20 / 50 tickers. Take `min()` of 3 runs. Speedup should scale until you hit the semaphore limit, then flatten. Explain the flattening.

**GIL contrast:** Run your Project 3 Monte Carlo pricer under `asyncio` (no speedup) and under `multiprocessing` (scales with cores). Put that table next to the ingest benchmark: I/O-bound → async wins; CPU-bound → only processes help. Two benchmarks you ran yourself are the best answer to "when does asyncio help?"

**Planted traps (all silent):**
- Timezone-naive timestamps. Normalize to UTC on ingest, or you'll discover the problem in Project 6 when your correlations are garbage.
- Adjusted vs unadjusted close. A 7-for-1 split looks like an 86% crash in unadjusted prices.
- Partial writes. A truncated file reads fine and is wrong. Write to a temp path and atomically rename.

**Stretch:** `--dry-run` and structured JSON logging.

---

## Project 6 — Returns, Risk & the Look-Ahead Trap

**Domain:** Quant risk analytics.

**Core Python skills:** pandas at professional level: `MultiIndex`, `align` / `reindex`, `resample`, `rolling`, `groupby().transform()`, `.shift()`, `.pct_change()`, NumPy vectorization, `np.errstate`, assertion-based data contracts, `scipy.stats` basics.

**What you build:** A returns and risk library over Project 5's data. It covers simple vs log returns, daily → weekly → monthly resampling, cross-sectional alignment across trading calendars, rolling annualized volatility, Sharpe, max drawdown and drawdown duration, historical and parametric VaR at 95%/99%, a correlation matrix, and one clean `fig`/`ax` matplotlib figure.

**Verification hook:**
- `sum(log_returns) == log(1 + total_simple_return)`.
- Compounding daily simple returns reproduces the price series to floating-point precision.
- Rolling vol × √252 ≈ direct annual std. Say what assumption makes them differ.

**Planted trap:** Use a rolling 20-day mean as a signal on the *same* bar, then shift it by one and watch every number get worse. Record both. Second trap: outer-join + forward-fill two tickers, so holidays become zero returns that drag vol down and Sharpe up.

**Caveats section:** Compute the Sharpe standard error `√((1 + 0.5·SR²)/n)` and show that a Sharpe of 1.2 over 6 months of daily data is not distinguishable from zero. This is the closed-form counterpart to Project 4's bootstrap.

---

## Project 7 — Event-Driven Backtester

**Domain:** Systematic trading.

**Core Python skills:** strategy pattern, dependency injection, `Protocol` for the strategy interface (reuse the Project 2 reasoning), generators driving an event loop, `dataclass` events, state machines, fakes/stubs/`monkeypatch`, controlling time in tests.

**What you build:** A bar-by-bar backtester made of four components that don't know about each other:
- **Data feed:** a generator yielding `Bar` events. It physically cannot yield the future.
- **Strategy:** bar in, `Order` out. Implement buy-and-hold, SMA crossover, and mean reversion.
- **Broker:** applies commission and slippage and fills at the *next* bar's open.
- **Portfolio:** tracks positions, cash, and the equity curve. Consider reusing the Project 1 ledger for cash/position postings; the zero-sum invariant becomes a free check.

Feed the equity curve into the Project 6 risk library.

**Verification hook:** Buy-and-hold with zero costs must equal the asset's raw price return to the cent.

**Planted trap:** Fill at the signal bar's close and measure the gap vs next-open fills. That gap is your look-ahead premium.

**Caveats section:** Multiple comparisons. Test 20 random entry/exit rules, take the best, and compare it to your "best" SMA pair.

---

## Project 8 — The Reproducible Pipeline

**Domain:** Data engineering, realistically 50–60% of an entry-level fintech DS job.

**Core Python skills:** plain-Python orchestration, `pyarrow` and partitioned Parquet, atomic writes, idempotent recomputation, `pydantic`/`pandera` contracts, content hashing, `tmp_path` fixtures, dependency-graph thinking, GitHub Actions CI.

**What you build:** `raw/` (Project 5 output, append-only) → `staging/` (typed, deduped, UTC, unique on `(ticker, date)`) → `analytics/` (returns, rolling metrics, Project 7 equity curves). `python -m pipeline run --layer analytics` rebuilds only stale layers, judged by content hash. Validation **halts** on violations: duplicate keys, nulls in required columns, negative volumes, gaps over 5 business days, or a >50% one-day return with no corporate-action flag.

**Build it generic:** Write the layer/contract machinery so it doesn't know it's handling stock prices. The capstone reuses it unchanged for a second dataset, which is the real test of whether the abstraction is right.

**Verification hook:** The analytics layer reproduces Project 6's in-memory results row-for-row within tolerance.

**Planted traps:** double rows from a rerun without dedup, a half-written multi-partition layer (fix by writing to a staging directory and atomically swapping), and schema drift (fix by pinning a schema at each boundary).

**Set up CI here:** `ruff`, `mypy`, and `pytest` across all projects so far. Fix whatever's red.

**Scope warning:** If you're running over, cut analytics to two datasets. Never cut validation or idempotency.

---

## Project 9 — Capstone: The Zetamac Performance Lab

**Domain:** Everything, applied to a dataset you generated yourself and understand better than anyone.

**Framing question:** *Am I actually getting better at mental arithmetic, how fast, and how much of what I see is just noise?*

You've been running 5 Zetamac trials a day. That's a small, messy, personal time series with a real learning signal and a lot of noise, which makes it a better capstone dataset than stock prices. You can't Google the answer, and every technique from Projects 1–8 has a natural job to do on it.

### The data

**Assumed raw schema. Adjust to match your file before starting:**

| Column | Type | Notes |
|---|---|---|
| `date` | date | local practice date |
| `trial` | int 1–5 | order within the day |
| `score` | int ≥ 0 | Zetamac score (default 120s, default settings) |
| `time` *(optional)* | time | start time, which enables time-of-day analysis |
| `settings` *(optional)* | str | only needed if you ever changed the duration or ranges |

That means 5 rows per day. Missed days are missing rows, not zeros.

### What you build

```
zetamac.csv → pipeline (P8) → analytics & model (P4, P6) → "strategy" backtest (P7) → API → dashboard
```

1. **Ingest (P5 patterns):** A loader that validates and appends the CSV to `raw/` idempotently. Re-importing the same file changes nothing, and a corrected row for an existing `(date, trial)` is handled explicitly (rejected or versioned; you decide and write it down). No async needed; the lesson is idempotency, not concurrency.

2. **Pipeline (P8, reused unchanged):** A new set of contracts: unique `(date, trial)`, `trial ∈ 1..5`, `score ≥ 0`, no future dates, at most 5 trials per day. A score more than ~40 away from the rolling median gets flagged, not dropped. It's either a typo or a real breakthrough, and the pipeline shouldn't decide which.

3. **Analytics (P6):** Treat your daily score as a price series.
   - Daily mean, max, and within-day std. The warm-up curve is the mean score by trial number 1→5.
   - Rolling 7- and 28-day means, and "daily return" = change in rolling mean.
   - **Drawdowns:** the longest stretch below your previous personal best, and max drawdown from peak rolling mean. A plateau is a drawdown.
   - **Volatility:** has your day-to-day consistency improved, even where the mean hasn't?

4. **Learning-curve model (P4 discipline):** Fit a learning curve (log-linear, or a power law `score = a·n^b` over cumulative trials `n`) and forecast.
   - **Time split only.** Train on the first ~80% of days and evaluate on the rest. A random split leaks the future exactly like Project 4's V1.
   - **Baseline that must be beaten:** "tomorrow = trailing 7-day mean." If the curve can't beat that, it isn't modelling learning.
   - Bootstrap the forecast error (resample *days*, not trials, since the 5 trials in a day aren't independent).

5. **Monte Carlo forecast (P3):** Simulate future daily scores from fitted trend + residual noise to answer "when do I hit 60 (or your next milestone)?" as a distribution, not a date. Report the median and a 90% interval, and check that the MC error shrinks as 1/√N.

6. **Practice "strategies" (P7, lightly):** The backtester's feed → strategy → result shape, applied to a question you can act on. Compare decision rules like *best-of-5*, *mean-of-5*, and *mean of trials 3–5 (post warm-up)* as your "daily score" metric, and ask which one tracks true skill most stably. Low noise with the same trend is the win. Keep it to one module; don't port the whole engine.

7. **API + dashboard (new):** FastAPI with ~3 Pydantic-validated endpoints: `GET /summary` (PB, streak, rolling means), `GET /forecast?target=60`, and `POST /sessions` (log today's 5 trials, which go through the same contracts). A single-page Streamlit dashboard that talks only to the API, never to the Parquet files.

8. **Ship:** `docker compose up` runs it. CI runs lint, types, tests, and a container smoke test. The README gets a stranger to a working system in under 5 minutes, and includes a bundled sample CSV so it runs without your personal data.

### Verification hooks

- **Two paths to one number:** the daily mean from the API == the mean computed directly from the raw CSV with plain pandas, for every day.
- **Conservation:** row count in `raw/` == 5 × days practiced (minus any documented partial days), and it's unchanged after re-importing the same file.
- **Known-answer model test:** generate synthetic data from a known power law plus noise and confirm the fit recovers `a` and `b` within tolerance before trusting it on your real data. This is the Project 3 lesson: self-consistency isn't enough.

### Planted traps

- **The warm-up trap:** Trial 1 is systematically lower. If you ever log fewer than 5 trials on a day, the daily *mean* drops for reasons unrelated to skill. Detect it by comparing within-trial-number means, and fix the metric rather than the data.
- **Autocorrelated trials:** Treating 5 trials/day × N days as 5N independent observations shrinks every confidence interval by ~√5. Bootstrap by day and watch your intervals widen.
- **Personal-best look-ahead:** "Days since PB" computed with a whole-series `max()` instead of `cummax()` uses future information. It's the same bug as Project 6's rolling signal, just in a costume.
- **Missing days as zeros:** Resampling to a daily calendar and filling gaps with 0 invents catastrophic sessions. Leave them missing and let rolling windows use `min_periods`.

### Caveats section (the point of the capstone)

- **Is the improvement real?** Estimate the slope with a confidence interval. Over a short window, a "+3 points" gain may not be distinguishable from noise, which is exactly Project 6's Sharpe argument applied to you.
- **Day-of-week / time-of-day effect:** You'll find one. Then compute its standard error and see if it survives. (The reference repo's "There is no Wednesday effect; there's a small sample" was written for this moment.)
- **Selection:** If you skipped practice on bad days, your data overstates your skill. Say so, even if you can't fix it.
- **Regime changes:** A new strategy (e.g., a different multiplication trick) is a structural break. A single learning curve across it is the wrong model.

### Scope warning

Two sittings only if you resist features. That means one CSV, three endpoints, one dashboard page, and one Dockerfile. Parts 5 and 6 are the first cuts if you're over time. Never cut the contracts, the time split, or the Caveats section.

---

# Pace

Assuming ~3 hours/day, 5 days/week:

| Week | Projects | Theme |
|---|---|---|
| 1 | P1, P2 | OOP, money, testing, valuation |
| 2 | P3, P4 | Numerical methods, ML discipline |
| 3 | P5, P6 | I/O, concurrency, pandas at depth |
| 4 | P7, P8 | Architecture, pipeline engineering, CI |
| 5 | P9 | Ship the Zetamac Lab |

About 5 weeks, ~55 hours, plus one unallocated slack day per week.

Keep logging Zetamac daily throughout. Every extra week of data makes the capstone's caveats more interesting and its intervals tighter.

**Scope discipline:** Ship Core, then stop. Don't add new libraries mid-project. The README is part of the deliverable. If you're 2 hours over, cut a feature, not the tests.

---

# How this maps to the hiring bar

| Question | Comes from |
|---|---|
| How do you store monetary amounts? | P1 |
| ABC vs Protocol? Inheritance vs composition? | P2, P7 |
| Why does Monte Carlo error scale as 1/√N? | P3 |
| How do you detect leakage? Your model got 99% accuracy, is that good? | P4 |
| When does `asyncio` help? What's the GIL's role? | P5 (both benchmarks) |
| How do you avoid look-ahead bias? | P6, P7, P9 |
| Walk me through your backtester's architecture. | P7 |
| Your pipeline died halfway through. What happens on rerun? | P8 |
| Tell me about an end-to-end project you own. | P9: your own data, your own question, deployed |

The capstone is also a good conversation opener. "I built a system that tracks and forecasts my own mental-math training, and here's why the day-of-week effect I found isn't real" shows quant instinct, engineering, and honesty in a single sentence.

# A note on honest self-assessment

Report the number, then attack it. The capstone is the hardest place to do this, because the data is about you and you'll want the improvement to be real. That's exactly why it's the right final exam.
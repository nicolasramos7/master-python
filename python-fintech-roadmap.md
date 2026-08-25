# Python Mastery Roadmap — Fintech & Data Science Track

**Baseline assumed:** Kaggle Basic Python + Pandas. You know syntax, control flow, functions, and `read_csv` / `groupby` / `merge`. You have never shipped a project, written a test, or debugged something that ran without erroring and gave the wrong answer.

**Format:** 9 projects. Each is scoped to ~6 hours of focused work — two sittings of ~3 hours. Each has a **Core** (ship this or the project doesn't count) and a **Stretch** (only if Core landed early). Do not do Stretch on the first pass of anything.

---

## What I took from the reference repo

The `jaumefl/python-learning` repo does four things that most learning repos don't, and they're the reason it works. Keep all four:

1. **Verification against a known answer.** From its own README: *"the simulation is only interesting once it reproduces the known answer."* Gambler's ruin prints the simulated ruin rate beside the closed-form value in the same table. Every project below has a designated **verification hook** — a quantity you can compute two independent ways that must agree. This is not optional garnish; it's the test suite before you have a test suite.

2. **Silent bugs are the curriculum.** The interesting failures in that log are the ones with no traceback: `rng.integers(0, 7)` building a seven-sided die, `return 1 - tails` instead of `n - tails`, averaging signed errors so overshoots cancel and the estimator looks 100× more accurate than it is. Each project below has a **planted trap** — a specific silent wrongness the project is designed to walk you into.

3. **The Caveats section.** Project 15 finds a 5.6-point weekday effect and then kills it with a standard-error argument. That instinct — *report the number, then attack it* — is worth more in a data science interview than any library knowledge. Every project README below gets a Caveats section, even when the answer is "no caveats, this is exact."

4. **Compute separated from display.** Recurs in projects 12 and 14 and is the reason the benchmark script was even possible. Keep functions returning values, never printing them.

## What I'm changing

The repo has real gaps for someone trying to get hired:

| Gap | Where it's fixed |
|---|---|
| Zero tests across 15 projects | pytest from Project 1, non-negotiable, every project |
| No type hints, no linting, no packaging | Project 1 sets up the toolchain once; reused everywhere |
| Money stored as `float` (project 13's bank account) | Project 1 makes this the central lesson |
| No reproducible pipeline, no data contracts | Project 7 |
| No network I/O, no concurrency | Project 3 (async) and Project 6 (CPU-bound contrast) |
| No decorators, context managers, generators, protocols | Projects 2, 3, 5 |
| No ML, and no leakage discipline | Project 8 |
| Nothing deployable, no CI, no Docker | Projects 7 and 9 |

Also: the reference repo's projects are ~1 day each and single-concept. Yours are 2 days and deliberately layer 2–3 concepts, because you're starting past the syntax stage.

---

## Repository conventions

Mirror the reference structure, upgraded:

```
NN-project-name/
├── README.md              # framing question, running it, findings, caveats, concepts
├── pyproject.toml         # from Project 1 onward
├── src/
│   └── package_name/      # small modules, one job each
└── tests/
    └── test_*.py
```

Root `README.md` keeps a **Learning Log** table, same as the reference, with one added column:

| # | Project | Date | New concepts | Verification hook | What I learned |
|---|---|---|---|---|---|

The "What I learned" column is the most valuable artifact you will produce. Write it the way the reference does — first person, specific, bug-forward, naming the thing that surprised you. An interviewer who reads it learns more about you than your code does.

**Git:** one branch per project, one PR into `main`, squash-merge. Write real PR descriptions. This costs 10 minutes per project and gives you a commit history that looks like an engineer's rather than a student's.

---

# The Projects

---

## Project 1 — Double-Entry Ledger

**Domain:** Core banking / payments infrastructure.

**Core Python skills:** `dataclass` (including `frozen=True`), `decimal.Decimal` and integer minor units, custom exception hierarchies, `__repr__` / `__eq__` / `__hash__`, `@property` and computed attributes, **pytest** (`fixtures`, `parametrize`, `raises`), type hints, `mypy`, `ruff`, `uv`/`pyproject.toml`.

**What you build:** An account and transaction system where a transaction is a set of *postings* that must sum to zero. `Account`, `Posting`, `Transaction`, `Ledger`. Deposits, withdrawals, transfers between accounts, and a statement generator. Balances are derived by replaying the posting log — never stored and mutated.

**Verification hook:** Two independent balance computations must agree — (a) sum of postings for that account, (b) the running balance from replaying the log in order. And globally: `sum(all postings) == 0` after every operation. If those disagree, you have a bug, and you'll know it without a traceback.

**Planted trap:** Write the first version with `float` balances, exactly as the reference repo's project 13 does. Then deposit `0.1` a hundred times and assert the balance is `10.00`. Watch it fail. Now do it with `Decimal`, then with integer cents, and write up which you'd choose for a real system and why. This is the single most reliable fintech screening question and you will have a scar to answer it with.

**Second trap:** Make `withdraw` reject an invalid amount rather than clamp it. The reference log already learned this in gambler's ruin — *"clamping bad input to a legal value is worse than rejecting it: it hands back a confident wrong answer"* — but re-learn it where it involves money.

**Why it matters:** Testing habits form in project 1 or they never form. Money representation is a genuine competency filter. Double-entry is the mental model behind every ledger, payment processor, and accounting system you will ever touch, and understanding *why* balances are derived rather than stored is the intuition behind event sourcing.

**Stretch:** `hypothesis` property test — for any random sequence of valid operations, the zero-sum invariant holds.

---

## Project 2 — Fixed Income Pricing Engine

**Domain:** Bond math, valuation, quant fundamentals.

**Core Python skills:** `ABC` / `@abstractmethod` vs `typing.Protocol` (nominal vs structural typing), polymorphism, composition over inheritance, writing your own decorator with `functools.wraps`, `functools.lru_cache`, `contextlib.contextmanager`, `__enter__` / `__exit__`, `Enum`, operator dunders.

**What you build:** An `Instrument` interface with `price(market)`. Implementations: `ZeroCouponBond`, `CouponBond`, `Annuity`, `CashPosition`. A `MarketSnapshot` holding a valuation date and a discount curve. A `Portfolio` that holds heterogeneous instruments and prices them all through the same interface. Yield-to-maturity by numerical root-finding, Macaulay and modified duration, convexity.

The context manager is the interesting part: `with market.as_of(date):` freezes the valuation date for everything inside the block and restores it after. Pricing anything with a date outside the snapshot raises. This is a real pattern — it's how you structurally prevent look-ahead bias, and it makes context managers feel necessary rather than syntactic.

**Verification hook:** Three of them, all exact.
- A bond priced at a yield equal to its coupon rate must price to exactly par.
- Modified duration from the analytic formula must match numerical bump-and-reprice: `(P(y-Δ) - P(y+Δ)) / (2ΔP)`, agreeing to ~4 decimals for small Δ.
- YTM solved numerically, fed back into the pricer, must return the original price.

**Planted trap:** Pick Δ too small in the bump-and-reprice and floating-point cancellation destroys the answer; pick it too large and the second-order term (convexity) contaminates it. Plot the duration error against Δ and find the U-shaped curve. This is the numerical-differentiation lesson and it generalizes to every finite-difference Greek you will ever compute.

**Why it matters:** "Explain inheritance vs composition" and "when would you use an ABC over a Protocol" are standard mid-screen questions and abstract answers are obvious. Bond math is the fintech literacy test — duration and convexity separate people who've done finance from people who've done Python. Also: `lru_cache` on a pricing function that silently caches across a changed market snapshot is a great bug to have hit once.

**Stretch:** Add an `FXForward` and make `Portfolio.price()` handle multi-currency.

---

## Project 3 — Async Market Data Ingestor

**Domain:** Market data plumbing — genuinely what a junior engineer does in month one at a fintech.

**Core Python skills:** generators and `yield`, the iterator protocol, `itertools`, `async` / `await`, `asyncio.gather`, `asyncio.Semaphore` for rate limiting, `httpx.AsyncClient`, exponential backoff with jitter (as a decorator — callback to Project 2), `argparse` or `typer`, the `logging` module, `pathlib`, environment-based secrets, `time.perf_counter` benchmarking.

**What you build:** A CLI that pulls daily OHLCV bars for a list of tickers from a public API (Yahoo/Stooq/Alpha Vantage — anything free), writes them to partitioned Parquet, and is **resumable and idempotent**: run it twice and nothing changes; kill it halfway and rerunning completes the job without re-fetching what it already has.

`python -m ingest --tickers AAPL,MSFT,SPY --start 2015-01-01 --out data/`

**Verification hook:** Two, mirroring the reference repo's benchmark script from project 14.
- **Idempotency:** run twice, diff the output. Byte-identical, zero duplicate `(ticker, date)` pairs. Assert it in a test.
- **Concurrency payoff:** benchmark the sequential version against the async version across 5 / 20 / 50 tickers. Take `min()` of 3 runs, not the mean — noise only makes runs slower. Speedup should scale with ticker count until you hit your semaphore limit, then flatten. Explain the flattening.

**Planted trap:** Three, all silent.
- Timezone-naive timestamps. A bar dated `2024-03-10` from a US source and one from a European source are not the same day. Normalize to UTC on ingest or discover this in Project 4 when your correlations are garbage.
- Adjusted vs unadjusted close. Fetch both for a stock that split, and look at the raw close on the split date. A 7-for-1 split looks like an 86% crash to any strategy reading unadjusted prices.
- Partial writes. Kill the process mid-write and you get a truncated file that reads fine and is wrong. Write to a temp path and atomically rename.

**Why it matters:** This is the async project, and asking someone to explain "when does `asyncio` help and when doesn't it" is a near-universal filter. You'll be able to answer with a benchmark you ran. The idempotency requirement is the thing that makes it real work rather than a script — every production pipeline is rerun, and pipelines that can't be safely rerun are how data engineers spend their weekends.

**Stretch:** Add a `--dry-run` flag and structured JSON logging.

---

## Project 4 — Returns, Risk & the Look-Ahead Trap

**Domain:** Quant risk analytics.

**Core Python skills:** pandas at professional level — `MultiIndex`, `align` / `reindex`, `resample`, `rolling`, `groupby().transform()`, `.shift()`, `.pct_change()`; NumPy vectorization; `np.errstate`; assertion-based data contracts; `scipy.stats` basics.

**What you build:** A returns and risk library over the data from Project 3. Simple vs log returns. Resampling daily → weekly → monthly. Cross-sectional alignment of tickers with different trading calendars. Rolling volatility, annualized. Sharpe ratio. Maximum drawdown and drawdown duration. Historical VaR and parametric VaR at 95% and 99%. A correlation matrix. One clean matplotlib figure with `fig` / `ax`, as in the reference repo's project 15.

**Verification hook:** Identities that must hold exactly.
- `sum(log_returns) == log(1 + total_simple_return)`. If this fails you've mixed the two conventions somewhere.
- Compounding daily simple returns must reproduce the price series to floating-point precision.
- Rolling volatility annualized by `√252` must match the direct annual standard deviation on a full year, approximately — and you should be able to say what assumption makes them differ.

**Planted trap:** The look-ahead bug. Compute a rolling 20-day mean and use it as a signal on the *same* bar. Then shift it by one and watch every performance number get worse. Write down both. This one bug is responsible for more fake backtest results than everything else combined, and you need to have seen the magnitude of the difference to respect it.

Second trap: aligning two tickers with an outer join and forward-filling. Now a stock that didn't trade on a holiday has a "return" of zero on that day, which drags its volatility down and its Sharpe up. Inner-join instead and note what you lost.

**Caveats section — this is the point of the project:** Compute the Sharpe ratio of a strategy over 6 months, then compute its standard error (roughly `√((1 + 0.5·SR²)/n)` for `n` periods). Discover that a Sharpe of 1.2 over 6 months of daily data is not statistically distinguishable from zero. This is exactly the move the reference repo makes in project 15 when it kills its own weekday effect — *"There is no Wednesday effect; there's a small sample"* — applied to a number the whole industry quotes without error bars.

**Why it matters:** This is the pandas fluency employers actually check, and it's well past the Kaggle course. The look-ahead lesson is the difference between a candidate who has backtested and one who has backtested *correctly*. The Sharpe standard error argument, deployed in an interview, is a genuine signal of quantitative maturity.

---

## Project 5 — Event-Driven Backtester

**Domain:** Systematic trading. The canonical quant project.

**Core Python skills:** the strategy pattern, dependency injection, `Protocol` for the strategy interface, composition, generators driving an event loop, `dataclass` events, state machines, testing hard things — fakes, stubs, `monkeypatch`, controlling time in tests.

**What you build:** A bar-by-bar backtester with three separate components that don't know about each other:
- A **data feed** — a generator yielding `Bar` events in chronological order. It physically cannot yield the future, which is what makes look-ahead structurally impossible rather than merely discouraged.
- A **strategy** — receives a bar, emits `Order` events. Implement three: buy-and-hold, SMA crossover, and a mean-reversion rule.
- A **broker** — receives orders, applies commission and slippage, emits `Fill` events at the *next* bar's open, not the current bar's close.
- A **portfolio** — receives fills, tracks positions, cash, and equity curve.

Then feed the equity curve into your Project 4 risk library. That reuse is the point.

**Verification hook:** The best one in the whole roadmap. Run buy-and-hold through the full backtester with zero commission and zero slippage. Its total return must equal the raw price return of the asset to the cent. If it doesn't, you have a plumbing bug — off-by-one in fill timing, cash accounting error, dividend handling, position sizing rounding. This single test catches almost everything.

**Planted trap:** Fill at the close of the signal bar. It's the natural thing to write and it's cheating — you decided to buy based on a price you then bought at. Compare the SMA crossover's returns under close-fill vs next-open-fill. The gap is your look-ahead premium.

**Caveats section:** Your SMA crossover will probably beat buy-and-hold on some window. Before believing it: how many parameter combinations did you try? If you tested 20 window pairs and picked the best, you should expect the best of 20 random strategies to look good too. Run that experiment — 20 random entry/exit rules, take the best — and compare. That's multiple-comparisons bias, and it's why most published trading strategies don't survive contact with live money.

**Why it matters:** Every quant/fintech interviewer has opinions about backtesters and will happily spend 20 minutes on yours. The architecture question — "why did you separate the broker from the strategy?" — is a real design conversation. And testing an event-driven system teaches you fakes and dependency injection in a context where they're obviously necessary rather than academic.

---

## Project 6 — Options Pricing: Three Ways, One Answer

**Domain:** Derivatives pricing.

**Core Python skills:** NumPy broadcasting and vectorization at scale, `numpy.random.Generator`, variance reduction (antithetic and control variates), profiling with `cProfile` and `timeit`, memory-vs-speed reasoning, `multiprocessing` for CPU-bound work.

**What you build:** Price a European call and put three independent ways:
1. **Closed form** — Black-Scholes.
2. **Binomial tree** — Cox-Ross-Rubinstein, vectorized backward induction.
3. **Monte Carlo** — vectorized GBM paths, no Python loop touching a single path.

Then Greeks: delta, gamma, vega, theta by analytic formula and again by finite difference (reusing the bump-and-reprice technique from Project 2). A convergence study for both numerical methods.

**Verification hook:** Everything must converge to the same number.
- Binomial → Black-Scholes as steps → ∞.
- Monte Carlo → Black-Scholes as paths → ∞, with error scaling as **1/√N**. The reference repo's project 11 already established this for π — *"each error divided by the next gives ~3.1 against √10 ≈ 3.16"* — so you know exactly what table to print and what number to look for. Same law, real instrument.
- **Put-call parity** as a free invariant: `C - P == S - K·e^(-rT)`. This holds for all three methods and catches sign errors instantly.

**Planted trap:** Measure your Monte Carlo error as `abs(mean(estimates) - BS)`. Overshoots cancel undershoots and you'll report an error 100× smaller than reality. The reference repo made exactly this mistake twice, in projects 11 and 14, and noted afterward: *"Knowing a lesson and applying it are apparently different skills."* Make it a third time on purpose, then fix it with `mean(abs(estimates - BS))` and confirm the 1/√N slope reappears.

**Second trap:** Memory. Ten million paths × 252 steps of `float64` is 20 GB. You need to either accumulate terminal values without storing paths, or chunk. The reference repo hit the small version of this in project 14 — *"Vectorization buys speed by spending memory, and once you're memory-bound it stops paying."* Here it's not a slowdown, it's a crash.

**Concurrency contrast:** Monte Carlo is CPU-bound. Try `asyncio` on it (your Project 3 tool) and measure — it will not help at all. Then try `multiprocessing` and watch it scale with cores. Being able to explain the GIL with two benchmarks you personally ran is a strong interview moment.

**Why it matters:** Variance reduction, convergence rates, and put-call parity are quant interview staples. More broadly: this project teaches you that a numerical answer without an error estimate is not an answer.

---

## Project 7 — The Reproducible Pipeline

**Domain:** Data engineering — realistically 50–60% of what an entry-level "data scientist" at a fintech actually does.

**Core Python skills:** pipeline orchestration in plain Python, `pyarrow` and partitioned Parquet, atomic writes and crash safety, idempotent recomputation, schema and data-contract validation (`pydantic` or `pandera`), content hashing for change detection, `pytest` with `tmp_path` fixtures, dependency-graph thinking, GitHub Actions CI.

**What you build:** A three-layer pipeline on the filesystem, each layer a directory of partitioned Parquet:

- **raw/** — exactly what Project 3 fetched, untouched, append-only, never rewritten.
- **staging/** — typed, deduplicated, timezone-normalized, with an enforced uniqueness contract on `(ticker, date)`.
- **analytics/** — derived datasets: daily returns, rolling metrics, and the positions and equity curve from Project 5's backtest runs.

A single CLI entrypoint (`python -m pipeline run --layer analytics`) that resolves which upstream layers are stale and rebuilds only those. Staleness by content hash and mtime, not by "did I remember to rerun it."

Plus a validation step that **halts the pipeline** on contract violations rather than warning: no duplicate keys, no nulls in required columns, no negative volumes, no gaps longer than 5 business days, no single-day return above 50% without a corporate-action flag. A pipeline that logs a warning and continues is a pipeline that silently ships bad numbers.

**Verification hook:** The pipeline's analytics layer must reproduce your Project 4 in-memory pandas results row-for-row within floating-point tolerance. Two independent paths to the same number is evidence; one path is a guess. Add it as a test.

**Planted trap:** Run the pipeline twice without dedup-on-write and watch your row count double while every downstream metric silently halves or doubles — no error, no traceback, just wrong. Then fix it and add a test that runs the pipeline twice and asserts row counts are stable. Same idempotency lesson as Project 3, but here it propagates through three layers before you see it, which is exactly how it happens in real life.

**Second trap:** Partial writes across a multi-file layer. Kill the process halfway through writing the analytics layer and you get three of five partitions updated — a dataset that loads fine and is internally inconsistent. Write to a staging directory and atomically swap, so a layer is either fully old or fully new.

**Third trap:** Schema drift. Add a column upstream and watch a downstream `select` silently pick up or drop it. Pin an explicit schema at each boundary and fail on mismatch.

**Why it matters:** Every production pipeline gets rerun, gets killed mid-run, and gets a schema change it wasn't expecting. Handling all three is the difference between a script and a system, and it's the actual content of most junior data work. "I built a three-layer pipeline with data contracts, atomic writes, and idempotent recomputation" is a materially different conversation than "I've used pandas."

**Set up CI here:** GitHub Actions running `ruff`, `mypy`, and `pytest` on every push, across all projects to date. If earlier projects fail, fix them. Green CI on a personal repo is rare enough among juniors to be a signal.

**Scope warning:** This is the tightest project in the roadmap. If you're running over, cut the analytics layer to two datasets. Do not cut the validation step or the idempotency work — those *are* the project.

---

## Project 8 — Credit Risk Model, Built Wrong Then Right

**Domain:** Credit risk / fraud detection — the two largest data science functions in fintech.

**Core Python skills:** scikit-learn `Pipeline` and `ColumnTransformer`, writing a custom transformer against the `fit`/`transform` API (your Project 1–2 OOP paying off), `TimeSeriesSplit` and `GroupKFold`, class imbalance handling, probability calibration, `joblib` serialization, reproducibility discipline.

**What you build:** A default-prediction model on a public lending dataset (Lending Club) or a fraud dataset (the Kaggle credit card fraud set). Build it **twice**:

**Version 1 — leaking.** Do the natural thing. Scale the features, then split. Use a random `KFold` on time-ordered data. Include features like `recoveries` or `last_payment_amount` that only exist *after* default is known. Impute missing values using the full dataset's mean. Report your AUC. It will be excellent.

**Version 2 — honest.** Fit the scaler inside the pipeline inside the fold. Split by time. Drop the post-outcome features. Report your AUC. It will be much worse.

The deliverable is the writeup of the gap.

**Verification hook:** A baseline that must be beaten. Predict the base rate for everyone and compute the log loss and PR-AUC. Any model that doesn't beat that is worthless regardless of its accuracy. On a 1%-default dataset, a model that predicts "no default" for everyone has 99% accuracy — say this in your README and never quote accuracy on imbalanced data again.

**Planted trap:** Beyond the leakage, use the default 0.5 classification threshold on a 1%-positive dataset and get a model that never predicts the positive class. Then plot precision and recall against threshold and pick one based on an actual cost assumption — what does a missed default cost vs a rejected good customer?

**Caveats section:** Your honest model beats logistic regression baseline by 0.02 AUC. Bootstrap the test set 1000 times and get a confidence interval on that gap. Is it distinguishable from zero? Also: check calibration with a reliability curve. A model with great ranking (AUC) and terrible calibration is useless for credit pricing, where you need the actual probability to set an interest rate — this distinction is a genuinely differentiating thing to understand.

**Why it matters:** "Tell me about a time your model was too good" is a question that separates people who've deployed models from people who've entered competitions. You'll have a written, quantified answer. Leakage is the number one cause of models that work in the notebook and fail in production, and every DS interviewer probes for it.

---

## Project 9 — Capstone: Ship the Whole Thing

**Domain:** Everything, under production constraints.

**Core Python skills:** src-layout packaging, configuration management, FastAPI + Pydantic v2, HTTP semantics and error handling across boundaries, Docker, dependency pinning, structured logging, documentation.

**What you build:** One deployed system that wires together everything above:

```
scheduled ingest (P3) → warehouse (P7) → risk & signals (P4, P5) → API (new) → dashboard
```

- A FastAPI service with endpoints for portfolio valuation (P2), risk metrics (P4), and backtest results (P5), all Pydantic-validated, with real status codes and error responses.
- A thin Streamlit dashboard consuming the API — not reaching into the database directly. The separation is deliberate and interviewers notice it.
- Dockerized, `docker compose up` and it runs.
- CI running lint, types, and the full test suite.
- A README that a stranger can follow to a working system in under five minutes.

**Verification hook:** A smoke test in CI that spins up the container, hits every endpoint, and asserts the responses are well-formed. Plus: the API's reported portfolio value must equal the value computed directly from your Project 2 pricing library. Same number, two paths.

**Why it matters:** This is your portfolio centerpiece and the thing you send with applications. Almost every junior candidate has notebooks. Very few have a containerized service with a test suite and CI. The gap between those two is roughly the gap between "we'll keep your résumé on file" and an onsite.

**Scope warning:** This is a 2-day project only if you resist adding features. Three endpoints, one dashboard page, one Dockerfile. The value is in it being *complete*, not in it being large.

---

# Pace and Timeline

Assuming ~3 hours/day, 5 days/week:

| Week | Projects | Theme |
|---|---|---|
| 1 | P1, P2 + slack day | OOP, money, testing, valuation |
| 2 | P3, P4 | I/O, concurrency, pandas at depth |
| 3 | P5, P6 | Architecture, numerical methods |
| 4 | P7, P8 | SQL/engineering, ML discipline |
| 5 | P9 + consolidation | Ship, then write everything up |

**About 5 weeks, ~55 hours.** That is aggressive but not fantasy, with two conditions:

- **Slack days are load-bearing.** One per week, unallocated. You will overrun somewhere. If you don't, use it to go back and improve the weakest README.
- **P7 and P9 are the ones that will overrun.** They're the least "algorithmic" and the most fiddly-tooling, which is exactly why they're valuable and exactly why they take longer than they look.

If you have 6 weeks, add a week between P6 and P7 to go back and retrofit type hints and missing tests across P1–P6, and to do a genuine refactoring pass. Code you wrote three weeks ago and now find embarrassing is the strongest evidence you're learning.

## Scope discipline

The failure mode for this roadmap is not difficulty, it's scope creep. Rules:

- **Ship Core, then stop.** Stretch goals are for when Core landed in 4 hours, not for when you're excited.
- **No new libraries mid-project.** If you find yourself evaluating three charting libraries on day 2, you've lost the thread.
- **The README is part of the deliverable, not a chore after it.** Budget 30 minutes. A project without its Findings and Caveats sections isn't finished.
- **If you're 2 hours over on day 2, cut a feature, not the tests.** The tests are the transferable skill; the feature isn't.

---

# How this maps to the entry-level hiring bar

**The screen** — Most entry-level fintech/DS screens filter on three things: can you write Python that isn't a notebook, do you know SQL, and have you ever been responsible for something being correct. P1 (tests, types, packaging), P7 (SQL, warehouse, CI), and the verification-hook discipline throughout hit all three directly.

**The technical interview** — These aren't LeetCode prep and shouldn't be. But the coding round at fintech firms is disproportionately about data structures under a domain constraint, and the design round is about tradeoffs. Specific things you'll be able to answer from experience rather than memory:

| Question | Comes from |
|---|---|
| How do you store monetary amounts? | P1 |
| ABC vs Protocol? Inheritance vs composition? | P2, P5 |
| When does `asyncio` help? What's the GIL's role? | P3 + P6 benchmarks |
| How do you avoid look-ahead bias? | P4, P5 (structurally, via the generator feed) |
| Walk me through your backtester's architecture. | P5 |
| Why does Monte Carlo error scale as 1/√N? | P6 |
| Write a query for a 30-day rolling average per ticker. | P7 |
| How do you detect leakage? | P8 |
| Your model got 99% accuracy — is that good? | P8 |

**The behavioral / project conversation** — This is where the reference repo's format is quietly the most valuable thing in this document. The Learning Log gives you a written record of specific bugs you caused, diagnosed, and fixed. "Tell me about a difficult bug" is nearly universal and most juniors answer it with something generic. You'll have fifteen candidates, each with a mechanism and a fix.

**Day-one readiness** — What junior engineers at fintechs actually do in month one: fix a pipeline that broke overnight, add a column to a report, write a test for someone else's code, investigate why two systems disagree about a number. That last one is the entire job, and the verification-hook habit — *always compute it two ways* — is the exact instinct it requires. P3's idempotency work and P7's data contracts are the other two.

**What this roadmap deliberately doesn't cover** — LeetCode-style algorithms (do these separately, ~30 min/day if your targets do algorithmic rounds), deep learning (not what entry-level fintech DS work looks like), and Kubernetes/infra beyond Docker. Adding any of them would dilute the rest.

---

# A note on honest self-assessment

The single best thing in the reference repo is its willingness to demolish its own results. Project 15 finds two effects and dissolves both: *"Two findings, both dissolved on inspection."* Project 11 catches its own error-averaging mistake and admits project 14 repeated it.

That habit is rarer and more valuable than any library on this list. It's what separates an analyst whose numbers you can trust from one whose numbers you have to check. Every Caveats section you write is practice for the moment in an interview when someone asks "how confident are you in that?" and the honest answer is "less than the number suggests, and here's why."

Carry it forward.
# Project 4 — Credit Risk, Built Wrong Then Right

**Framing question:** your model scores an AUC of 1.00 on loan defaults. How do
you prove it's cheating — without anyone telling you which column is the leak?

This is the data science project (roadmap Project 8). As in Project 3, the
domain is kept thin: one dataset, one target, one model family, and nothing
you need to look up. What you're learning is **pandas cleaning, the
scikit-learn API, and how to measure a model honestly**.

## What you build

| Module | What it does | The Python in it |
|---|---|---|
| `data.py` | **Given.** A synthetic Lending Club-style loan book with an answer key | — |
| `clean.py` | Messy strings → typed columns | `.str` accessor, `to_datetime`, `map`, `assign`, rejecting bad input |
| `features.py` | Column contract + `QuantileClipper` | Your first class written against someone else's API (`fit`/`transform`) |
| `models.py` | Time split, `Pipeline`, and the experiment twice (leaky, honest) | `ColumnTransformer`, `Pipeline`, frozen dataclass results |
| `evaluate.py` | Baseline, cost-based threshold, bootstrap CI | Broadcasting (again), `Generator`, paired resampling |
| `__main__.py` | Prints the report | Display only — no arithmetic here |

## Development cycle — two sittings

The tests are the spec: read the test file for a module *before* you write it.
Each checkpoint means that test file is fully green.

### Sitting 1 (~3h) — data in, pipeline built

| Step | Module | Checkpoint | Time |
|---|---|---|---|
| 1 | Read `data.py`'s docstring and `tests/` top to bottom. Don't code yet. | — | 15m |
| 2 | `clean.py` | `uv run pytest tests/test_clean.py` green | 45m |
| 3 | `features.py`: `add_features`, then `QuantileClipper` | `tests/test_features.py` green | 60m |
| 4 | `models.py`: `time_split`, `build_preprocessor`, `build_pipeline` | first 6 tests in `test_models.py` green | 45m |
| 5 | Commit. | | 5m |

Until `clean.py` works, almost every test **errors** rather than fails — the
shared `loans` fixture calls `clean()`. That's intended: it's why you start there.

### Sitting 2 (~3h) — the experiment, and being honest about it

| Step | Module | Checkpoint | Time |
|---|---|---|---|
| 1 | `models.py`: `honest_experiment`, `oracle`, `leaky_experiment` | `tests/test_models.py` green | 45m |
| 2 | `evaluate.py`: `score`, `baseline` | baseline tests green | 20m |
| 3 | `evaluate.py`: `expected_costs`, `best_threshold` | threshold tests green | 30m |
| 4 | `evaluate.py`: `bootstrap_auc_diff` | whole suite green | 25m |
| 5 | `__main__.py`: `format_table`; run it | the report prints | 20m |
| 6 | Fill in Findings, Caveats, Python I used. Squash-merge. | | 30m |

**If you're over time on sitting 2:** cut step 5 to a bare `print`, not the
tests and not the README. A project without Findings and Caveats isn't done.

## Running it

```bash
uv sync
uv run pytest                  # 40 tests, 38 red until you implement
uv run mypy
uv run ruff check .
uv run python -m credit_risk   # the report
```

## Verification hooks

1. **The oracle.** Every loan carries `true_pd`, the real probability that
   generated its outcome. No model can beat that — so a model that does has
   seen the answer. `test_leaky_model_beats_the_oracle` passing is the
   *proof* of leakage; `test_honest_model_cannot_beat_the_oracle` is the
   proof you removed it.
2. **The baseline.** Predict the training base rate for everyone. Its AUC is
   exactly 0.5, and its log loss has a closed form. Any model must beat it on
   log loss and PR-AUC — and it will score ~89% accuracy, which tells you
   everything about accuracy on imbalanced data.
3. **Decision theory.** For calibrated probabilities, the cost-minimising
   threshold is `cost_fp / (cost_fp + cost_fn)`. The oracle is calibrated by
   construction, so your `best_threshold` must find that number from data.

## The traps

All silent. All asserted in the tests, so you'll walk into them rather than
read past them.

**Leakage, three ways at once.** `leaky_experiment` uses columns that only
exist after the outcome, fits preprocessing on every row before splitting, and
splits at random on time-ordered data. Once it's working, find out which of
the three actually matters — the answer isn't "all equally."

**`Series.map` makes NaN out of anything it doesn't recognise.** A new
`emp_length` label becomes a NaN that looks exactly like "n/a", and then the
imputer fills it in. Nothing ever errors.

**The unseen category.** `home_ownership == "OTHER"` first appears after the
training window. A pipeline without `handle_unknown="ignore"` passes every
random-split test and crashes on the first new loan.

**The 0.5 threshold.** On an ~8%-default book, a perfectly decent model flags
almost nobody at 0.5. That isn't a bad model; it's a missing decision.

## Findings

The core result:

| Model    | AUC    |
|----------|--------|
| baseline | 0.5000 |
| honest   | 0.7506 |
| oracle   | 0.7601 |
| leaky    | 1.0000 |

The honest model lands 0.0095 below the oracle ceiling — close to the best any
model trained on application data could do. The leaky model beats the oracle
by 0.2399, which is impossible for an honest model: it proves the leak exists
without needing to know which column caused it.

Of the three leaks (post-outcome columns, preprocessing on all rows before
the split, random split on time-ordered data), the post-outcome columns
(`recoveries`, `last_pymnt_amnt`) did almost all the damage — `recoveries`
alone is 0 for every good loan and >0 for nearly every defaulted one, so it's
close to the answer restated as a feature.

## Caveats

The honest model's mean predicted probability on the test set is 7.08%, but
the actual test-period default rate is 11.05%. The model is trained only on
the past (2019–2022), where the true default rate is lower, and 2023's
downturn pushes the real rate up. The model can't know about a downturn it
hasn't seen yet — this is a real limitation of time-based validation, not a
bug: it's the honest number, and the leaky version hides exactly this problem.

At a 0.5 classification threshold, the model flags almost none of the actual
defaults, because most predicted probabilities sit well below 50% on an
~11%-default book — this is why accuracy is a useless metric here (baseline
88.95% vs honest 88.90%, despite one model doing nothing at all).

## Python I used

- `Series.map()` silently turns anything not in the dict into `NaN`. I had to
  validate with `.isin()` *before* mapping, both for `emp_length` labels and
  for `loan_status`, or a typo'd or new category would vanish into a missing
  value instead of raising.
- Writing `QuantileClipper` taught me the fit/transform split isn't a style
  choice — it's the only thing that keeps a pipeline from leaking. My first
  draft clipped to the settings (`self.lower`, `self.upper`) instead of the
  learned bounds (`self.bounds_`), which would have silently mangled every
  numeric column.
- `np.clip(...)` (and most NumPy/pandas calls) returns a new array rather
  than mutating in place — I lost a return value this way at least once.
- Broadcasting `y_score[None, :] >= thresholds[:, None]` evaluates every
  threshold against every row in one shot, no loop — the same trick as
  Project 3's Black-Scholes broadcasting, just in 2-D.
- A missing `import numpy as np` in one file caused three *unrelated-looking*
  test failures elsewhere. Different tests failing with different error
  messages was a good reminder to check imports before debugging logic.
- `mypy`'s `Any` can leak through `.sum(axis=...)` even when the code is
  correct — `np.asarray(..., dtype=np.float64)` at the return line pins the
  type back down.
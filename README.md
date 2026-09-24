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
| 3 | [03-options-pricing](03-options-pricing/) | 2026-09-24 | NumPy broadcasting and array slicing, `numpy.random.Generator`, chunked accumulators for flat memory, keyword-only parameters, narrowing types for strict mypy | Put-call parity (exact); binomial → Black-Scholes at 5,000 steps; Monte Carlo within 4 standard errors; the 1/√N slope across the convergence table | Every bug this project was silent. Parenthesising `(r + σ²)/2` instead of `r + σ²/2` cost 0.1% and would have passed every test except the one checking against a known value — parity and the convergence tests are all self-consistent, so they'd have agreed on the wrong number. Collapsing the binomial tree once instead of `steps` times returned 0.0, which reads as a plausible price. The real finding was the ratio column: the tree converges as 1/n (10× the steps, 10× the accuracy, dead on) and Monte Carlo as 1/√n, so matching a 10,000-step tree would need ~10^10 paths. Also learned that an over-wide type alias, not wrong code, is what makes strict mypy produce 56 errors from 3 lines. |

## Roadmap

1. **Double-Entry Ledger** - OOP, dataclasses, money representation, pytest
2. **Fixed Income Pricing Engine** - ABCs vs Protocols, decorators, context managers
3. **Async Market Data Ingestor** - generators, async/await, idempotency
4. Pending
5. Pending
6. Pending
7. **Data Pipeline** - layered storage, data contracts, CI
8. **Credit Risk Model** - sklearn pipelines, leakage discipline
9. **Capstone** - API, dashboard, Docker

See [python-fintech-roadmap.md](python-fintech-roadmap.md) for full project specs.
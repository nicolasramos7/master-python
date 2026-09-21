# Fixed Income Pricing Engine

**Framing question:** what is a stream of future cashflows worth today, and
how sensitive is that value to interest rates?

## What it does

Prices bonds and portfolios by discounting their cashflows to a valuation
date. One interface (`Instrument`) so a portfolio can hold a mixed bag of
instruments and price them uniformly.

- `Cashflow` — a dated amount (the atom).
- `MarketSnapshot` — a valuation date, a flat rate, a compounding convention.
- Instruments — `ZeroCouponBond`, `CouponBond`, `Annuity`, `CashPosition`,
  and a `Portfolio` that holds any of them.
- Analytics — yield to maturity (bisection), duration and convexity
  (bump-and-reprice).

## Running it
uv sync
uv run pytest
uv run mypy src tests
uv run ruff check .
uv run python analysis/delta_study.py # regenerates the plot
## Findings

- **A coupon bond is a bag of zero-coupon bonds.** Pricing it as the sum of
  its individual discounted cashflows equals pricing each as a separate zero.
  Two routes, same number.
- **Verification by self-inversion beats verification against theory.**
  `price -> ytm -> price` round-trips tightly; asserting YTM equals the
  coupon rate at par fails, because the pricer doesn't hit exact par (below).
- **`mypy --strict` caught a real bug for free:** `float ** float` is typed
  `Any` because a negative base with a fractional exponent returns `complex`.
  Switching to `math.pow` (declared `-> float`, raises on that input) made the
  guarantee real instead of asserted.
- **Finite differences have a sweet spot.** Duration error climbs above a
  bump size of ~1e-3 as the bump starts measuring curvature, not slope.
  Default of 1e-4 is safe.

## Caveats

- **Par pricing is approximate.** A 5% bond at a 5% rate should equal face
  value exactly. It prices to ~999.96. The cause is the ACT/365.25 day count:
  annual coupon periods spanning Feb 29 are 366/365.25 ≈ 1.002 years, not 1.0.
  The correct fix is a 30/360 day count, deferred as outside this project's
  scope. This same bias reappears in the YTM (~0.04% off theoretical) and as
  the noise floor of the delta study.
- **Flat rate, not a yield curve.** One rate for all maturities. Real markets
  have a different rate per maturity; a curve is the obvious next step.
- **`as_of` is a context manager where a plain method would do.** It yields a
  new snapshot and mutates nothing, so the `with` block only provides scoping.
  A `with_valuation_date()` method returning a new snapshot would be cleaner;
  the context manager was chosen to learn the pattern.
- **`Annuity` and `CouponBond` duplicate their schedule logic.** A shared
  helper is the refactor; deferred to avoid mid-project scope creep.
- **`lru_cache` caches on arguments only.** A pricing function closing over a
  `market` variable returns stale prices silently when the market changes.
  Cache only functions pure in their arguments.

## Python used

Frozen/slotted dataclasses, `Protocol` (structural typing) vs ABC, `Enum` +
`match` + `assert_never` for exhaustiveness, context managers via
`@contextmanager` and `yield`, `dataclasses.replace` for immutable updates,
custom decorators (`functools.wraps`, `*args`/`**kwargs`), `lru_cache`,
layered modules to prevent import cycles, `pytest` (`parametrize`,
`raises`, `approx`), `mypy --strict`, `ruff`.


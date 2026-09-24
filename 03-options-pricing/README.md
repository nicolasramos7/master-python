# Project 3 — One Number, Three Ways

**Framing question:** if three independent methods disagree about the price of
the same option, which one is wrong — and how would you know without being told?

This is the NumPy project. The finance is deliberately thin: one instrument, one
model, every formula written into the docstring where it is used. The skills
being built are broadcasting, array slicing, seeded random generation, and
reporting a numerical answer with an error bar attached.

## What it does

| Module | What it does | The NumPy in it |
|---|---|---|
| `black_scholes.py` | Closed form. The known answer. | Broadcasting — one call prices a grid of strikes × expiries |
| `binomial.py` | CRR tree, backward induction | Array slicing; one loop over steps, none over nodes |
| `monte_carlo.py` | Vectorized GBM, chunked | `Generator`, running accumulators, flat memory |
| `convergence.py` | Error tables for both numerical methods | Measuring error honestly |
| `__main__.py` | Prints the tables | Display only — no arithmetic here |

## Running it

```bash
uv sync
uv run pytest                       # 23 tests
uv run mypy
uv run ruff check .
uv run python -m options_pricing    # the convergence tables
```

## Verification hooks

- **Put-call parity**: `c - p == s - k·e^(-rt)`. Exact, method-independent, and
  it catches a sign error instantly.
- **Binomial → Black-Scholes** at 5,000 steps, to within 1e-3.
- **Monte Carlo within four standard errors** of the exact value. Not equality —
  a random estimator can only be tested against its own error bar. Seeded, so a
  failure reproduces.
- **1/√N**: sixteen times the paths must buy roughly four times the precision.

## Findings

Black-Scholes call, S=K=100, r=2%, σ=20%, T=1: **8.916037**.

```
                         Monte Carlo
-------------------------------------------------------------
           n     estimate   mean_abs_err  ratio  abs_mean_err
       1,000       8.7553       0.388333      -      0.127825
      10,000       8.8296       0.078369   4.96      0.022110
     100,000       8.8961       0.041401   1.89      0.016690
   1,000,000       8.9019       0.010580   3.91      0.003154

                       Binomial (CRR)
-------------------------------------------------------------
           n     estimate   mean_abs_err  ratio  abs_mean_err
          10       8.7207       0.195354      -      0.195354
         100       8.8962       0.019788   9.87      0.019788
       1,000       8.9141       0.001981   9.99      0.001981
      10,000       8.9158       0.000198  10.00      0.000198
```

**The two methods converge at completely different rates, and the ratio column
is where you see it.** Ten times the work buys the binomial tree a factor of
**10** — dead on, three rows running. It buys Monte Carlo a factor of **√10 ≈
3.16**. To match the 10,000-step tree's error of 0.0002, Monte Carlo would need
roughly 10^10 paths.

That is the honest case for closed forms and lattice methods where they exist.
Monte Carlo wins on problems the tree cannot represent at all — path dependence,
many underlyings — not on this one.

**The trap column is 3–6× smaller than the truth at every row.** `abs_mean_err`
averages signed errors first, so overshoots cancel undershoots and what survives
measures the quality of the averaging rather than the quality of the estimator.
Report it and you would claim a Monte Carlo several times more accurate than it
is, with no error and no traceback. In the binomial table the two columns are
identical, because a deterministic estimator has nothing to cancel — which is
the point of printing them side by side.

## Caveats

**The Monte Carlo ratios are noisy: 4.96, 1.89, 3.91.** They average near 3.16
but no single row is evidence of anything. `mc_convergence` defaults to 20
trials, so the measured error is itself an estimate with roughly 16% relative
noise, and a ratio divides two of those. The test suite uses 120 trials and
compares the ends of the table rather than adjacent rows, for exactly this
reason. Quoting a convergence rate from one pair of rows would be the same
mistake as quoting a Sharpe ratio without a standard error.

**The price is an estimate and quoting it to the cent is a lie.** At 1,000,000
paths the standard error is about 0.004, so the answer is 8.90 ± 0.01. Getting
the error under a cent needs ~10^8 paths; under a tenth of a cent, ~10^10. The
error falls as 1/√N, so each extra decimal place costs 100× the work. This is
why variance reduction exists, and why a Monte Carlo number without an error bar
is not an answer.

**The variance identity can lose precision.** `mc_price` computes variance as
`(total_sq - n·mean²)/(n-1)`, which subtracts two large nearly-equal numbers
when the mean is large relative to the spread. Option payoffs here are mostly
zeros with a wide spread, so it is fine. On data with a large offset it would
not be, and Welford's algorithm is the standard fix.

**Ratios divide by an error that could be zero.** Never happens with floats in
practice, so `format_table` does not guard it. Noted rather than handled.

## Python I used

The bugs were all silent. Not one raised anything that pointed at the cause.

**Parenthesising `(r + σ²)/2` instead of `r + σ²/2`** gave 8.906 against the
correct 8.916 — 0.1% off, and entirely invisible without a known-good value to
check against. It is the reason `test_base_case_matches_reference` hardcodes a
reference number instead of only asserting internal consistency: parity and the
convergence tests would all have passed on the wrong formula.

**`crr_price` collapsed the tree once instead of `steps` times.** Returned 0.0
for every step count above 1, because the bottom node of a call's payoff row is
always the deep out-of-the-money one. A zero is a plausible-looking number.

**`np.std(payoff, ddof=1)`** passed the *function* where the array belonged. The
traceback landed two frames downstream on a division, with a warning about
degrees of freedom — nothing about the argument that was actually wrong. NumPy
type errors surface late.

**The error/append block indented one level too deep**, inside the trials loop
instead of the path-counts loop. 360 rows instead of 3, and every error computed
against a half-filled array of zeros. Nothing was misaligned relative to its
neighbours — the block was just at the wrong depth. Reading top to bottom does
not catch this; asking "how many times does this run?" does.

**`npt.ArrayLike` in a strict-mypy codebase produced 56 errors from 3 lines.**
The alias covered floats, arrays, nested lists, buffers, `bytes` and `str`, so
every arithmetic operator had to be checked against every combination. Narrowing
it to `float | NDArray[float64]` cleared all 56. The count was alarming; the
*line* count — three — was what said it was one mistake. When strict mypy
explodes on arithmetic, the type is too wide, not the code too wrong.

**`kind` as a keyword-only parameter caught me from the other side.** I had
written `*` in three signatures before being on the receiving end of
`crr_price() takes 6 positional arguments but 7 were given`.

Also new: `np.exp` takes one argument and its second positional is `out`, which
is why passing an exponent there complains about return array types. `np.mean`
over `np.average` unless there are weights. `**` over `np.power` — the operator
*is* the ufunc. And `min(chunk_size, n_paths - done)` with sum-based accumulators
rather than a mean of chunk means, because the last chunk is short.

## Scope

Cut from the roadmap's version to keep this to two sittings: Greeks, variance
reduction, and the multiprocessing benchmark. In order, if revisited:

1. **Greeks by finite difference** — sweep the bump size and find the U-shaped
   error curve (too small: floating-point cancellation; too large: curvature).
2. **Antithetic variates** — price with `z` and `-z` in one batch, watch the
   standard error fall for free.
3. **`multiprocessing`** — Monte Carlo is CPU-bound, so `asyncio` does nothing
   and processes scale with cores.
"""Convergence studies. Compute here, print in `__main__.py` -- never both.

This module holds the planted trap of the project, in the open:

`Row.abs_mean_error` is `abs(mean(estimates) - exact)` -- the wrong way. Over
independent trials the overshoots cancel the undershoots, so this shrinks far
faster than the real error and flatters the estimator by a factor of ~sqrt of
the trial count.

`Row.mean_abs_error` is `mean(abs(estimates - exact))` -- the right way. This
is the one that falls as 1/sqrt(n_paths).

Print both side by side and read the two `ratio` columns. Doubling the work
should divide the real error by sqrt(2) ~ 1.41, and quadrupling it by 2.
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator

from .binomial import crr_price
from .black_scholes import call_price, put_price
from .monte_carlo import mc_price


@dataclass(frozen=True)
class Row:
    """One line of a convergence table."""

    n: int
    """Paths (Monte Carlo) or steps (binomial)."""
    estimate: float
    """A representative estimate at this n -- the first trial, say."""
    mean_abs_error: float
    """mean(abs(estimate - exact)) over trials. The honest error."""
    abs_mean_error: float
    """abs(mean(estimate) - exact) over trials. The flattering one. The trap."""


def mc_convergence(
    rng: Generator,
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    path_counts: tuple[int, ...],
    *,
    kind: str = "call",
    trials: int = 20,
) -> list[Row]:
    """Run `trials` independent Monte Carlo estimates at each n in `path_counts`.

    The exact value is Black-Scholes. Returns one Row per path count.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")

    if kind == "call":
        exact_price = call_price(s, k, r, sigma, t)
    elif kind == "put":
        exact_price = put_price(s, k, r, sigma, t)

    rows = []

    for count in path_counts:
        arr = np.zeros(trials)
        for i in range(trials):
            price = mc_price(rng, s, k, r, sigma, t, count, kind=kind)[0]
            arr[i] = price
        err = arr - exact_price
        honest = np.abs(err).mean()
        trap = abs(err.mean())

        rows.append(
            Row(
                n=count,
                estimate=float(arr[0]),
                mean_abs_error=float(honest),
                abs_mean_error=float(trap),
            )
        )
    return rows


def binomial_convergence(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    step_counts: tuple[int, ...],
    *,
    kind: str = "call",
) -> list[Row]:
    """Binomial error against Black-Scholes at each step count.

    The tree is deterministic, so there is one estimate per n and both error
    columns are equal. That contrast is the point: only a random estimator can
    be flattered by averaging signed errors.
    """
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")

    if kind == "call":
        exact_price = call_price(s, k, r, sigma, t)
    elif kind == "put":
        exact_price = put_price(s, k, r, sigma, t)

    rows = []

    for count in step_counts:
        price = crr_price(s, k, r, sigma, t, count, kind=kind)
        err = abs(price - exact_price)

        rows.append(
            Row(
                n=count,
                estimate=float(price),
                mean_abs_error=float(err),
                abs_mean_error=float(err),
            )
        )

    return rows

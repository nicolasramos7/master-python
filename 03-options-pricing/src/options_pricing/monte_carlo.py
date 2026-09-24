"""Way 3: Monte Carlo. The answer that arrives with an error bar attached.

Under geometric Brownian motion the terminal price is exact in one step -- you
do not need to simulate a path to price a European option:

    s_t = s * exp((r - sigma**2 / 2) * t + sigma * sqrt(t) * z)

where z is standard normal. Draw z with a `numpy.random.Generator`, never with
`np.random.seed` and the legacy top-level functions.

The price is `exp(-r * t) * mean(payoff)`, and its standard error is
`exp(-r * t) * std(payoff, ddof=1) / sqrt(n_paths)`. Return both. A Monte
Carlo estimate reported without its standard error is not an answer.

Memory: `n_paths` of float64 is 8 bytes each, so 1e8 paths is 800 MB for the
draws alone, and you need a second array for the payoffs. `chunk_size` exists
so that price() can run more paths than fit in memory at once -- accumulate
running sums, never a list of every payoff.
"""

import numpy as np
from numpy.random import Generator

from .types import Floats


def terminal_prices(
    rng: Generator,
    s: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
) -> Floats:
    """Draw `n_paths` terminal prices under risk-neutral GBM. Shape: (n_paths,)."""
    z = rng.standard_normal(n_paths)
    s_T = s * np.exp((r - ((sigma**2) / 2)) * t + sigma * np.sqrt(t) * z)

    return s_T


def payoff(spots: Floats, k: float, kind: str) -> Floats:
    """Undiscounted payoff at expiry: max(s - k, 0) for a call, max(k - s, 0) for a put."""
    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")
    if kind == "call":
        vals = np.maximum(spots - k, 0.0)
    else:
        vals = np.maximum(k - spots, 0.0)

    return vals


def mc_price(
    rng: Generator,
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    n_paths: int,
    *,
    kind: str = "call",
    chunk_size: int = 1_000_000,
) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    done = 0

    while done < n_paths:
        n = min(chunk_size, n_paths - done)

        payoffs = payoff(terminal_prices(rng, s, r, sigma, t, n), k, kind)

        total = total + payoffs.sum()
        total_sq = total_sq + (np.power(payoffs, 2)).sum()

        done = done + n

    mean = total / n_paths
    var = (total_sq - (n_paths * np.power(mean, 2))) / (n_paths - 1)

    price = mean * np.exp(-r * t)
    standard_error = np.exp(-r * t) * np.sqrt(var / n_paths)

    return float(price), float(standard_error)

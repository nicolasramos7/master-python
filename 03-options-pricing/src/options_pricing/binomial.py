"""Way 2: a Cox-Ross-Rubinstein tree, priced by vectorized backward induction.

This module is the NumPy slicing exercise. The tree parameters, given:

    dt = t / steps
    u  = exp(sigma * sqrt(dt))        # up factor
    d  = 1 / u                        # down factor
    p  = (exp(r * dt) - d) / (u - d)  # risk-neutral up probability

At expiry there are `steps + 1` terminal nodes. Node j (of steps + 1) has spot
`s * u**j * d**(steps - j)`. Build that whole row as one array.

Then walk backwards. Each step collapses a row of length n into a row of
length n - 1:

    values = exp(-r * dt) * (p * values[1:] + (1 - p) * values[:-1])

That single line, in a loop over `steps` (not over nodes), is the whole method.
One Python loop of length `steps` is correct here; a loop over nodes is not.
"""

import numpy as np

from .monte_carlo import payoff
from .types import Floats


def crr_price(
    s: float,
    k: float,
    r: float,
    sigma: float,
    t: float,
    steps: int,
    *,
    kind: str = "call",
) -> float:
    """Price a European option on a CRR tree.

    Args:
        s, k, r, sigma, t: As in `black_scholes.call_price`. Scalars only --
            broadcasting a tree is a stretch goal, not core.
        steps: Number of time steps in the tree.
        kind: "call" or "put". Reject anything else; do not silently default.

    Returns:
        The option price.
    """

    if kind not in ("call", "put"):
        raise ValueError(f"kind must be 'call' or 'put', got {kind}")
    # (e^(r·Δt) − d) / (u − d)
    time_step = t / steps
    up_factor = np.exp(sigma * np.sqrt(time_step))
    down_factor = 1 / up_factor

    p = (np.exp(r * time_step) - down_factor) / (up_factor - down_factor)
    disc = np.exp(-r * time_step)

    ts = terminal_spots(s, sigma, t, steps)

    vals = payoff(ts, k, kind)

    for _ in range(steps):
        vals = disc * (p * vals[1:] + (1 - p) * vals[:-1])

    return float(vals[0])


def terminal_spots(s: float, sigma: float, t: float, steps: int) -> Floats:
    """The `steps + 1` spot prices at expiry, lowest first.

    Split out from `crr_price` so you can test it on its own: the highest node
    must be `s * u**steps` and the lowest `s * d**steps`.
    """
    time_step = t / steps
    up_factor = np.exp(sigma * np.sqrt(time_step))
    down_factor = 1 / up_factor
    j = np.arange(steps + 1)  # [0, 1, 2, ..., steps]

    return np.asarray(s * up_factor**j * down_factor ** (steps - j))

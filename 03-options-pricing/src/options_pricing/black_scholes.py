"""Way 1: the closed form. This is the known answer the other two must reach.

The formula, so you do not have to go looking for it:

    d1 = (log(s / k) + (r + sigma**2 / 2) * t) / (sigma * sqrt(t))
    d2 = d1 - sigma * sqrt(t)

    call = s * Phi(d1) - k * exp(-r * t) * Phi(d2)
    put  = k * exp(-r * t) * Phi(-d2) - s * Phi(-d1)

Phi is the standard normal CDF. Use `scipy.special.ndtr`, which is vectorized;
`scipy.stats.norm.cdf` also works but is much slower in a loop-free hot path.

Every argument must broadcast. `call_price(100.0, np.arange(80, 121), 0.02,
0.2, 1.0)` should return 41 prices without a loop anywhere in this file.
"""

import numpy as np
from scipy.special import ndtr

from .types import Floats, Numeric


def call_price(s: Numeric, k: Numeric, r: Numeric, sigma: Numeric, t: Numeric) -> Floats:
    """Black-Scholes price of a European call.

    Args:
        s: Spot price. k: Strike. r: Risk-free rate, annualized, as a decimal.
        sigma: Volatility, annualized, as a decimal. t: Time to expiry in years.

    Returns:
        Price, broadcast to the common shape of the inputs.
    """
    d1, d2 = _d1_d2(s, k, r, sigma, t)

    return s * ndtr(d1) - k * np.exp(-r * t) * ndtr(d2)


def put_price(s: Numeric, k: Numeric, r: Numeric, sigma: Numeric, t: Numeric) -> Floats:
    """Black-Scholes price of a European put. Same arguments as `call_price`."""
    d1, d2 = _d1_d2(s, k, r, sigma, t)

    return k * np.exp(-r * t) * ndtr(-d2) - s * ndtr(-d1)


def _d1_d2(s: Numeric, k: Numeric, r: Numeric, sigma: Numeric, t: Numeric) -> tuple[Floats, Floats]:
    """d1 and d2, the two standardized distances that drive both prices."""
    d1_numerator = np.log(s / k) + (r + (sigma**2) / 2) * t
    d1_denominator = sigma * np.sqrt(t)
    d1 = d1_numerator / d1_denominator

    d2 = d1 - sigma * np.sqrt(t)

    return d1, d2

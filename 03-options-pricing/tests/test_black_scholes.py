"""Way 1 must be right on its own before it can referee the other two."""

import numpy as np
import pytest

from conftest import Inputs
from options_pricing.black_scholes import call_price, put_price

# Reference values for the `base` fixture, from a known-good implementation.
BASE_CALL = 8.916037278572539
BASE_PUT = 6.935904609248070


def test_base_case_matches_reference(base: Inputs) -> None:
    assert call_price(**base) == pytest.approx(BASE_CALL, rel=1e-10)
    assert put_price(**base) == pytest.approx(BASE_PUT, rel=1e-10)


def test_put_call_parity(base: Inputs) -> None:
    """c - p == s - k * exp(-r * t). Exact, and it catches every sign error."""
    call = call_price(**base)
    put = put_price(**base)
    forward = base["s"] - base["k"] * np.exp(-base["r"] * base["t"])
    assert call - put == pytest.approx(forward, abs=1e-12)


def test_broadcasts_over_strikes(base: Inputs) -> None:
    """A vector of strikes in, a vector of prices out, with no loop."""
    strikes = np.arange(80.0, 121.0)
    prices = call_price(base["s"], strikes, base["r"], base["sigma"], base["t"])
    assert prices.shape == strikes.shape
    # A call is worth less as the strike rises. Strictly, with no ties.
    assert np.all(np.diff(prices) < 0.0)


def test_broadcasts_over_two_axes(base: Inputs) -> None:
    """Strikes down, expiries across: a price surface from one call."""
    strikes = np.arange(80.0, 121.0).reshape(-1, 1)
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    prices = call_price(base["s"], strikes, base["r"], base["sigma"], expiries)
    assert prices.shape == (41, 4)


def test_deep_in_the_money_call_is_worth_the_forward() -> None:
    """Certain to be exercised, so it is worth spot minus discounted strike."""
    price = call_price(1e6, 100.0, 0.02, 0.2, 1.0)
    assert price == pytest.approx(1e6 - 100.0 * np.exp(-0.02), rel=1e-9)


def test_deep_out_of_the_money_call_is_worthless() -> None:
    assert call_price(1e-6, 100.0, 0.02, 0.2, 1.0) == pytest.approx(0.0, abs=1e-9)


# TODO (yours): a call is worth more with more time to expiry, all else equal.
# TODO (yours): a call is worth more with higher volatility.

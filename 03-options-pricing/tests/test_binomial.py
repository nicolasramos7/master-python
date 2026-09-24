"""Way 2 against way 1. The tree is the vectorization exercise; this is its proof."""

import numpy as np
import pytest

from conftest import Inputs
from options_pricing.binomial import crr_price, terminal_spots
from options_pricing.black_scholes import call_price, put_price


def test_terminal_spots_spans_the_tree(base: Inputs) -> None:
    """steps + 1 nodes, lowest first, spanning s * d**steps to s * u**steps."""
    steps = 50
    spots = terminal_spots(base["s"], base["sigma"], base["t"], steps)
    up = np.exp(base["sigma"] * np.sqrt(base["t"] / steps))
    assert spots.shape == (steps + 1,)
    assert spots[-1] == pytest.approx(base["s"] * up**steps)
    assert spots[0] == pytest.approx(base["s"] * up**-steps)
    assert np.all(np.diff(spots) > 0.0)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_converges_to_black_scholes(base: Inputs, kind: str) -> None:
    """The whole point of the project: two methods, one number."""
    exact = call_price(**base) if kind == "call" else put_price(**base)
    price = crr_price(**base, steps=5_000, kind=kind)
    assert price == pytest.approx(float(exact), abs=1e-3)


def test_error_shrinks_with_steps(base: Inputs) -> None:
    """Coarse tree, bad. Fine tree, good. If this fails the induction is wrong."""
    exact = float(call_price(**base))
    coarse = abs(crr_price(**base, steps=10) - exact)
    fine = abs(crr_price(**base, steps=1_000) - exact)
    assert fine < coarse / 5.0


def test_put_call_parity_holds_on_the_tree(base: Inputs) -> None:
    """Parity is method-independent -- it must hold at any step count."""
    call = crr_price(**base, steps=200, kind="call")
    put = crr_price(**base, steps=200, kind="put")
    forward = base["s"] - base["k"] * np.exp(-base["r"] * base["t"])
    assert call - put == pytest.approx(forward, abs=1e-9)


def test_rejects_unknown_kind(base: Inputs) -> None:
    """Reject bad input; never quietly fall back to a call. (Project 1's lesson.)"""
    with pytest.raises(ValueError):
        crr_price(**base, steps=10, kind="straddle")

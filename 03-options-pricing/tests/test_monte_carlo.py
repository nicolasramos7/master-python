"""Way 3 against way 1 -- but statistically, which is a different kind of test.

A Monte Carlo test cannot assert equality. It asserts that the estimate sits
inside its own error bar. That makes these tests a lesson in how to test a
random estimator without writing a flaky test: seed the generator, and assert
against the standard error the estimator itself reports.
"""

import numpy as np
import pytest
from numpy.random import Generator

from conftest import Inputs
from options_pricing.black_scholes import call_price, put_price
from options_pricing.monte_carlo import mc_price, payoff, terminal_prices


def test_terminal_prices_have_the_right_mean(rng: Generator, base: Inputs) -> None:
    """Risk-neutral drift: E[s_t] == s * exp(r * t), whatever the volatility."""
    spots = terminal_prices(rng, base["s"], base["r"], base["sigma"], base["t"], 2_000_000)
    assert spots.shape == (2_000_000,)
    expected = base["s"] * np.exp(base["r"] * base["t"])
    standard_error = float(spots.std(ddof=1)) / np.sqrt(spots.size)
    assert abs(float(spots.mean()) - expected) < 4.0 * standard_error


def test_payoff_is_floored_at_zero() -> None:
    spots = np.array([80.0, 100.0, 120.0])
    assert np.array_equal(payoff(spots, 100.0, "call"), np.array([0.0, 0.0, 20.0]))
    assert np.array_equal(payoff(spots, 100.0, "put"), np.array([20.0, 0.0, 0.0]))


@pytest.mark.parametrize("kind", ["call", "put"])
def test_price_within_four_standard_errors(rng: Generator, base: Inputs, kind: str) -> None:
    """Four sigma. Seeded, so this is deterministic despite being statistical."""
    exact = float(call_price(**base) if kind == "call" else put_price(**base))
    price, standard_error = mc_price(rng, **base, n_paths=500_000, kind=kind)
    assert standard_error > 0.0
    assert abs(price - exact) < 4.0 * standard_error


def test_standard_error_falls_as_one_over_sqrt_n(rng: Generator, base: Inputs) -> None:
    """100x the paths, 10x the precision. The defining property of the method."""
    _, se_small = mc_price(rng, **base, n_paths=10_000)
    _, se_large = mc_price(rng, **base, n_paths=1_000_000)
    assert se_large == pytest.approx(se_small / 10.0, rel=0.1)


def test_chunking_does_not_change_the_answer(base: Inputs) -> None:
    """Same seed, same paths, different chunk size -- the chunking is bookkeeping.

    If this fails you are accumulating wrong (a mean of means over unequal
    chunks is not the mean) or drawing a fresh generator per chunk.
    """
    from numpy.random import default_rng

    whole, se_whole = mc_price(default_rng(7), **base, n_paths=400_000, chunk_size=400_000)
    chunked, se_chunked = mc_price(default_rng(7), **base, n_paths=400_000, chunk_size=51_000)
    assert chunked == pytest.approx(whole, rel=1e-12)
    assert se_chunked == pytest.approx(se_whole, rel=1e-12)


def test_memory_stays_flat_across_chunks(rng: Generator, base: Inputs) -> None:
    """20M paths, 250k at a time. Peak memory must reflect the chunk, not the total.

    Build the whole 20M array and peak memory is 160 MB -- this fails. That is
    the point: the test is what stops you writing the version that works fine
    until the day someone asks for 1e9 paths.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        mc_price(rng, **base, n_paths=20_000_000, chunk_size=250_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 50 * 1024 * 1024


def test_put_call_parity_needs_shared_draws(rng: Generator, base: Inputs) -> None:
    """Parity is exact path by path, so one set of draws prices both sides exactly.

    Price the call and the put from the SAME terminal prices and parity holds to
    floating point. Draw twice and it only holds to within sampling error -- try
    it and note the difference in your README.
    """
    spots = terminal_prices(rng, base["s"], base["r"], base["sigma"], base["t"], 200_000)
    discount = float(np.exp(-base["r"] * base["t"]))
    call = discount * float(payoff(spots, base["k"], "call").mean())
    put = discount * float(payoff(spots, base["k"], "put").mean())
    forward = discount * (float(spots.mean()) - base["k"])
    assert call - put == pytest.approx(forward, abs=1e-9)

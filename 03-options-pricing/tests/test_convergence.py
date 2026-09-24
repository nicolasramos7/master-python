"""The planted trap, asserted. This test is the reason the project exists.

`abs_mean_error` averages signed errors, so overshoots cancel undershoots and
the number comes out far too small -- roughly sqrt(trials) times too small.
`mean_abs_error` does not. Only the second one falls as 1/sqrt(n).
"""

import pytest
from numpy.random import Generator

from conftest import Inputs
from options_pricing.convergence import binomial_convergence, mc_convergence

PATHS = (10_000, 40_000, 160_000)
TRIALS = 120
"""Enough trials that the measured error is itself stable. Too few and this
file fails at random, which is worse than no test."""


def test_honest_error_falls_as_one_over_sqrt_n(rng: Generator, base: Inputs) -> None:
    """16x the paths must buy ~4x the precision.

    The band is wide because the measured error is itself an estimate. Assert
    the law, not a number: adjacent rows are too noisy to compare, so this
    compares the ends of the table.
    """
    rows = mc_convergence(rng, **base, path_counts=PATHS, trials=TRIALS)
    assert [row.n for row in rows] == list(PATHS)
    errors = [row.mean_abs_error for row in rows]
    assert errors == sorted(errors, reverse=True)
    assert errors[0] / errors[-1] == pytest.approx(4.0, rel=0.4)


def test_the_trap_flatters_the_estimator(rng: Generator, base: Inputs) -> None:
    """Averaging signed errors reports an error several times smaller than the truth."""
    rows = mc_convergence(rng, **base, path_counts=(10_000,), trials=TRIALS)
    row = rows[0]
    assert row.abs_mean_error < row.mean_abs_error / 3.0


def test_binomial_error_columns_agree(base: Inputs) -> None:
    """One deterministic estimate per n, so there are no signed errors to cancel."""
    rows = binomial_convergence(**base, step_counts=(50, 500))
    for row in rows:
        assert row.abs_mean_error == pytest.approx(row.mean_abs_error)
    assert rows[-1].mean_abs_error < rows[0].mean_abs_error

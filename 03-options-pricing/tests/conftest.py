"""Shared fixtures. One base case, used everywhere, so numbers are comparable."""

from typing import TypedDict

import pytest
from numpy.random import Generator, default_rng


class Inputs(TypedDict):
    """The five pricing inputs.

    A TypedDict rather than a plain dict so that `crr_price(**base, steps=100)`
    type-checks: mypy can only match unpacked keys to parameters if it knows
    the keys. This is the kind of thing strict mode teaches you.
    """

    s: float
    k: float
    r: float
    sigma: float
    t: float


BASE: Inputs = {"s": 100.0, "k": 100.0, "r": 0.02, "sigma": 0.2, "t": 1.0}


@pytest.fixture
def base() -> Inputs:
    """At-the-money, one year, 20% vol. The case every test starts from."""
    return BASE.copy()


@pytest.fixture
def rng() -> Generator:
    """A seeded generator. Seeded, so a failing test fails again on rerun."""
    return default_rng(20260922)

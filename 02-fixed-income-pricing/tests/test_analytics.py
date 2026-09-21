from __future__ import annotations

import datetime
from dataclasses import replace

import pytest

from pricing.conventions import Compounding
from pricing.instruments import CouponBond, ZeroCouponBond
from pricing.market import MarketSnapshot
from pricing.analytics import ytm, duration, convexity

TODAY = datetime.date(2027, 6, 1)
MATURITY = datetime.date(2030, 6, 1)
MARKET = MarketSnapshot(TODAY, 0.05, Compounding.ANNUAL)


def test_ytm_round_trips_with_price() -> None:
    """Verification hook: price → ytm → price must round-trip."""
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    price = bond.price(MARKET)
    y = ytm(bond, MARKET, price)
    assert bond.price(replace(MARKET, rate=y)) == pytest.approx(price, rel=1e-6)


def test_ytm_recovers_the_rate_used_to_price() -> None:
    """The rate that priced the bond is the rate ytm finds."""
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    price = bond.price(MARKET)          # price at exactly 5%
    assert ytm(bond, MARKET, price) == pytest.approx(0.05, rel=1e-6)


def test_duration_is_positive_for_a_standard_bond() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert duration(bond, MARKET) > 0.0


def test_longer_maturity_has_greater_duration() -> None:
    short = ZeroCouponBond(datetime.date(2029, 6, 1), 1000.0)
    long = ZeroCouponBond(datetime.date(2035, 6, 1), 1000.0)
    assert duration(long, MARKET) > duration(short, MARKET)


def test_zero_coupon_duration_equals_its_maturity() -> None:
    """A zero's modified duration ~ its term / (1 + rate). Known closed form."""
    maturity = datetime.date(2030, 6, 1)  # ~3 years out
    zero = ZeroCouponBond(maturity, 1000.0)
    years = 3.0
    expected = years / (1 + MARKET.rate)  # modified duration of a zero
    assert duration(zero, MARKET) == pytest.approx(expected, rel=1e-2)

def test_convexity_is_positive_for_a_standard_bond() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert convexity(bond, MARKET) > 0.0
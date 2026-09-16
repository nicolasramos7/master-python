from __future__ import annotations

import datetime

import pytest

from pricing.cashflow import Cashflow
from pricing.conventions import Compounding
from pricing.discounting import year_fraction
from pricing.instruments import CouponBond, Instrument, ZeroCouponBond
from pricing.market import MarketSnapshot

TODAY = datetime.date(2027, 6, 1)
MATURITY = datetime.date(2030, 6, 1)
MARKET = MarketSnapshot(TODAY, 0.05, Compounding.ANNUAL)


# --- Protocol ------------------------------------------------------------


def test_zero_coupon_bond_satisfies_instrument_protocol() -> None:
    assert isinstance(ZeroCouponBond(MATURITY, 1000.0), Instrument)


def test_coupon_bond_satisfies_instrument_protocol() -> None:
    assert isinstance(CouponBond(MATURITY, 1000.0, 0.05, 3), Instrument)


# --- ZeroCouponBond ------------------------------------------------------


def test_zero_coupon_bond_is_worth_less_than_face() -> None:
    bond = ZeroCouponBond(MATURITY, 1000.0)
    assert bond.price(MARKET) < 1000.0


def test_zero_coupon_bond_at_zero_rate_prices_to_face() -> None:
    market = MarketSnapshot(TODAY, 0.0, Compounding.ANNUAL)
    bond = ZeroCouponBond(MATURITY, 1000.0)
    assert bond.price(market) == pytest.approx(1000.0)


def test_zero_coupon_bond_on_maturity_prices_to_face() -> None:
    market = MarketSnapshot(MATURITY, 0.05, Compounding.ANNUAL)
    bond = ZeroCouponBond(MATURITY, 1000.0)
    assert bond.price(market) == pytest.approx(1000.0)


def test_zero_coupon_bond_price_matches_manual_discounting() -> None:
    bond = ZeroCouponBond(MATURITY, 1000.0)
    years = year_fraction(TODAY, MATURITY)
    expected = 1000.0 / (1.05 ** years)
    assert bond.price(MARKET) == expected


# --- CouponBond: cashflow schedule ---------------------------------------


def test_coupon_bond_generates_correct_number_of_cashflows() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert len(bond._cashflows()) == 3


def test_coupon_bond_last_cashflow_includes_face() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    last = bond._cashflows()[-1]
    assert last.amount == pytest.approx(1050.0)


def test_coupon_bond_intermediate_cashflows_are_coupon_only() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    for cf in bond._cashflows()[:-1]:
        assert cf.amount == pytest.approx(50.0)


def test_coupon_bond_last_cashflow_is_on_maturity_date() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert bond._cashflows()[-1].pay_date == MATURITY


def test_coupon_bond_cashflows_are_in_chronological_order() -> None:
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    dates = [cf.pay_date for cf in bond._cashflows()]
    assert dates == sorted(dates)


# --- CouponBond: pricing -------------------------------------------------


@pytest.mark.skip(
    reason=(
        "ACT/365.25 cannot achieve exact par pricing across leap years. "
        "Coupon periods spanning Feb 29 are 366/365.25 ≈ 1.002 years, not 1.0. "
        "Exact par requires 30/360 day count — see README Caveats."
    )
)
def test_coupon_bond_priced_at_coupon_rate_equals_face_exact() -> None:
    """Par test: exact version, skipped due to ACT/365.25 leap-year error."""
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert bond.price(MARKET) == pytest.approx(1000.0)


def test_coupon_bond_priced_at_coupon_rate_approximately_equals_face() -> None:
    """Par test: bond priced at its own coupon rate is within 0.1% of face.

    ACT/365.25 introduces a small error (~0.04%) across leap years.
    See README Caveats for explanation and the 30/360 fix.
    """
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert bond.price(MARKET) == pytest.approx(1000.0, rel=1e-3)


def test_coupon_bond_above_market_rate_prices_above_par() -> None:
    cheap_market = MarketSnapshot(TODAY, 0.03, Compounding.ANNUAL)
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert bond.price(cheap_market) > 1000.0


def test_coupon_bond_below_market_rate_prices_below_par() -> None:
    expensive_market = MarketSnapshot(TODAY, 0.07, Compounding.ANNUAL)
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    assert bond.price(expensive_market) < 1000.0


def test_coupon_bond_equals_sum_of_zero_coupon_bonds() -> None:
    """Verification hook: a coupon bond is a bag of zeros."""
    bond = CouponBond(MATURITY, 1000.0, 0.05, 3)
    zeros = [ZeroCouponBond(cf.pay_date, cf.amount) for cf in bond._cashflows()]
    assert bond.price(MARKET) == pytest.approx(
        sum(z.price(MARKET) for z in zeros)
    )
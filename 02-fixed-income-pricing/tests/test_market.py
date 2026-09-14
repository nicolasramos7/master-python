import datetime
from dataclasses import FrozenInstanceError

import pytest

from pricing.cashflow import Cashflow
from pricing.conventions import Compounding
from pricing.discounting import discount_factor, year_fraction
from pricing.errors import PastDatedCashflow, PricingError
from pricing.market import MarketSnapshot

TODAY = datetime.date(2027, 6, 1)
YESTERDAY = datetime.date(2027, 5, 31)
TOMORROW = datetime.date(2027, 6, 2)
YEAR_LATER = datetime.date(2028, 5, 31)

RATE = 0.05


def make_market(
    rate: float = RATE,
    compounding: Compounding = Compounding.ANNUAL,
) -> MarketSnapshot:
    return MarketSnapshot(TODAY, rate, compounding)


@pytest.mark.parametrize("compounding", list(Compounding))
def test_cashflow_today_is_worth_exactly_its_amount(compounding: Compounding) -> None:
    market = make_market(compounding=compounding)
    assert market.present_value(Cashflow(TODAY, 100.0)) == 100.0


def test_future_cashflow_is_worth_less_at_a_positive_rate() -> None:
    market = make_market()
    assert market.present_value(Cashflow(YEAR_LATER, 100.0)) < 100.0


def test_future_cashflow_is_worth_more_at_a_negative_rate() -> None:
    market = make_market(rate=-0.005)
    assert market.present_value(Cashflow(YEAR_LATER, 100.0)) > 100.0


def test_zero_rate_leaves_future_cashflow_unchanged() -> None:
    market = make_market(rate=0.0)
    assert market.present_value(Cashflow(YEAR_LATER, 100.0)) == 100.0


def test_value_shrinks_as_the_cashflow_moves_further_out() -> None:
    market = make_market()
    near = market.present_value(Cashflow(TOMORROW, 100.0))
    far = market.present_value(Cashflow(YEAR_LATER, 100.0))
    assert far < near


def test_negative_cashflow_discounts_to_a_negative_value() -> None:
    market = make_market()
    assert market.present_value(Cashflow(YEAR_LATER, -100.0)) < 0.0


def test_zero_cashflow_is_worth_zero() -> None:
    market = make_market()
    assert market.present_value(Cashflow(YEAR_LATER, 0.0)) == 0.0


def test_value_scales_linearly_with_amount() -> None:
    market = make_market()
    one = market.present_value(Cashflow(YEAR_LATER, 100.0))
    two = market.present_value(Cashflow(YEAR_LATER, 200.0))
    assert two == pytest.approx(2 * one)


@pytest.mark.parametrize("compounding", list(Compounding))
def test_present_value_matches_manual_composition(compounding: Compounding) -> None:
    market = make_market(compounding=compounding)
    cf = Cashflow(YEAR_LATER, 100.0)

    years = year_fraction(TODAY, YEAR_LATER)
    expected = 100.0 * discount_factor(RATE, years, compounding)

    assert market.present_value(cf) == expected


def test_one_year_annual_matches_independent_formula() -> None:
    market = make_market()
    value = market.present_value(Cashflow(YEAR_LATER, 100.0))
    assert value == pytest.approx(100.0 / 1.05)


def test_cashflow_before_valuation_date_is_rejected() -> None:
    market = make_market()

    with pytest.raises(PastDatedCashflow) as excinfo:
        market.present_value(Cashflow(YESTERDAY, 100.0))

    assert excinfo.value.pay_date == YESTERDAY
    assert excinfo.value.valuation_date == TODAY


def test_cashflow_on_the_valuation_date_is_accepted() -> None:
    market = make_market()
    assert market.present_value(Cashflow(TODAY, 100.0)) == 100.0


def test_past_dated_cashflow_is_a_pricing_error() -> None:
    assert issubclass(PastDatedCashflow, PricingError)


def test_snapshot_is_frozen() -> None:
    market = make_market()
    with pytest.raises(FrozenInstanceError):
        market.rate = 0.10

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


# as_of tests:


MID_YEAR = datetime.date(2027, 12, 1)


def test_as_of_yields_a_snapshot_with_the_new_valuation_date() -> None:
    market = make_market()
    with market.as_of(MID_YEAR) as shifted:
        assert shifted.valuation_date == MID_YEAR


def test_as_of_carries_over_the_other_fields() -> None:
    market = make_market(rate=0.07, compounding=Compounding.CONTINUOUS)
    with market.as_of(MID_YEAR) as shifted:
        assert shifted.rate == 0.07
        assert shifted.compounding is Compounding.CONTINUOUS


def test_as_of_leaves_the_original_untouched() -> None:
    market = make_market()
    with market.as_of(MID_YEAR):
        pass
    assert market.valuation_date == TODAY


def test_original_is_untouched_even_if_the_block_raises() -> None:
    market = make_market()

    with pytest.raises(ZeroDivisionError):
        with market.as_of(MID_YEAR):
            raise ZeroDivisionError

    assert market.valuation_date == TODAY


def test_shifted_snapshot_does_not_leak_out_of_the_block() -> None:
    market = make_market()
    with market.as_of(MID_YEAR) as shifted:
        pass
    assert shifted is not market
    assert market.valuation_date == TODAY


def test_moving_the_valuation_date_closer_raises_the_value() -> None:
    market = make_market()
    cf = Cashflow(YEAR_LATER, 100.0)

    today_value = market.present_value(cf)
    with market.as_of(MID_YEAR) as shifted:
        later_value = shifted.present_value(cf)

    assert today_value < later_value < 100.0


def test_valuing_on_the_pay_date_gives_the_full_amount() -> None:
    market = make_market()
    cf = Cashflow(YEAR_LATER, 100.0)

    with market.as_of(YEAR_LATER) as shifted:
        assert shifted.present_value(cf) == 100.0


def test_as_of_matches_constructing_the_snapshot_directly() -> None:
    market = make_market()
    cf = Cashflow(YEAR_LATER, 100.0)
    direct = MarketSnapshot(MID_YEAR, RATE, Compounding.ANNUAL)

    with market.as_of(MID_YEAR) as shifted:
        assert shifted.present_value(cf) == direct.present_value(cf)


def test_cashflow_already_paid_is_rejected_inside_the_block() -> None:
    market = make_market()
    cf = Cashflow(TOMORROW, 100.0)

    assert market.present_value(cf) > 0.0  # fine from today

    with market.as_of(YEAR_LATER) as shifted:
        with pytest.raises(PastDatedCashflow):
            shifted.present_value(cf)


def test_nested_blocks_each_yield_their_own_snapshot() -> None:
    market = make_market()

    with market.as_of(MID_YEAR) as outer:
        with outer.as_of(YEAR_LATER) as inner:
            assert inner.valuation_date == YEAR_LATER
        assert outer.valuation_date == MID_YEAR

    assert market.valuation_date == TODAY

import datetime
import math

import pytest

from pricing.conventions import Compounding
from pricing.discounting import discount_factor, year_fraction
from pricing.errors import InvalidRate, InvalidTerm, PricingError

DAY = datetime.date(2027, 6, 1)
YEAR_LATER = datetime.date(2028, 5, 31)  # exactly 365 days after DAY
EARLIER = datetime.date(2026, 6, 1)


def test_same_date_is_zero_years() -> None:
    assert year_fraction(DAY, DAY) == 0.0


def test_three_hundred_sixty_five_days_is_approximately_one_year() -> None:
    assert year_fraction(DAY, YEAR_LATER) == pytest.approx(1.0, rel=1e-3)


def test_reversed_dates_give_negative_years() -> None:
    assert year_fraction(DAY, EARLIER) < 0.0


def test_year_fraction_is_antisymmetric() -> None:
    assert year_fraction(DAY, YEAR_LATER) == -year_fraction(YEAR_LATER, DAY)


@pytest.mark.parametrize("compounding", list(Compounding))
def test_zero_years_gives_factor_of_exactly_one(compounding: Compounding) -> None:
    assert discount_factor(0.05, 0.0, compounding) == 1.0


@pytest.mark.parametrize("compounding", list(Compounding))
def test_zero_rate_gives_factor_of_exactly_one(compounding: Compounding) -> None:
    assert discount_factor(0.0, 7.5, compounding) == 1.0


@pytest.mark.parametrize("compounding", list(Compounding))
def test_positive_rate_discounts_future_money(compounding: Compounding) -> None:
    factor = discount_factor(0.05, 2.0, compounding)
    assert 0.0 < factor < 1.0


@pytest.mark.parametrize("compounding", list(Compounding))
def test_factor_shrinks_as_maturity_lengthens(compounding: Compounding) -> None:
    near = discount_factor(0.05, 1.0, compounding)
    far = discount_factor(0.05, 10.0, compounding)
    assert far < near


def test_negative_rate_above_minus_one_inflates_value() -> None:
    assert discount_factor(-0.005, 1.0, Compounding.ANNUAL) > 1.0


def test_negative_years_compounds_forward() -> None:
    assert discount_factor(0.05, -1.0, Compounding.ANNUAL) == pytest.approx(1.05)


def test_discounting_then_compounding_round_trips() -> None:
    rate, years = 0.05, 3.0
    factor = discount_factor(rate, years, Compounding.ANNUAL)
    assert factor * (1 + rate) ** years == pytest.approx(1.0)


def test_annual_factor_matches_hand_computed_value() -> None:
    assert discount_factor(0.05, 1.0, Compounding.ANNUAL) == pytest.approx(1 / 1.05)


def test_continuous_factor_matches_hand_computed_value() -> None:
    assert discount_factor(0.05, 1.0, Compounding.CONTINUOUS) == pytest.approx(math.exp(-0.05))


def test_semiannual_equals_two_half_year_annual_steps() -> None:
    semi = discount_factor(0.05, 1.0, Compounding.SEMIANNUAL)
    assert semi == pytest.approx(1 / (1.025 * 1.025))


def test_more_frequent_compounding_discounts_harder() -> None:
    annual = discount_factor(0.05, 1.0, Compounding.ANNUAL)
    semi = discount_factor(0.05, 1.0, Compounding.SEMIANNUAL)
    cont = discount_factor(0.05, 1.0, Compounding.CONTINUOUS)

    assert annual > semi > cont


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf"), -1.0, -1.5, -100.0])
def test_unusable_rates_rejected(rate: float) -> None:
    with pytest.raises(InvalidRate) as excinfo:
        discount_factor(rate, 1.0, Compounding.ANNUAL)

    assert excinfo.value.rate == pytest.approx(rate, nan_ok=True)


@pytest.mark.parametrize("years", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_terms_rejected(years: float) -> None:
    with pytest.raises(InvalidTerm) as excinfo:
        discount_factor(0.05, years, Compounding.ANNUAL)

    assert not math.isfinite(excinfo.value.years)


def test_rate_errors_are_pricing_errors() -> None:
    assert issubclass(InvalidRate, PricingError)
    assert issubclass(InvalidTerm, PricingError)

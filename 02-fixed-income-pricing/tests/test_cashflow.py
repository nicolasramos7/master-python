import datetime
import math
from dataclasses import FrozenInstanceError

import pytest

from pricing.cashflow import Cashflow
from pricing.errors import InvalidCashflow, PricingError

SAMPLE_DATE = datetime.date(2027, 6, 1)
OTHER_DATE = datetime.date(2028, 6, 1)


@pytest.mark.parametrize("amount", [0.0, -4.0, 4.0, 1e10])
def test_valid_amounts_construct(amount: float) -> None:
    cf = Cashflow(SAMPLE_DATE, amount)
    assert cf.amount == amount
    assert cf.pay_date == SAMPLE_DATE


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_amounts_rejected(amount: float) -> None:
    with pytest.raises(InvalidCashflow) as excinfo:
        Cashflow(SAMPLE_DATE, amount)

    assert not math.isfinite(excinfo.value.amount)
    assert excinfo.value.pay_date == SAMPLE_DATE


def test_invalid_cashflow_is_a_pricing_error() -> None:
    assert issubclass(InvalidCashflow, PricingError)


def test_instance_is_frozen() -> None:
    cf = Cashflow(SAMPLE_DATE, 100.0)
    with pytest.raises(FrozenInstanceError):
        cf.amount = 200.0  # type: ignore[misc]


def test_equality_and_hashing() -> None:
    c1 = Cashflow(SAMPLE_DATE, 100.0)
    c2 = Cashflow(SAMPLE_DATE, 100.0)
    c3 = Cashflow(OTHER_DATE, 100.0)
    c4 = Cashflow(SAMPLE_DATE, 200.0)

    assert c1 == c2
    assert c1 != c3
    assert c1 != c4
    assert hash(c1) == hash(c2)
    assert len({c1, c2}) == 1

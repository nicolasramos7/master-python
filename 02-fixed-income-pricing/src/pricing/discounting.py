import datetime
import math
from typing import assert_never

from pricing.conventions import Compounding
from pricing.errors import InvalidRate, InvalidTerm


def year_fraction(start_date: datetime.date, end_date: datetime.date) -> float:
    days = end_date - start_date
    return days.days / 365.25


def discount_factor(rate: float, years: float, compounding: Compounding) -> float:
    if not math.isfinite(rate) or rate <= -1.0:
        raise InvalidRate(rate)
    if not math.isfinite(years):
        raise InvalidTerm(years)

    match compounding:
        case Compounding.ANNUAL:
            return math.pow(1 + rate, -years)
        case Compounding.SEMIANNUAL:
            return math.pow(1 + rate / 2, -2 * years)
        case Compounding.CONTINUOUS:
            return math.exp(-rate * years)
        case _:
            assert_never(compounding)

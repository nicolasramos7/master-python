import datetime
import math
from dataclasses import dataclass

from pricing.errors import InvalidCashflow


@dataclass(frozen=True, slots=True)
class Cashflow:
    pay_date: datetime.date
    amount: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):  # both a nan and infinity check
            raise InvalidCashflow(self.amount, self.pay_date)

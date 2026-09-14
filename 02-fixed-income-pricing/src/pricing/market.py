import datetime
from dataclasses import dataclass

from pricing.cashflow import Cashflow
from pricing.conventions import Compounding
from pricing.discounting import discount_factor, year_fraction
from pricing.errors import PastDatedCashflow


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    valuation_date: datetime.date
    rate: float
    compounding: Compounding

    def present_value(self, cashflow: Cashflow) -> float:
        if cashflow.pay_date < self.valuation_date:
            raise PastDatedCashflow(cashflow.pay_date, self.valuation_date)

        years = year_fraction(self.valuation_date, cashflow.pay_date)
        factor = discount_factor(self.rate, years, self.compounding)
        return cashflow.amount * factor

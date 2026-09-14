import datetime
import math


class PricingError(Exception):
    """Base for everything this library raises."""


class InvalidCashflow(PricingError):
    def __init__(self, amount: float, pay_date: datetime.date) -> None:
        super().__init__(
            f"Cashflow on {pay_date.isoformat()}: amount must be finite, got {amount!r}"
        )
        self.amount = amount
        self.pay_date = pay_date


class InvalidRate(PricingError):
    """A discount rate was outside the meaningful or representable range."""

    def __init__(self, rate: float) -> None:
        if not math.isfinite(rate):
            reason = "must be finite"
        else:
            reason = "must be greater than -1.0 (-100%)"

        super().__init__(f"Invalid rate {rate!r}: {reason}")
        self.rate = rate


class InvalidTerm(PricingError):
    """A time-to-maturity was not a usable number."""

    def __init__(self, years: float) -> None:
        super().__init__(f"Invalid term {years!r}: must be finite")
        self.years = years

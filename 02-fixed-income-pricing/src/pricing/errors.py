import datetime


class PricingError(Exception):
    """Base for everything this library raises."""


class InvalidCashflow(PricingError):
    def __init__(self, amount: float, pay_date: datetime.date) -> None:
        super().__init__(
            f"Cashflow on {pay_date.isoformat()}: amount must be finite, got {amount!r}"
        )
        self.amount = amount
        self.pay_date = pay_date

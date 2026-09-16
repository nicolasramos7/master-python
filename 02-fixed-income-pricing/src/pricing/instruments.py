from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from dateutil.relativedelta import relativedelta

from pricing.cashflow import Cashflow
from pricing.market import MarketSnapshot


@runtime_checkable
class Instrument(Protocol):
    def price(self, market: MarketSnapshot) -> float: ...


@dataclass(frozen=True, slots=True)
class ZeroCouponBond:
    maturity_date: datetime.date
    face: float

    def price(self, market: MarketSnapshot) -> float:
        cf = Cashflow(self.maturity_date, self.face)
        return market.present_value(cf)


@dataclass(frozen=True, slots=True)
class CouponBond:
    maturity_date: datetime.date
    face: float
    coupon_rate: float
    n_periods: int

    def _cashflows(self) -> list[Cashflow]:
        cs = []

        for i in range(1, self.n_periods + 1):
            d = self.maturity_date - relativedelta(years=(self.n_periods - i))
            if i == self.n_periods:
                a = self.face * self.coupon_rate + self.face
                cs.append(Cashflow(d, a))
            else:
                a = self.face * self.coupon_rate
                cs.append(Cashflow(d, a))

        return cs

    def price(self, market: MarketSnapshot) -> float:
        return sum(market.present_value(cf) for cf in self._cashflows())

@dataclass(frozen=True, slots=True)
class Annuity:
    maturity_date: datetime.date
    face: float
    coupon_rate: float
    n_periods: int

    def _cashflows(self) -> list[Cashflow]:
        cs = []
        
        for i in range(1, self.n_periods + 1):
            d = self.maturity_date - relativedelta(years=(self.n_periods - i))
            a = self.face * self.coupon_rate
            cs.append(Cashflow(d, a))
        return cs

    def price(self, market: MarketSnapshot) -> float:
        return sum(market.present_value(cf) for cf in self._cashflows())

@dataclass(frozen=True, slots=True)
class CashPosition:
    amount: float

    def price(self, market: MarketSnapshot) -> float:
        return market.present_value(Cashflow(market.valuation_date, self.amount))

@dataclass
class Portfolio:
    instruments: list[Instrument]

    def price(self, market: MarketSnapshot) -> float:
        return sum(inst.price(market) for inst in self.instruments)
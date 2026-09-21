import datetime
from functools import lru_cache

from pricing.conventions import Compounding
from pricing.instruments import ZeroCouponBond
from pricing.market import MarketSnapshot

bond = ZeroCouponBond(datetime.date(2030, 6, 1), 1000.0)
market = MarketSnapshot(datetime.date(2027, 6, 1), 0.05, Compounding.ANNUAL)


@lru_cache
def price_it(instrument: ZeroCouponBond) -> float:
    return instrument.price(market)   # ← reads `market` from outside, not an argument

print(price_it(bond))   # computes at 5%

market = MarketSnapshot(datetime.date(2027, 6, 1), 0.10, Compounding.ANNUAL)  # rate now 10%

print(price_it(bond))   # SAME bond → cache hit → still the 5% price. Wrong.
from pricing.instruments import Instrument
from pricing.market import MarketSnapshot
from dataclasses import replace


def ytm(
    instrument: Instrument,
    market: MarketSnapshot,
    target_price: float,
    tol: float = 1e-8,
    max_iter: int = 1000,
) -> float:
    """Find the rate where instrument prices to target_price, by bisection."""
    lo, hi = -0.9, 10.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if abs(hi - lo) < tol:
            return mid
        p = instrument.price(replace(market, rate=mid))
        if p > target_price:
            lo = mid
        else:
            hi = mid

    raise ValueError(f"YTM did not converge after {max_iter} iterations")

def duration(instrument: Instrument, market: MarketSnapshot, delta: float = 1e-4) -> float:
    p = instrument.price(market)
    p_up = instrument.price(replace(market, rate=market.rate + delta))
    p_dn = instrument.price(replace(market, rate=market.rate - delta))
    return (p_dn - p_up) / (2 * delta * p)


def convexity(instrument: Instrument, market: MarketSnapshot, delta: float = 1e-4) -> float:
    p = instrument.price(market)
    p_up = instrument.price(replace(market, rate=market.rate + delta))
    p_dn = instrument.price(replace(market, rate=market.rate - delta))
    return (p_up + p_dn - 2 * p) / (delta ** 2 * p)
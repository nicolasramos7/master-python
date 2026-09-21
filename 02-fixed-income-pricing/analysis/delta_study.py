import datetime
import numpy as np
import matplotlib.pyplot as plt

from pricing.conventions import Compounding
from pricing.instruments import ZeroCouponBond
from pricing.market import MarketSnapshot
from pricing.analytics import duration
from pathlib import Path

market = MarketSnapshot(datetime.date(2027, 6, 1), 0.05, Compounding.ANNUAL)
zero = ZeroCouponBond(datetime.date(2030, 6, 1), 1000.0)
true_duration = 3.0 / (1 + 0.05)

deltas = np.logspace(-10, -1, 50)
errors = [abs(duration(zero, market, delta=d) - true_duration) for d in deltas]

plt.loglog(deltas, errors)
plt.xlabel("Δ (bump size)")
plt.ylabel("absolute error in duration")
plt.title("Finite-difference error vs bump size")
Path("analysis").mkdir(exist_ok=True)
plt.savefig("analysis/delta_study.png")
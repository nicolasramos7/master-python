"""A synthetic loan book, shaped like Lending Club's public data. GIVEN -- not a stub.

Why synthetic: the real Lending Club file is ~1.5 GB behind a Kaggle login, and
it does not come with an answer key. This one does. Every row carries
`true_pd`, the actual probability of default that generated its outcome. That
gives you the project's verification hook for free:

    No honest model can beat a model that predicts `true_pd` exactly.
    If yours does, it is cheating -- and you have to find out how.

The raw frame is deliberately messy in the ways the real file is messy:

    int_rate      "13.56%"           a string, with a percent sign
    term          " 36 months"       a string, with a leading space
    emp_length    "10+ years", "< 1 year", "n/a"
    issue_d       "Jan-2019"         month-year string, not a date
    loan_status   "Fully Paid" / "Charged Off"   the target, as text

Two columns only exist AFTER the outcome is known -- `recoveries` (money
clawed back from a defaulted loan) and `last_pymnt_amnt`. In the real data they
sit innocently alongside the application fields. They are the leak.

The world also changes over time: loan volume grows, a downturn in 2023 raises
default rates across the board, and a new `home_ownership` value ("OTHER")
first appears late in 2023. A random split hides all three. A time split does
not.

Do not read the generating formula below until the project is done -- the
point is to discover the structure through the model, not to copy it.
"""

import numpy as np
import pandas as pd

from .arrays import Floats

GRADES = np.array(["A", "B", "C", "D", "E"])
MONTHS = pd.date_range("2019-01-01", "2023-12-01", freq="MS")
DOWNTURN_START = pd.Timestamp("2023-01-01")
OTHER_START = pd.Timestamp("2023-07-01")


def _sigmoid(x: Floats) -> Floats:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


def make_loans(n: int = 20_000, seed: int = 20260925) -> pd.DataFrame:
    """Generate `n` raw loans, in random (not date) order.

    Deterministic for a given seed. The `true_pd` column is the answer key:
    use it to check your work, never as a feature.
    """
    rng = np.random.default_rng(seed)

    # When: volume grows linearly over the five years.
    weights = np.linspace(1.0, 3.0, MONTHS.size)
    month_idx = rng.choice(MONTHS.size, size=n, p=weights / weights.sum())
    issue = MONTHS[month_idx]
    downturn = np.asarray(issue >= DOWNTURN_START, dtype=np.float64)

    # Who: application fields.
    grade_score = rng.choice(5, size=n, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    int_rate = np.clip(
        np.array([0.07, 0.11, 0.15, 0.19, 0.24])[grade_score] + rng.normal(0, 0.01, n), 0.05, 0.3
    )
    term = np.where(rng.random(n) < 0.15 + 0.1 * grade_score, 60, 36)
    annual_inc = np.round(rng.lognormal(np.log(65_000), 0.5, n), -2)
    loan_amnt = np.clip(np.round(rng.lognormal(np.log(12_000), 0.6, n) / 25) * 25, 1_000, 40_000)
    emp_years = rng.choice(11, size=n, p=[0.08] + [0.07] * 9 + [0.29])
    emp_missing = rng.random(n) < 0.07
    dti = np.clip(rng.normal(18, 9, n), 0, 60).round(2)
    dti_missing = rng.random(n) < 0.01
    home = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n, p=[0.4, 0.45, 0.15])
    home = np.where((issue >= OTHER_START) & (rng.random(n) < 0.03), "OTHER", home)

    # The answer key.
    logit = (
        -4.0
        + 0.45 * grade_score
        + 1.2 * np.log(loan_amnt / annual_inc / 0.2).clip(-2, 2)
        + 0.35 * (term == 60)
        + 1.0 * (dti > 35)
        + 0.5 * emp_missing
        + 0.25 * (home == "RENT")
        + 0.5 * (home == "OTHER")
        + 0.6 * downturn
    )
    true_pd = _sigmoid(logit)
    default = rng.random(n) < true_pd

    # After the fact: these are only known once the loan has resolved.
    recoveries = np.where(default, loan_amnt * rng.uniform(0.02, 0.25, n), 0.0).round(2)
    last_pymnt = np.where(
        default, loan_amnt * rng.uniform(0.0, 0.05, n), loan_amnt * rng.uniform(0.02, 0.6, n)
    ).round(2)

    # Make it look like the real file.
    emp_labels = np.array(
        ["< 1 year", "1 year"] + [f"{i} years" for i in range(2, 10)] + ["10+ years"]
    )
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "issue_d": issue.strftime("%b-%Y"),
            "loan_amnt": loan_amnt,
            "term": np.char.add(" ", np.char.add(term.astype(str), " months")),
            "int_rate": [f"{r * 100:.2f}%" for r in int_rate],
            "grade": GRADES[grade_score],
            "emp_length": np.where(emp_missing, "n/a", emp_labels[emp_years]),
            "home_ownership": home,
            "annual_inc": annual_inc,
            "dti": np.where(dti_missing, np.nan, dti),
            "loan_status": np.where(default, "Charged Off", "Fully Paid"),
            "recoveries": recoveries,
            "last_pymnt_amnt": last_pymnt,
            "true_pd": true_pd,
        }
    )

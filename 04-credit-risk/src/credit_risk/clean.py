"""Sitting 1, part 1: raw strings in, typed columns out. The pandas workout.

Everything in the raw file is text that should not be. Turn it into dtypes:

    raw column     raw value            clean column       clean value
    ----------     ---------            ------------       -----------
    issue_d        "Jan-2019"      ->   issue_date         Timestamp("2019-01-01")
    term           " 36 months"    ->   term_months        36            (int64)
    int_rate       "13.56%"        ->   int_rate           0.1356        (float64, a fraction)
    emp_length     "10+ years"     ->   emp_length_years   10.0          (float64)
                   "< 1 year"      ->                      0.0
                   "n/a"           ->                      NaN
    loan_status    "Charged Off"   ->   default            1             (int64)
                   "Fully Paid"    ->                      0

Drop the four raw columns once parsed. Every other column passes through as-is,
including `true_pd` (the answer key -- it is not a feature, but the tests need it).

The Python in it: the `.str` accessor (`.str.strip`, `.str.rstrip`,
`.str.removesuffix`), `pd.to_datetime(..., format=...)`, `Series.map` with a
dict, `.astype`, and `DataFrame.assign` for building new columns without
mutating the input. Hint: a dict comprehension builds the emp_length map in
one line.

Reject, don't guess. An unknown `loan_status` or `emp_length` raises
`ValueError` naming the bad values. `Series.map` silently turns anything
missing from the dict into NaN -- which is exactly the silent wrongness this
project is about. Check with `.isin` before you map.
"""

import pandas as pd

STATUS = {"Fully Paid": 0, "Charged Off": 1}
"""loan_status -> default. Any other status is an error, not a NaN."""


def parse_percent(s: pd.Series) -> pd.Series:
    """ "13.56%" -> 0.1356. Returns float64."""
    return s.str.rstrip("%").astype("float64") / 100


def parse_term(s: pd.Series) -> pd.Series:
    """ " 36 months" -> 36. Returns int64. Note the leading space."""
    return s.str.strip().str.replace(" months", "", regex=False).astype("int64")


def parse_emp_length(s: pd.Series) -> pd.Series:
    """ "10+ years" -> 10.0, "< 1 year" -> 0.0, "n/a" -> NaN. Returns float64.

    Raises:
        ValueError: on any value that is not one of the known labels or "n/a".
    """
    mapper = {
        "< 1 year": 0.0,
        "1 year": 1.0,
        "2 years": 2.0,
        "3 years": 3.0,
        "4 years": 4.0,
        "5 years": 5.0,
        "6 years": 6.0,
        "7 years": 7.0,
        "8 years": 8.0,
        "9 years": 9.0,
        "10+ years": 10.0,
        "n/a": float("nan"),
    }

    known = s.isin(mapper.keys())
    if not known.all():
        raise ValueError(f"unknown emp_length values: {sorted(s[~known].unique())}")

    return s.map(mapper).astype("float64")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse every messy column into a proper dtype. Never mutates `raw`.

    Raises:
        ValueError: on any `loan_status` outside `STATUS`.
    """
    new_column_names = {
        "issue_d": "issue_date",
        "term": "term_months",
        "emp_length": "emp_length_years",
        "loan_status": "default",
    }

    raw = raw.rename(columns=new_column_names)

    raw["issue_date"] = pd.to_datetime(raw["issue_date"], format="%b-%Y")

    raw["term_months"] = parse_term(raw["term_months"])

    raw["int_rate"] = parse_percent(raw["int_rate"])

    raw["emp_length_years"] = parse_emp_length(raw["emp_length_years"])

    loans = raw["default"]

    known = loans.isin(STATUS.keys())
    if not known.all():
        raise ValueError(f"unknown loan types: {sorted(loans[~known].unique())}")

    raw["default"] = raw["default"].map(STATUS)

    return raw

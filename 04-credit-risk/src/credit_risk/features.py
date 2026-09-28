"""Sitting 1, part 2: which columns are features, and your first sklearn class.

The column lists below are the most important design decision in the project,
so they are given. Read them as a contract:

    NUMERIC, CATEGORICAL   known on the day the loan is applied for
    LEAKY                  only known after the loan has resolved
    TARGET                 what we predict

`id`, `issue_date` and `true_pd` are in none of them, deliberately.

`QuantileClipper` is a transformer written against scikit-learn's API. The
rules sklearn imposes are the lesson -- they are how the library makes
leakage-free pipelines possible:

1. `__init__` only stores its arguments, unchanged, under the same names. No
   validation, no computation. (`sklearn.base.clone` rebuilds your object from
   `get_params()`; anything else you do in `__init__` gets lost.) Given below.
2. `fit(X, y=None)` learns state from the TRAINING data, stores it in
   attributes ending in an underscore (`bounds_`), and returns `self`.
3. `transform(X)` applies that stored state to ANY data -- training, test, or
   tomorrow's applications. It never re-learns. If it has not been fitted, it
   raises `sklearn.exceptions.NotFittedError`
   (`sklearn.utils.validation.check_is_fitted` does this for you).

Rule 3 is the entire leakage lesson in miniature. A clipper that recomputes
its quantiles inside `transform` quietly peeks at the test set.

The NumPy in it: `np.nanquantile(X, [lower, upper], axis=0)` gives a (2,
n_features) array; `np.clip(X, low, high)` broadcasts it back over rows.
NaNs must pass through untouched -- imputation happens after clipping.

Inherit from `OneToOneFeatureMixin, TransformerMixin, BaseEstimator`, in that
order. BaseEstimator gives you get_params/set_params from your __init__
signature; TransformerMixin gives you fit_transform; OneToOneFeatureMixin
gives you get_feature_names_out.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .arrays import Floats

NUMERIC = [
    "loan_amnt",
    "term_months",
    "int_rate",
    "emp_length_years",
    "annual_inc",
    "dti",
    "loan_to_income",
]
CATEGORICAL = ["grade", "home_ownership"]
LEAKY = ["recoveries", "last_pymnt_amnt"]
TARGET = "default"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add `loan_to_income` = loan_amnt / annual_inc. Never mutates `df`.

    Stateless -- it looks at one row at a time -- so it is safe to run before
    the split. Anything that looks at a whole column (a mean, a quantile, a
    scaler) is not, and belongs in the pipeline.
    """
    return df.assign(loan_to_income=df["loan_amnt"] / df["annual_inc"])


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Clip each column to quantiles learned in `fit`. A winsorizer."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99) -> None:
        self.lower = lower
        self.upper = upper

    def fit(self, X: npt.ArrayLike, y: object = None) -> "QuantileClipper":
        """Learn `bounds_`, shape (2, n_features): row 0 lows, row 1 highs.

        Validate here, not in `__init__`: raise ValueError unless
        0 <= lower < upper <= 1. Also set `n_features_in_`.
        """
        if not 0.0 <= self.lower < self.upper <= 1.0:
            raise ValueError(
                f"need 0 <= lower < upper <= 1, got lower={self.lower}, upper={self.upper}"
            )

        values = np.asarray(X, dtype=np.float64)

        self.bounds_ = np.nanquantile(values, [self.lower, self.upper], axis=0)

        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X: npt.ArrayLike) -> Floats:
        """Clip X to the fitted bounds. NaN in, NaN out."""
        check_is_fitted(self, "bounds_")

        values = np.asarray(X, dtype=np.float64)

        return np.clip(values, self.bounds_[0], self.bounds_[1])

"""Sitting 1. Your first class written against someone else's API."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from credit_risk.features import (
    CATEGORICAL,
    LEAKY,
    NUMERIC,
    TARGET,
    QuantileClipper,
    add_features,
)


def test_column_lists_are_disjoint() -> None:
    """A leaky column that is also listed as a feature leaks in both experiments."""
    groups = [set(NUMERIC), set(CATEGORICAL), set(LEAKY), {TARGET}]
    assert sum(len(g) for g in groups) == len(set().union(*groups))


def test_add_features(loans: pd.DataFrame) -> None:
    out = add_features(loans)
    expected = loans["loan_amnt"] / loans["annual_inc"]
    assert np.allclose(out["loan_to_income"], expected)
    assert "loan_to_income" not in loans.columns  # input untouched


def test_clipper_learns_bounds_from_fit_data_only() -> None:
    """Fit on one set, transform another: the bounds must not move."""
    train = np.arange(101, dtype=np.float64).reshape(-1, 1)  # 0..100
    clipper = QuantileClipper(lower=0.1, upper=0.9).fit(train)
    assert clipper.bounds_.shape == (2, 1)
    assert clipper.bounds_[:, 0].tolist() == pytest.approx([10.0, 90.0])
    new = np.array([[-1000.0], [50.0], [1000.0]])
    assert clipper.transform(new)[:, 0].tolist() == pytest.approx([10.0, 50.0, 90.0])


def test_clipper_works_per_column() -> None:
    """One (low, high) pair per column, broadcast over rows."""
    x = np.column_stack([np.arange(101.0), 1000.0 * np.arange(101.0)])
    out = QuantileClipper(lower=0.0, upper=0.5).fit_transform(x)
    assert out[:, 0].max() == pytest.approx(50.0)
    assert out[:, 1].max() == pytest.approx(50_000.0)


def test_clipper_passes_nan_through() -> None:
    """Imputation comes after clipping, so NaN must survive -- and not poison the quantiles."""
    x = np.array([[1.0], [np.nan], [2.0], [3.0]])
    clipper = QuantileClipper(lower=0.0, upper=1.0).fit(x)
    assert not np.isnan(clipper.bounds_).any()
    out = clipper.transform(x)
    assert np.isnan(out[1, 0])
    assert out[[0, 2, 3], 0].tolist() == [1.0, 2.0, 3.0]


def test_clipper_refuses_to_transform_before_fit() -> None:
    with pytest.raises(NotFittedError):
        QuantileClipper().transform(np.ones((3, 1)))


def test_clipper_validates_in_fit() -> None:
    with pytest.raises(ValueError):
        QuantileClipper(lower=0.9, upper=0.1).fit(np.ones((3, 1)))


def test_clipper_survives_clone() -> None:
    """`clone` rebuilds from get_params(). If __init__ did anything else, it is gone."""
    original = QuantileClipper(lower=0.05, upper=0.95)
    copy = clone(original)
    assert copy.get_params() == {"lower": 0.05, "upper": 0.95}
    assert copy is not original

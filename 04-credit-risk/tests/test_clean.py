"""Sitting 1. Nothing downstream works until these pass -- start here."""

import numpy as np
import pandas as pd
import pytest

from credit_risk.clean import clean, parse_emp_length, parse_percent, parse_term


def test_parse_percent() -> None:
    parsed = parse_percent(pd.Series(["13.56%", "7.00%"]))
    assert parsed.dtype == np.float64
    assert parsed.tolist() == pytest.approx([0.1356, 0.07])


def test_parse_term_handles_the_leading_space() -> None:
    parsed = parse_term(pd.Series([" 36 months", " 60 months"]))
    assert parsed.dtype == np.int64
    assert parsed.tolist() == [36, 60]


def test_parse_emp_length() -> None:
    parsed = parse_emp_length(pd.Series(["< 1 year", "1 year", "5 years", "10+ years", "n/a"]))
    assert parsed.dtype == np.float64
    assert parsed.iloc[:4].tolist() == [0.0, 1.0, 5.0, 10.0]
    assert np.isnan(parsed.iloc[4])


def test_parse_emp_length_rejects_unknown_labels() -> None:
    """`.map` would turn this into a NaN that looks exactly like "n/a". Reject it."""
    with pytest.raises(ValueError, match="11 years"):
        parse_emp_length(pd.Series(["1 year", "11 years"]))


def test_clean_produces_typed_columns(loans: pd.DataFrame) -> None:
    assert pd.api.types.is_datetime64_any_dtype(loans["issue_date"])
    assert loans["term_months"].dtype == np.int64
    assert loans["int_rate"].between(0.0, 1.0).all()
    assert set(loans["default"].unique()) == {0, 1}
    for gone in ("issue_d", "term", "emp_length", "loan_status"):
        assert gone not in loans.columns


def test_clean_keeps_every_row(raw: pd.DataFrame, loans: pd.DataFrame) -> None:
    assert len(loans) == len(raw)
    assert loans["id"].tolist() == raw["id"].tolist()


def test_default_matches_loan_status(raw: pd.DataFrame, loans: pd.DataFrame) -> None:
    charged_off = (raw["loan_status"] == "Charged Off").to_numpy()
    assert np.array_equal(loans["default"].to_numpy() == 1, charged_off)


def test_clean_does_not_mutate_its_input(raw: pd.DataFrame) -> None:
    before = raw.copy()
    clean(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_clean_rejects_unknown_loan_status(raw: pd.DataFrame) -> None:
    bad = raw.head(3).assign(loan_status=["Fully Paid", "Current", "Charged Off"])
    with pytest.raises(ValueError, match="Current"):
        clean(bad)

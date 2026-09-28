"""Shared fixtures. Session-scoped: the loan book is built once per test run.

Session scope means every test sees the SAME DataFrame object. A test that
mutates it corrupts every test after it, in an order-dependent way. None of
these tests mutate it, and your functions must not either -- `clean` and
`add_features` are tested for exactly that.
"""

import pandas as pd
import pytest

from credit_risk.clean import clean
from credit_risk.data import make_loans


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    """20,000 raw loans, messy strings and all."""
    return make_loans()


@pytest.fixture(scope="session")
def loans(raw: pd.DataFrame) -> pd.DataFrame:
    """The same loans, cleaned. Every test past test_clean.py starts here."""
    return clean(raw)

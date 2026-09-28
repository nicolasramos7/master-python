"""Sittings 1 and 2. The split, the pipeline, and the two experiments.

The key test in the whole project is `test_leaky_model_beats_the_oracle`.
Beating a model that knows the true probability is impossible -- unless you
have seen the answer. That is how you detect leakage without being told.
"""

import numpy as np
import pandas as pd
import pytest

from credit_risk.features import TARGET, add_features
from credit_risk.models import (
    Result,
    build_pipeline,
    default_model,
    honest_experiment,
    leaky_experiment,
    oracle,
    time_split,
)


@pytest.fixture(scope="module")
def honest(loans: pd.DataFrame) -> Result:
    return honest_experiment(loans)


@pytest.fixture(scope="module")
def leaky(loans: pd.DataFrame) -> Result:
    return leaky_experiment(loans)


@pytest.fixture(scope="module")
def best(loans: pd.DataFrame) -> Result:
    return oracle(loans)


def auc(result: Result) -> float:
    """Local helper so this file does not depend on evaluate.py being done."""
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(result.y_true, result.y_score))


# --- time_split -------------------------------------------------------------


def test_time_split_puts_the_past_before_the_future(loans: pd.DataFrame) -> None:
    train, test = time_split(loans)
    assert train["issue_date"].max() < test["issue_date"].min()


def test_time_split_loses_nothing(loans: pd.DataFrame) -> None:
    train, test = time_split(loans)
    assert len(train) + len(test) == len(loans)
    assert set(train["id"]).isdisjoint(test["id"])


def test_time_split_fraction_is_roughly_right(loans: pd.DataFrame) -> None:
    _, test = time_split(loans, test_frac=0.25)
    assert len(test) / len(loans) == pytest.approx(0.25, abs=0.03)


def test_the_future_is_worse_than_the_past(loans: pd.DataFrame) -> None:
    """Default rates rise in the test period. A random split averages this away."""
    train, test = time_split(loans)
    assert test[TARGET].mean() > train[TARGET].mean() * 1.2


# --- the pipeline -----------------------------------------------------------


def test_pipeline_has_named_steps() -> None:
    pipe = build_pipeline(default_model())
    assert list(pipe.named_steps) == ["prep", "model"]


def test_pipeline_survives_a_category_it_never_saw(loans: pd.DataFrame) -> None:
    """ "OTHER" first appears in late 2023 -- after the training window closes.

    Without handle_unknown="ignore", predict() raises on the first such row.
    In production that is an outage on the day a new category ships.
    """
    train, test = time_split(add_features(loans))
    assert "OTHER" not in set(train["home_ownership"])
    assert "OTHER" in set(test["home_ownership"])
    pipe = build_pipeline(default_model()).fit(train, train[TARGET])
    probs = pipe.predict_proba(test)[:, 1]
    assert probs.shape == (len(test),)


# --- the experiments --------------------------------------------------------


def test_results_are_probabilities(honest: Result, leaky: Result) -> None:
    for result in (honest, leaky):
        assert result.y_true.shape == result.y_score.shape
        assert ((result.y_score >= 0.0) & (result.y_score <= 1.0)).all()
        assert set(np.unique(result.y_true)) == {0.0, 1.0}


def test_honest_test_set_is_the_time_split(loans: pd.DataFrame, honest: Result) -> None:
    _, test = time_split(loans)
    assert honest.y_true.size == len(test)
    assert honest.train_base_rate == pytest.approx(time_split(loans)[0][TARGET].mean())


def test_honest_model_learns_something(honest: Result) -> None:
    assert auc(honest) > 0.70


def test_honest_model_cannot_beat_the_oracle(honest: Result, best: Result) -> None:
    """Same test rows, so the comparison is fair. A small margin for sampling luck."""
    assert np.array_equal(honest.y_true, best.y_true)
    assert auc(honest) < auc(best) + 0.01


def test_leaky_model_beats_the_oracle(leaky: Result, best: Result) -> None:
    """Impossible without cheating. This assertion passing is the proof of leakage."""
    assert auc(leaky) > auc(best) + 0.1

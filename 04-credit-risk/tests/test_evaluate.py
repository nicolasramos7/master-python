"""Sitting 2. Baselines, thresholds, and error bars."""

import numpy as np
import pandas as pd
import pytest
from numpy.random import default_rng

from credit_risk.evaluate import (
    baseline,
    best_threshold,
    bootstrap_auc_diff,
    expected_costs,
    score,
)
from credit_risk.models import Result, honest_experiment, oracle


@pytest.fixture(scope="module")
def honest(loans: pd.DataFrame) -> Result:
    return honest_experiment(loans)


@pytest.fixture(scope="module")
def best(loans: pd.DataFrame) -> Result:
    return oracle(loans)


# --- the baseline ----------------------------------------------------------


def test_baseline_is_a_constant(honest: Result) -> None:
    pred = baseline(honest.train_base_rate, honest.y_true)
    assert pred.shape == honest.y_true.shape
    assert np.all(pred == honest.train_base_rate)


def test_baseline_scores(honest: Result) -> None:
    """AUC exactly 0.5; log loss is the cross-entropy of two base rates, exactly."""
    p = honest.train_base_rate
    q = float(honest.y_true.mean())
    s = score(honest.y_true, baseline(p, honest.y_true))
    assert s.auc == 0.5
    assert s.log_loss == pytest.approx(-(q * np.log(p) + (1 - q) * np.log(1 - p)))
    assert s.base_rate == pytest.approx(q)


def test_accuracy_is_a_useless_metric_here(honest: Result) -> None:
    """The do-nothing baseline scores ~89% accuracy -- and so does the real model."""
    dumb = score(honest.y_true, baseline(honest.train_base_rate, honest.y_true))
    real = score(honest.y_true, honest.y_score)
    assert dumb.accuracy_at_half > 0.85
    assert real.accuracy_at_half == pytest.approx(dumb.accuracy_at_half, abs=0.01)


def test_model_beats_baseline_where_it_counts(honest: Result) -> None:
    dumb = score(honest.y_true, baseline(honest.train_base_rate, honest.y_true))
    real = score(honest.y_true, honest.y_score)
    assert real.log_loss < dumb.log_loss
    assert real.pr_auc > 2 * dumb.pr_auc


# --- thresholds ------------------------------------------------------------


def test_half_threshold_misses_nearly_every_default(honest: Result) -> None:
    """The planted trap: at 0.5 the model flags almost nobody."""
    flagged = honest.y_score >= 0.5
    recall = (flagged & (honest.y_true == 1)).sum() / (honest.y_true == 1).sum()
    assert recall < 0.05


def test_expected_costs_by_hand() -> None:
    y_true = np.array([1.0, 0.0, 0.0, 1.0])
    y_score = np.array([0.9, 0.6, 0.2, 0.1])
    thresholds = np.array([0.0, 0.5, 1.0])
    # 0.0: flag all -> 2 FP.  0.5: flag first two -> 1 FP, 1 FN.  1.0: flag none -> 2 FN.
    costs = expected_costs(y_true, y_score, thresholds, cost_fn=10.0, cost_fp=1.0)
    assert costs.shape == (3,)
    assert costs.tolist() == pytest.approx([2 / 4, 11 / 4, 20 / 4])


@pytest.mark.parametrize(("cost_fn", "cost_fp"), [(4.0, 1.0), (9.0, 1.0)])
def test_oracle_threshold_matches_decision_theory(
    best: Result, cost_fn: float, cost_fp: float
) -> None:
    """For calibrated probabilities the optimal threshold is cost_fp / (cost_fp + cost_fn)."""
    found = best_threshold(best.y_true, best.y_score, cost_fn, cost_fp)
    assert found == pytest.approx(cost_fp / (cost_fp + cost_fn), abs=0.06)


def test_expensive_misses_lower_the_threshold(honest: Result) -> None:
    cheap = best_threshold(honest.y_true, honest.y_score, cost_fn=2.0, cost_fp=1.0)
    dear = best_threshold(honest.y_true, honest.y_score, cost_fn=20.0, cost_fp=1.0)
    assert dear < cheap


# --- the bootstrap ---------------------------------------------------------


def test_bootstrap_of_a_model_against_itself_is_exactly_zero(honest: Result) -> None:
    """Paired resampling: identical scores, identical AUC on every resample."""
    point, low, high = bootstrap_auc_diff(
        honest.y_true, honest.y_score, honest.y_score, default_rng(0), n_boot=50
    )
    assert (point, low, high) == (0.0, 0.0, 0.0)


def test_bootstrap_separates_a_real_gap(honest: Result) -> None:
    noise = default_rng(1).random(honest.y_true.size)
    point, low, high = bootstrap_auc_diff(
        honest.y_true, honest.y_score, noise, default_rng(0), n_boot=200
    )
    assert low < point < high
    assert low > 0.0


def test_bootstrap_is_reproducible(honest: Result, best: Result) -> None:
    first = bootstrap_auc_diff(honest.y_true, best.y_score, honest.y_score, default_rng(3), 100)
    second = bootstrap_auc_diff(honest.y_true, best.y_score, honest.y_score, default_rng(3), 100)
    assert first == second

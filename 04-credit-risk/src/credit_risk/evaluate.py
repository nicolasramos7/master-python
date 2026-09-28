"""Sitting 2, part 2: turning predictions into numbers you can defend.

Three questions, one function group each:

1. **Is the model better than nothing?** `baseline` predicts the training base
   rate for every row. Its AUC is exactly 0.5 and its accuracy at a 0.5
   threshold is ~89% -- because ~89% of loans don't default. Any model must
   beat its log loss and PR-AUC. Nobody should ever quote accuracy here.

2. **Where do you draw the line?** A probability is not a decision. At the
   default 0.5 threshold, a model on an ~8%-default book flags almost nobody.
   Pick the threshold that minimises expected cost instead:

       cost = (cost_fn * missed_defaults + cost_fp * rejected_good_loans) / n

   Decision theory says that for a *calibrated* model the answer is
   `cost_fp / (cost_fp + cost_fn)`. The oracle is calibrated by construction,
   so it must land there -- that's a test.

   `expected_costs` evaluates every threshold at once by broadcasting:
   `y_score[None, :] >= thresholds[:, None]` is a (n_thresholds, n_rows)
   boolean matrix. Sum along axis 1. No loop over thresholds. (Project 3's
   broadcasting, applied.)

3. **Is model A really better than model B?** `bootstrap_auc_diff` resamples
   the test rows with replacement, recomputes both AUCs on the SAME resample
   each time, and returns a 95% percentile interval on the difference. Paired
   resampling matters: the two models' errors are correlated, and resampling
   them independently would give a needlessly wide interval.
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .arrays import Floats


@dataclass(frozen=True)
class Scores:
    """Every metric we report for one set of predictions."""

    auc: float
    """ROC AUC. Ranking quality; 0.5 is random, 1.0 is perfect."""
    pr_auc: float
    """Average precision. Unlike AUC, its no-skill level is the base rate, not 0.5."""
    log_loss: float
    """Punishes confident wrong probabilities. The one that cares about calibration."""
    accuracy_at_half: float
    """Accuracy at a 0.5 threshold. Reported only so you can see how useless it is."""
    base_rate: float
    """Fraction of positives in y_true."""


def score(y_true: Floats, y_score: Floats) -> Scores:
    """All the metrics in `Scores`. Use `sklearn.metrics` for the first three."""
    auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    log_l = log_loss(y_true, y_score)

    accuracy_at_half = ((y_score >= 0.5) == y_true).mean()

    base_rate = float(y_true.mean())

    return Scores(auc, pr_auc, log_l, float(accuracy_at_half), base_rate)


def baseline(train_base_rate: float, y_test: Floats) -> Floats:
    """Predict `train_base_rate` for every test row. Same shape as `y_test`."""
    return np.full(y_test.shape, train_base_rate, dtype=np.float64)


def expected_costs(
    y_true: Floats,
    y_score: Floats,
    thresholds: Floats,
    cost_fn: float,
    cost_fp: float,
) -> Floats:
    """Mean cost per loan at each threshold. Shape: (len(thresholds),).

    A row is flagged (rejected) when `y_score >= threshold`.
    """
    flagged = y_score[None, :] >= thresholds[:, None]

    pos = y_true.astype(bool)[None, :]
    fn = (~flagged & pos).sum(axis=1)
    fp = (flagged & ~pos).sum(axis=1)

    return np.asarray((cost_fn * fn + cost_fp * fp) / y_true.size, dtype=np.float64)


def best_threshold(
    y_true: Floats,
    y_score: Floats,
    cost_fn: float,
    cost_fp: float,
    n_grid: int = 201,
) -> float:
    """The threshold on `np.linspace(0, 1, n_grid)` with the lowest expected cost."""
    grid = np.linspace(0, 1, n_grid)
    costs = expected_costs(y_true, y_score, grid, cost_fn, cost_fp)
    return float(grid[np.argmin(costs)])


def bootstrap_auc_diff(
    y_true: Floats,
    score_a: Floats,
    score_b: Floats,
    rng: Generator,
    n_boot: int = 1_000,
) -> tuple[float, float, float]:
    """AUC(a) - AUC(b), with a paired-bootstrap 95% interval.

    Returns:
        (point_estimate, low, high). The point estimate is on the full test
        set, not the mean of the resamples.
    """
    point = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    n = y_true.size
    idx = rng.integers(0, n, size=(n_boot, n))

    diffs = np.array(
        [roc_auc_score(y_true[i], score_a[i]) - roc_auc_score(y_true[i], score_b[i]) for i in idx]
    )
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)

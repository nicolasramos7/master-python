"""Sitting 1, part 3 and sitting 2, part 1: the pipeline, and the experiment twice.

Split by time, then build the pipeline:

    Pipeline
    ├── "prep": ColumnTransformer
    │     ├── "num": QuantileClipper -> SimpleImputer(median) -> StandardScaler   on NUMERIC
    │     └── "cat": OneHotEncoder(handle_unknown="ignore")                     on CATEGORICAL
    └── "model": any classifier with predict_proba

Every stateful step (clip bounds, medians, means, categories) is learned in
`fit` and only in `fit`. That is what the Pipeline buys you: call `fit` on the
training rows and nothing downstream can see the test rows.

Then run the same model two ways. Both use `LogisticRegression` so the only
thing that differs is the discipline:

    leaky_experiment                         honest_experiment
    ----------------                         -----------------
    features: NUMERIC + LEAKY + CATEGORICAL  NUMERIC + CATEGORICAL
    preprocess ALL rows, THEN split          split, THEN fit the pipeline on train
    random split (train_test_split)          time_split: past trains, future tests

Both return a `Result` -- predictions only, no metrics. Scoring lives in
`evaluate.py`, printing in `__main__.py`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .arrays import Floats
from .features import CATEGORICAL, LEAKY, NUMERIC, TARGET, QuantileClipper, add_features


@dataclass(frozen=True)
class Result:
    """One experiment's out-of-sample predictions."""

    name: str
    y_true: Floats
    """Test-set labels, 0/1, as float64 so every array in here has one dtype."""
    y_score: Floats
    """Predicted probability of default for each test row."""
    train_base_rate: float
    """Default rate in the TRAINING rows. The baseline predicts this for everyone."""


def default_model() -> LogisticRegression:
    """The model both experiments use. Given."""
    return LogisticRegression(max_iter=1_000)


def time_split(df: pd.DataFrame, test_frac: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Oldest ~(1 - test_frac) of rows train, newest ~test_frac test.

    Split on a month boundary: every loan from a given `issue_date` lands on
    the same side, so `train.issue_date.max() < test.issue_date.min()` holds
    strictly. The fraction is therefore approximate. Order of rows within each
    side does not matter.
    """
    dates = df["issue_date"].sort_values()
    cutoff = dates.iloc[int(len(df) * (1 - test_frac))]
    mask = df["issue_date"] < cutoff
    training_set = df[mask]
    testing_set = df[~mask]
    return (training_set, testing_set)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """The "prep" step from the diagram above, on the given columns."""
    number_lane = make_pipeline(
        QuantileClipper(),
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    return ColumnTransformer(
        [
            ("num", number_lane, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )


def build_pipeline(
    estimator: object,
    numeric: list[str] = NUMERIC,
    categorical: list[str] = CATEGORICAL,
) -> Pipeline:
    """Preprocessor + estimator, as the named steps "prep" and "model"."""
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )


def leaky_experiment(df: pd.DataFrame, seed: int = 0) -> Result:
    """Everything wrong at once. Expect a spectacular AUC.

    `df` is the cleaned frame. Add features, preprocess every row, random-split
    75/25 with `random_state=seed`, fit `default_model()`, predict the test rows.
    """
    data = add_features(df)
    leaky_prepo = build_preprocessor(NUMERIC + LEAKY, CATEGORICAL)
    X = leaky_prepo.fit_transform(data)
    y = data[TARGET].to_numpy(dtype=np.float64)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    model = default_model()
    probs = model.fit(X_train, y_train).predict_proba(X_test)

    return Result(
        name="leaky",
        y_true=y_test,
        y_score=probs[:, 1],
        train_base_rate=float(y_train.mean()),
    )


def honest_experiment(
    df: pd.DataFrame,
    estimator: object | None = None,
    name: str = "honest",
) -> Result:
    """Everything right. `estimator` defaults to `default_model()`.

    Taking the estimator as a parameter means the stretch goal (gradient
    boosting vs logistic) is one extra call, not a copy-pasted function.
    """
    data = add_features(df)
    train, test = time_split(data)
    if estimator is None:
        estimator = default_model()
    probs = build_pipeline(estimator).fit(train, train[TARGET]).predict_proba(test)

    y_score = probs[:, 1]

    return Result(
        name=name,
        y_true=test[TARGET].to_numpy(dtype=np.float64),
        y_score=y_score,
        train_base_rate=float(train[TARGET].mean()),
    )


def oracle(df: pd.DataFrame) -> Result:
    """The answer key as a model: `true_pd` on the same test rows as `honest_experiment`.

    The ceiling. No model trained on application data can beat this in
    expectation. Anything that does has seen the answer.
    """
    train, test = time_split(df)

    return Result(
        name="oracle",
        y_true=test[TARGET].to_numpy(dtype=np.float64),
        y_score=test["true_pd"].to_numpy(dtype=np.float64),
        train_base_rate=float(train[TARGET].mean()),
    )

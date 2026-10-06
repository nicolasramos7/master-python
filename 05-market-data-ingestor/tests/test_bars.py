"""Sitting 1, step 4. JSON to a typed frame -- and the two silent traps in the data."""

from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import pytest

from ingest.bars import SCHEMA, Bar, Payload, empty_frame, suspicious_moves, to_frame
from ingest.fake_api import bars_for, exchange_for


def payload(symbol: str, start: date, end: date) -> Payload:
    """Exactly what the vendor would send for this window."""
    bars = cast(list[Bar], bars_for(symbol, start, end))
    return {"symbol": symbol, "timezone": exchange_for(symbol).timezone, "bars": bars}


def flat(symbol: str, price: float, days: int = 3) -> Payload:
    """A hand-made payload: `days` bars, all at the same price."""
    bars: list[Bar] = [
        {
            "t": f"2021-03-0{d + 1}T16:00:00",
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "adj_close": price,
            "volume": 1_000,
        }
        for d in range(days)
    ]
    return {"symbol": symbol, "timezone": "America/New_York", "bars": bars}


def ts(frame: pd.DataFrame, day: str) -> pd.Timestamp:
    return pd.Timestamp(frame.loc[frame["date"] == day, "ts_utc"].iloc[0])


# --- shape --------------------------------------------------------------------


def test_columns_and_dtypes_are_the_schema() -> None:
    frame = to_frame(payload("AAPL", date(2019, 1, 1), date(2019, 12, 31)))
    assert list(frame.columns) == list(SCHEMA)
    assert {c: str(t) for c, t in frame.dtypes.items()} == SCHEMA


def test_one_row_per_weekday_sorted_by_date() -> None:
    frame = to_frame(payload("MSFT", date(2019, 1, 1), date(2019, 12, 31)))
    assert len(frame) == len(pd.bdate_range("2019-01-01", "2019-12-31"))
    assert frame["date"].is_monotonic_increasing
    assert (frame["ticker"] == "MSFT").all()
    assert frame.index.tolist() == list(range(len(frame)))


def test_sorts_bars_that_arrive_out_of_order() -> None:
    data = payload("SPY", date(2019, 1, 1), date(2019, 1, 31))
    data["bars"].reverse()
    assert to_frame(data)["date"].is_monotonic_increasing


def test_an_empty_window_still_has_the_schema() -> None:
    """Written to disk as proof the job ran. It must read back like any other file."""
    frame = to_frame({"symbol": "AAPL", "timezone": "America/New_York", "bars": []})
    assert frame.empty
    assert frame.dtypes.equals(empty_frame().dtypes)
    assert list(frame.columns) == list(SCHEMA)


# --- trap 1: timezones ----------------------------------------------------------


def test_new_york_close_in_utc_on_both_sides_of_daylight_saving() -> None:
    """16:00 New York is 21:00 UTC in winter and 20:00 UTC in summer.

    "Add five hours" gets exactly one of these right.
    """
    frame = to_frame(payload("AAPL", date(2019, 1, 1), date(2019, 12, 31)))
    assert ts(frame, "2019-01-02") == pd.Timestamp("2019-01-02 21:00", tz="UTC")
    assert ts(frame, "2019-07-01") == pd.Timestamp("2019-07-01 20:00", tz="UTC")


def test_frankfurt_closes_hours_before_new_york() -> None:
    """Naive, 17:30 (Frankfurt) looks later than 16:00 (New York). In UTC it's 4.5h earlier."""
    ny = to_frame(payload("AAPL", date(2019, 1, 1), date(2019, 1, 31)))
    de = to_frame(payload("SAP.DE", date(2019, 1, 1), date(2019, 1, 31)))
    assert ts(de, "2019-01-02") == pd.Timestamp("2019-01-02 16:30", tz="UTC")
    assert ts(ny, "2019-01-02") - ts(de, "2019-01-02") == pd.Timedelta(hours=4, minutes=30)


def test_date_is_the_local_session_date_at_midnight() -> None:
    frame = to_frame(payload("SAP.DE", date(2019, 1, 1), date(2019, 1, 31)))
    assert frame["date"].iloc[0] == pd.Timestamp("2019-01-01")
    assert (frame["date"] == frame["date"].dt.normalize()).all()


# --- trap 2: splits ---------------------------------------------------------------


def test_the_split_is_a_crash_in_close_and_nothing_in_adj_close() -> None:
    """7-for-1: the unadjusted price drops by ~6/7 overnight. The company didn't."""
    frame = to_frame(payload("AAPL", date(2020, 1, 1), date(2020, 12, 31)))
    crashes = suspicious_moves(frame, "close")
    assert crashes["date"].tolist() == [pd.Timestamp("2020-08-31")]
    assert crashes["change"].iloc[0] == pytest.approx(-6 / 7, abs=0.05)
    assert suspicious_moves(frame, "adj_close").empty


def test_suspicious_moves_never_compares_two_tickers() -> None:
    """Stacked, the last $10 bar of CHEAP sits right above the first $100 bar of DEAR.

    Without groupby that boundary is a +900% "move". Neither stock moved at all.
    """
    both = pd.concat([to_frame(flat("CHEAP", 10.0)), to_frame(flat("DEAR", 100.0))])
    assert suspicious_moves(both, "close").empty


def test_suspicious_moves_honours_the_threshold() -> None:
    frame = to_frame(payload("AAPL", date(2020, 1, 1), date(2020, 12, 31)))
    everything = suspicious_moves(frame, "adj_close", threshold=0.0)
    changes = frame["adj_close"].pct_change().dropna()
    assert len(everything) == int(np.count_nonzero(changes))

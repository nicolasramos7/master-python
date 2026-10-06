"""Sitting 1, part 3: the vendor's JSON in, a typed DataFrame out. Two silent traps.

The wire format is described by `TypedDict`s below -- given. `response.json()`
returns `Any`; `typing.cast` it to `Payload` once, at the boundary, and mypy
checks every access after that.

`to_frame` produces exactly these columns, in this order, with these dtypes
(`SCHEMA`, given):

    ticker      str                   "AAPL"
    date        datetime64[ns]        2015-01-02         the exchange-LOCAL session date
    ts_utc      datetime64[ns, UTC]   2015-01-02 21:00Z  the bar's close, in UTC
    open ... adj_close   float64
    volume      int64

**Trap 1: naive timestamps.** `"t": "2015-01-02T16:00:00"` has no offset. It
is New York time -- the payload's `timezone` field says so -- but pandas parses
it as a naive timestamp that means nothing. Put AAPL and SAP.DE side by side
naive and 16:00 New York looks earlier than 17:30 Frankfurt; it is actually
five hours later. And "just add 5 hours" is wrong for half the year: New York
is UTC-5 in winter and UTC-4 in summer. Let the timezone database do it:
`.dt.tz_localize(tz)` says what the clock meant, `.dt.tz_convert("UTC")` moves
it. The tests pick one date either side of daylight-saving time.

**Trap 2: an empty window.** A year with no bars (before FIRST_DATE, say) must
still produce a frame with every column and dtype in `SCHEMA` -- it gets
written to disk as the proof the job is done, and the next run reads it back.
`pd.DataFrame([])` has no columns at all. Pass `columns=` explicitly, and end
with `.astype(SCHEMA)` so an empty frame and a full one are the same shape.

`suspicious_moves` is the check that catches the adjusted-vs-unadjusted trap:
a 7-for-1 split turns into an ~86% one-day crash in `close`, and nothing in
`adj_close`. Compute it per ticker -- with `groupby`, or the last day of one
ticker gets compared with the first day of the next.

The Python in it: `TypedDict`, `typing.cast`, the `.dt` accessor,
`tz_localize` / `tz_convert`, `astype` with a dict, `groupby(...).pct_change()`.
"""

from typing import TypedDict

import pandas as pd


class Bar(TypedDict):
    """One element of `payload["bars"]`."""

    t: str
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int


class Payload(TypedDict):
    """A 200 response body from `GET /v1/bars/{symbol}`."""

    symbol: str
    timezone: str
    bars: list[Bar]


BAR_FIELDS = ["t", "open", "high", "low", "close", "adj_close", "volume"]
"""The keys of `Bar`, in order. Pass as `columns=` so an empty list still has them."""

SCHEMA = {
    "ticker": "str",
    "date": "datetime64[ns]",
    "ts_utc": "datetime64[ns, UTC]",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "adj_close": "float64",
    "volume": "int64",
}
"""Every column `to_frame` returns, in order, with its dtype. The contract."""


def empty_frame() -> pd.DataFrame:
    """A frame with no rows and exactly `SCHEMA`. Given; `storage.read_all` uses it."""
    return pd.DataFrame(columns=list(SCHEMA)).astype(SCHEMA)


def to_frame(payload: Payload) -> pd.DataFrame:
    """Turn one response body into a typed frame, one row per bar, sorted by date."""
    df = pd.DataFrame(payload["bars"], columns=BAR_FIELDS)
    local = pd.to_datetime(df["t"], format="%Y-%m-%dT%H:%M:%S")
    df["date"] = local.dt.normalize()
    df["ts_utc"] = local.dt.tz_localize(payload["timezone"]).dt.tz_convert("UTC")
    df["ticker"] = payload["symbol"]     
    df = df.drop(columns="t")
    df = df[list(SCHEMA)].astype(SCHEMA)     
    return df.sort_values("date").reset_index(drop=True)


def suspicious_moves(
    frame: pd.DataFrame, column: str = "close", threshold: float = 0.5
) -> pd.DataFrame:
    """Rows whose one-day change in `column` exceeds `threshold` in absolute value.

    Per ticker, in date order. Returns those rows with an extra `change` column
    (the fractional move, e.g. -0.857), or an empty frame if there are none.
    """
    ordered = frame.sort_values(["ticker", "date"])
    change = ordered.groupby("ticker")[column].pct_change()
    over = change.abs() > threshold
    return ordered.assign(change=change).loc[over]
"""A fake market-data vendor that runs in-process. GIVEN -- not a stub.

Why fake: every free market-data API either needs a sign-up, rate-limits you to
a handful of calls a day, or changes its response format without notice. None
of that is what this project is about. This one lives inside `httpx` as a
`MockTransport`, so your code makes *real* `httpx` requests -- real status
codes, real headers, real JSON -- and nothing leaves your machine.

It behaves like a real vendor in all the ways that matter:

    latency          every request waits `latency` seconds (an `asyncio.sleep`,
                     so waiting requests genuinely overlap -- that is I/O-bound)
    429              more than `max_concurrent` requests in flight at once
    503              a random `fail_rate` of requests fail, as real APIs do
    500              after `outage_after` successful responses, every request
                     fails: the vendor went down mid-run. Use it to test resume.
    401              wrong or missing `Authorization: Bearer <key>` header
    404              an unknown or delisted symbol
    400              a malformed or inverted date range

The endpoint:

    GET {BASE_URL}/v1/bars/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD

    200 -> {"symbol": "AAPL",
            "timezone": "America/New_York",
            "bars": [{"t": "2015-01-02T16:00:00",      <- LOCAL exchange time,
                      "open": 101.3, "high": 102.9,       no offset. The trap.
                      "low": 100.8, "close": 102.1,
                      "adj_close": 14.5857, "volume": 3120000}, ...]}

The data is synthetic, deterministic, and has an answer key built in:

- One bar per weekday (no holidays -- the vendor is fake, not wrong), between
  FIRST_DATE and LAST_DATE. A window outside that range returns zero bars.
- Prices depend only on the symbol, never on which window you asked for or in
  what order. So fetching 2015 then 2016 gives the same rows as one request
  for both years -- which is what makes byte-identical reruns possible.
- `SPLITS` holds one stock split. Before the split date `close` is the
  unadjusted price, several times larger than `adj_close`. After, they agree.
  (The date is AAPL's real 2020 split; the 7-for-1 ratio is the roadmap's, not
  history's -- the real one was 4-for-1.)
- Symbols ending in `.DE` trade in Frankfurt, with timestamps in Berlin time.
  Everything else trades in New York.

The server counts what it sees, so tests can ask it questions:
`requests` (every request received, as "SYMBOL START END"), `served` (200s
returned), and `peak_in_flight` (the most concurrent requests it saw).
"""

import asyncio
import functools
import json
import random
import re
import zlib
from dataclasses import dataclass
from datetime import date, time, timedelta

import httpx
import numpy as np
import numpy.typing as npt

BASE_URL = "https://api.fakemarket.test"
API_KEY = "demo"
"""The key the fake vendor accepts. Put it in MARKETDATA_API_KEY, never in code."""

FIRST_DATE = date(2000, 1, 3)
LAST_DATE = date(2024, 12, 31)

SPLITS: dict[str, tuple[date, int]] = {"AAPL": (date(2020, 8, 31), 7)}
"""symbol -> (first trading day at the new share count, ratio)."""

DELISTED = frozenset({"ENRN", "LEHM"})
"""Well-formed symbols the vendor has never heard of. Always 404."""

_SYMBOL = re.compile(r"[A-Z]{1,5}(\.DE)?")
_PATH = re.compile(r"/v1/bars/(?P<symbol>[^/]+)")


@dataclass(frozen=True)
class Exchange:
    timezone: str
    close: time


NEW_YORK = Exchange("America/New_York", time(16, 0))
FRANKFURT = Exchange("Europe/Berlin", time(17, 30))


def exchange_for(symbol: str) -> Exchange:
    """Where a symbol trades. `.DE` is Frankfurt; everything else is New York."""
    return FRANKFURT if symbol.endswith(".DE") else NEW_YORK


@dataclass(frozen=True)
class _History:
    """Every bar the vendor has for one symbol, as parallel NumPy arrays."""

    days: npt.NDArray[np.datetime64]
    open: npt.NDArray[np.float64]
    high: npt.NDArray[np.float64]
    low: npt.NDArray[np.float64]
    close: npt.NDArray[np.float64]
    adj_close: npt.NDArray[np.float64]
    volume: npt.NDArray[np.int64]


@functools.lru_cache(maxsize=512)
def _history(symbol: str) -> _History:
    """Generate a symbol's whole history once, vectorized, and cache it.

    Seeded with crc32, NOT `hash()`: Python salts `hash(str)` per process
    (PYTHONHASHSEED), so a `hash()` seed gives different prices on every run.
    Vectorized so that generating it costs ~1ms: this runs inside the event
    loop, and a slow fake would pollute every timing you take.
    """
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    all_days = np.arange(np.datetime64(FIRST_DATE, "D"), np.datetime64(LAST_DATE, "D") + 1)
    days = all_days[np.is_busday(all_days)]
    n = days.size

    start = rng.uniform(20.0, 200.0)
    price = start * np.cumprod(1.0 + rng.normal(0.0003, 0.018, n))
    prev = np.concatenate(([start], price[:-1]))
    open_ = prev * (1.0 + rng.normal(0.0, 0.004, n))
    high = np.maximum(open_, price) * (1.0 + np.abs(rng.normal(0.0, 0.006, n)))
    low = np.minimum(open_, price) * (1.0 - np.abs(rng.normal(0.0, 0.006, n)))
    volume = rng.integers(1_000_000, 50_000_000, n)

    split_day, ratio = SPLITS.get(symbol, (LAST_DATE + timedelta(days=1), 1))
    k = np.where(days < np.datetime64(split_day), ratio, 1)
    return _History(
        days=days,
        open=np.round(open_ * k, 2),
        high=np.round(high * k, 2),
        low=np.round(low * k, 2),
        close=np.round(price * k, 2),
        adj_close=np.round(price, 4),
        volume=volume // k,
    )


def bars_for(symbol: str, start: date, end: date) -> list[dict[str, object]]:
    """The JSON bars for one request. Public so tests can build payloads by hand."""
    h = _history(symbol)
    lo = np.searchsorted(h.days, np.datetime64(start), side="left")
    hi = np.searchsorted(h.days, np.datetime64(end), side="right")
    close_time = exchange_for(symbol).close.isoformat()
    columns = zip(
        h.days[lo:hi].tolist(),
        h.open[lo:hi].tolist(),
        h.high[lo:hi].tolist(),
        h.low[lo:hi].tolist(),
        h.close[lo:hi].tolist(),
        h.adj_close[lo:hi].tolist(),
        h.volume[lo:hi].tolist(),
        strict=True,
    )
    return [
        {
            "t": f"{day.isoformat()}T{close_time}",
            "open": o,
            "high": hh,
            "low": ll,
            "close": c,
            "adj_close": a,
            "volume": v,
        }
        for day, o, hh, ll, c, a, v in columns
    ]


def _error(status: int, message: str, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json={"error": message}, headers=headers)


class FakeMarketAPI:
    """One fake vendor. Build a fresh one per test so its counters start at zero."""

    def __init__(
        self,
        *,
        api_key: str = API_KEY,
        latency: float = 0.0,
        fail_rate: float = 0.0,
        max_concurrent: int | None = None,
        outage_after: int | None = None,
        seed: int = 0,
    ) -> None:
        self.api_key = api_key
        self.latency = latency
        self.fail_rate = fail_rate
        self.max_concurrent = max_concurrent
        self.outage_after = outage_after
        self.requests: list[str] = []
        self.served = 0
        self.in_flight = 0
        self.peak_in_flight = 0
        self._rng = random.Random(seed)

    def transport(self) -> httpx.MockTransport:
        """Hand this to `httpx.AsyncClient(transport=...)`. See `client.make_client`."""
        return httpx.MockTransport(self._handle)

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        match = _PATH.fullmatch(request.url.path)
        symbol = match["symbol"] if match else "?"
        start_arg = request.url.params.get("start", "")
        end_arg = request.url.params.get("end", "")
        self.requests.append(f"{symbol} {start_arg} {end_arg}")

        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            return await self._respond(request, symbol, start_arg, end_arg)
        finally:
            self.in_flight -= 1

    async def _respond(
        self, request: httpx.Request, symbol: str, start_arg: str, end_arg: str
    ) -> httpx.Response:
        if request.headers.get("Authorization") != f"Bearer {self.api_key}":
            return _error(401, "missing or invalid API key")
        if self.max_concurrent is not None and self.in_flight > self.max_concurrent:
            return _error(429, "too many concurrent requests", {"Retry-After": "1"})
        if not _SYMBOL.fullmatch(symbol) or symbol in DELISTED:
            return _error(404, f"unknown symbol {symbol!r}")
        try:
            start, end = date.fromisoformat(start_arg), date.fromisoformat(end_arg)
        except ValueError:
            return _error(400, "start and end must be YYYY-MM-DD")
        if start > end:
            return _error(400, "start is after end")

        await asyncio.sleep(self.latency)

        if self.outage_after is not None and self.served >= self.outage_after:
            return _error(500, "internal server error")
        if self._rng.random() < self.fail_rate:
            return _error(503, "service unavailable, try again")

        self.served += 1
        body = {
            "symbol": symbol,
            "timezone": exchange_for(symbol).timezone,
            "bars": bars_for(symbol, start, end),
        }
        return httpx.Response(
            200, content=json.dumps(body), headers={"Content-Type": "application/json"}
        )

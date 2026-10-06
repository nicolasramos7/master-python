"""Sitting 2, step 1. One request: status codes in, the right exception out.

Retries here use real (tiny) sleeps -- `fetch_bars` is decorated with 10ms-base
backoff -- so the slowest test in this file takes a fraction of a second.
"""

import asyncio
import logging
from datetime import date

import httpx
import pytest

from conftest import fetch
from ingest.client import API_KEY_VAR, api_key_from_env, fetch_bars
from ingest.errors import ConfigError, PermanentError, TransientError
from ingest.fake_api import BASE_URL, FakeMarketAPI
from ingest.jobs import Job

AAPL_2019 = Job("AAPL", date(2019, 1, 1), date(2019, 12, 31))


# --- the secret -----------------------------------------------------------------


def test_reads_the_key_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_VAR, "sk-live-123")
    assert api_key_from_env() == "sk-live-123"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_a_missing_key_is_a_config_error_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch, value: str | None
) -> None:
    if value is None:
        monkeypatch.delenv(API_KEY_VAR, raising=False)
    else:
        monkeypatch.setenv(API_KEY_VAR, value)
    with pytest.raises(ConfigError, match=API_KEY_VAR):
        api_key_from_env()


def test_the_key_never_reaches_the_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Every log line, every level, through retries and failures. Not once."""
    secret = "sk-test-do-not-log-4c1d"
    api = FakeMarketAPI(api_key=secret, fail_rate=0.5, seed=3)
    with caplog.at_level(logging.DEBUG):
        fetch(api, AAPL_2019, api_key=secret)
        with pytest.raises(PermanentError):
            fetch(api, AAPL_2019, api_key="sk-wrong-key")
    assert caplog.records, "expected at least one retry warning"
    assert secret not in caplog.text
    assert "sk-wrong-key" not in caplog.text


# --- success ------------------------------------------------------------------------


def test_returns_the_payload(api: FakeMarketAPI) -> None:
    payload = fetch(api, AAPL_2019)
    assert payload["symbol"] == "AAPL"
    assert payload["timezone"] == "America/New_York"
    assert len(payload["bars"]) == 261
    assert payload["bars"][0]["t"] == "2019-01-01T16:00:00"


def test_asks_for_exactly_the_jobs_window(api: FakeMarketAPI) -> None:
    fetch(api, Job("SAP.DE", date(2019, 3, 1), date(2019, 3, 31)))
    assert api.requests == ["SAP.DE 2019-03-01 2019-03-31"]


# --- failures ------------------------------------------------------------------------


def test_an_unknown_symbol_is_permanent_and_asked_once(api: FakeMarketAPI) -> None:
    with pytest.raises(PermanentError, match="404"):
        fetch(api, Job("ENRN", date(2019, 1, 1), date(2019, 12, 31)))
    assert len(api.requests) == 1


def test_a_wrong_key_is_permanent(api: FakeMarketAPI) -> None:
    with pytest.raises(PermanentError, match="401"):
        fetch(api, AAPL_2019, api_key="not-the-key")
    assert len(api.requests) == 1


def test_a_flaky_vendor_is_retried_until_it_answers() -> None:
    api = FakeMarketAPI(fail_rate=0.5, seed=3)
    payload = fetch(api, AAPL_2019)
    assert len(payload["bars"]) == 261
    assert len(api.requests) > 1
    assert api.served == 1


def test_an_outage_uses_every_attempt_then_raises() -> None:
    """500 on every request: five attempts (the decorator's setting), then give up."""
    api = FakeMarketAPI(outage_after=0)
    with pytest.raises(TransientError, match="500"):
        fetch(api, AAPL_2019)
    assert len(api.requests) == 5


def test_a_dropped_connection_is_transient_and_keeps_its_cause() -> None:
    """`raise ... from exc` -- the httpx error stays attached as __cause__."""
    calls = 0

    def unplugged(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("connection refused", request=request)

    async def go() -> None:
        transport = httpx.MockTransport(unplugged)
        async with httpx.AsyncClient(transport=transport, base_url=BASE_URL) as client:
            await fetch_bars(client, AAPL_2019)

    with pytest.raises(TransientError) as info:
        asyncio.run(go())
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert calls == 5

"""Sitting 2, step 2. The verification hooks: idempotent, resumable, concurrent.

`api.requests` is the vendor's own log of what it was asked for. The strongest
statements in this file are about that log: a rerun asks for nothing, and a
resumed run asks for exactly what is missing.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from conftest import ingest
from ingest.fake_api import FakeMarketAPI
from ingest.jobs import make_jobs
from ingest.storage import read_all, snapshot

JOBS = list(make_jobs(["AAPL", "SAP.DE", "SPY"], date(2019, 1, 1), date(2020, 12, 31)))
"""3 tickers x 2 years = 6 jobs."""


def test_fetches_and_writes_every_job(api: FakeMarketAPI, tmp_path: Path) -> None:
    report = ingest(api, JOBS, tmp_path)
    assert (report.fetched, report.skipped, report.failed) == (6, 0, ())
    assert all(job.path(tmp_path).is_file() for job in JOBS)
    assert report.seconds > 0.0


def test_every_weekday_exactly_once(api: FakeMarketAPI, tmp_path: Path) -> None:
    """No gaps, no duplicate (ticker, date) pairs, across the year boundary too."""
    ingest(api, JOBS, tmp_path)
    data = read_all(tmp_path)
    weekdays = len(pd.bdate_range("2019-01-01", "2020-12-31"))
    assert data.groupby("ticker").size().to_dict() == {
        "AAPL": weekdays,
        "SAP.DE": weekdays,
        "SPY": weekdays,
    }
    assert not data.duplicated(["ticker", "date"]).any()


# --- verification hook 1: idempotency ------------------------------------------------


def test_a_second_run_asks_for_nothing_and_changes_nothing(tmp_path: Path) -> None:
    ingest(FakeMarketAPI(), JOBS, tmp_path)
    first = snapshot(tmp_path)

    again = FakeMarketAPI()
    report = ingest(again, JOBS, tmp_path)

    assert (report.fetched, report.skipped) == (0, 6)
    assert again.requests == []
    assert snapshot(tmp_path) == first


def test_two_runs_from_scratch_are_byte_identical(tmp_path: Path) -> None:
    """Same input, same bytes. This is what makes 'diff the output' a meaningful check."""
    ingest(FakeMarketAPI(), JOBS, tmp_path / "a", concurrency=6)
    ingest(FakeMarketAPI(), JOBS, tmp_path / "b", concurrency=1)
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")


# --- verification hook 2: resumability ------------------------------------------------


def test_resume_after_an_outage_fetches_only_what_is_missing(tmp_path: Path) -> None:
    """The vendor dies after 2 good responses. Rerun: exactly 4 requests, one per gap.

    And the end state is byte-identical to a run that never failed.
    """
    down = FakeMarketAPI(outage_after=2)
    first = ingest(down, JOBS, tmp_path / "data", concurrency=1)
    assert (first.fetched, len(first.failed)) == (2, 4)

    healthy = FakeMarketAPI()
    second = ingest(healthy, JOBS, tmp_path / "data")
    assert (second.fetched, second.skipped, second.failed) == (4, 2, ())
    assert len(healthy.requests) == 4

    ingest(FakeMarketAPI(), JOBS, tmp_path / "clean")
    assert snapshot(tmp_path / "data") == snapshot(tmp_path / "clean")


def test_one_bad_ticker_does_not_sink_the_run(api: FakeMarketAPI, tmp_path: Path) -> None:
    jobs = list(make_jobs(["AAPL", "ENRN"], date(2019, 1, 1), date(2020, 12, 31)))
    report = ingest(api, jobs, tmp_path)
    assert report.fetched == 2
    assert [(job.ticker, job.year) for job, _ in report.failed] == [("ENRN", 2019), ("ENRN", 2020)]
    assert all("404" in message for _, message in report.failed)
    assert report.fetched + report.skipped + len(report.failed) == len(jobs)


def test_a_bug_is_raised_not_recorded_as_a_failed_ticker(
    api: FakeMarketAPI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`return_exceptions=True` also catches your TypeErrors. Don't write them down; raise."""

    def broken(payload: object) -> None:
        raise TypeError("a bug in to_frame")

    monkeypatch.setattr("ingest.run.to_frame", broken)
    with pytest.raises(TypeError, match="a bug in to_frame"):
        ingest(api, JOBS, tmp_path)


# --- concurrency ---------------------------------------------------------------------


def test_requests_actually_overlap(tmp_path: Path) -> None:
    """8 one-month requests x 100ms: ~0.1s concurrently, 0.8s one at a time.

    If this fails, something in your code blocks the event loop -- a
    `time.sleep`, or a synchronous HTTP client inside an `async def`.
    """
    api = FakeMarketAPI(latency=0.1)
    tickers = ["AAPL", "MSFT", "SPY", "QQQ", "IBM", "KO", "XOM", "SAP.DE"]
    jobs = list(make_jobs(tickers, date(2019, 1, 1), date(2019, 1, 31)))
    report = ingest(api, jobs, tmp_path, concurrency=8)
    assert api.peak_in_flight == 8
    assert report.seconds < 0.4


def test_the_semaphore_caps_requests_in_flight(tmp_path: Path) -> None:
    api = FakeMarketAPI(latency=0.02)
    jobs = list(make_jobs(["AAPL", "MSFT", "SPY"], date(2017, 1, 1), date(2020, 12, 31)))
    ingest(api, jobs, tmp_path, concurrency=3)
    assert api.peak_in_flight == 3


def test_too_wide_and_the_vendor_pushes_back(tmp_path: Path) -> None:
    """Ask for 12 at once from a vendor that allows 4: it answers with 429s.

    Retries rescue some of them, but only by accident -- the semaphore is the fix.
    """
    api = FakeMarketAPI(latency=0.02, max_concurrent=4)
    jobs = list(make_jobs(["AAPL", "MSFT", "SPY"], date(2017, 1, 1), date(2020, 12, 31)))
    ingest(api, jobs, tmp_path, concurrency=12)
    assert len(api.requests) > api.served

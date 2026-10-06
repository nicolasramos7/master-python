"""Sitting 1, step 2. Generators: lazy values, and lazy errors."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from ingest.jobs import Job, make_jobs, pending, year_windows


def test_job_path(tmp_path: Path) -> None:
    job = Job("SAP.DE", date(2015, 3, 10), date(2015, 12, 31))
    assert job.path(tmp_path) == tmp_path / "SAP.DE" / "2015.parquet"


def test_year_windows_split_on_calendar_years() -> None:
    windows = list(year_windows(date(2015, 3, 10), date(2017, 6, 30)))
    assert windows == [
        (date(2015, 3, 10), date(2015, 12, 31)),
        (date(2016, 1, 1), date(2016, 12, 31)),
        (date(2017, 1, 1), date(2017, 6, 30)),
    ]


def test_year_windows_single_day() -> None:
    day = date(2020, 2, 29)
    assert list(year_windows(day, day)) == [(day, day)]


def test_year_windows_is_an_iterator_not_a_list() -> None:
    """Lazy: nothing is computed until someone asks for the next window."""
    windows = year_windows(date(2015, 1, 1), date(2024, 12, 31))
    assert isinstance(windows, Iterator)
    assert next(windows) == (date(2015, 1, 1), date(2015, 12, 31))


def test_year_windows_rejects_an_inverted_range_at_call_time() -> None:
    """The generator trap. Note there's no list() and no next() here.

    A plain generator function would NOT raise on this line -- its body hasn't
    started running yet. It would raise later, wherever it was first iterated.
    """
    with pytest.raises(ValueError, match="after"):
        year_windows(date(2020, 1, 1), date(2019, 1, 1))


def test_make_jobs_is_every_ticker_times_every_year() -> None:
    jobs = list(make_jobs(["MSFT", "AAPL"], date(2015, 6, 1), date(2017, 3, 1)))
    assert [(j.ticker, j.year) for j in jobs] == [
        ("MSFT", 2015),
        ("MSFT", 2016),
        ("MSFT", 2017),
        ("AAPL", 2015),
        ("AAPL", 2016),
        ("AAPL", 2017),
    ]
    assert jobs[0] == Job("MSFT", date(2015, 6, 1), date(2015, 12, 31))
    assert jobs[-1] == Job("AAPL", date(2017, 1, 1), date(2017, 3, 1))


def test_make_jobs_dedupes_tickers_and_keeps_their_order() -> None:
    jobs = make_jobs(["SPY", "AAPL", "SPY"], date(2020, 1, 1), date(2020, 12, 31))
    assert [j.ticker for j in jobs] == ["SPY", "AAPL"]


def test_pending_skips_jobs_whose_file_exists(tmp_path: Path) -> None:
    jobs = list(make_jobs(["AAPL"], date(2015, 1, 1), date(2017, 12, 31)))
    done = jobs[1].path(tmp_path)
    done.parent.mkdir(parents=True)
    done.touch()
    assert list(pending(jobs, tmp_path)) == [jobs[0], jobs[2]]


def test_pending_checks_each_file_when_it_gets_there(tmp_path: Path) -> None:
    """Lazy, again: a file that appears mid-iteration is seen.

    `[j for j in jobs if ...]` would have checked every file up front.
    """
    jobs = list(make_jobs(["AAPL"], date(2015, 1, 1), date(2016, 12, 31)))
    todo = pending(jobs, tmp_path)
    assert next(todo) == jobs[0]
    jobs[1].path(tmp_path).parent.mkdir(parents=True)
    jobs[1].path(tmp_path).touch()
    assert list(todo) == []

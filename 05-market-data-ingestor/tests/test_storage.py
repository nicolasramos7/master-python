"""Sitting 1, step 5. A crash must never leave a file that looks finished."""

from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from ingest.bars import Bar, empty_frame, to_frame
from ingest.fake_api import bars_for, exchange_for
from ingest.jobs import Job, pending
from ingest.storage import atomic_write_parquet, read_all, snapshot


def frame(symbol: str = "AAPL", year: int = 2019) -> pd.DataFrame:
    bars = cast(list[Bar], bars_for(symbol, date(year, 1, 1), date(year, 12, 31)))
    return to_frame({"symbol": symbol, "timezone": exchange_for(symbol).timezone, "bars": bars})


def crash_halfway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every to_parquet write half a file, then die -- like a kill -9 mid-write."""

    def half_written(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        Path(path).write_bytes(b"PAR1 and then the power went out")
        raise OSError("disk yanked")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", half_written)


# --- atomic writes ------------------------------------------------------------


def test_round_trips_exactly(tmp_path: Path) -> None:
    original = frame()
    path = tmp_path / "AAPL" / "2019.parquet"
    atomic_write_parquet(original, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path), original)


def test_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "deep" / "er" / "2019.parquet"
    atomic_write_parquet(frame(), path)
    assert path.is_file()


def test_leaves_only_the_target_behind(tmp_path: Path) -> None:
    path = tmp_path / "AAPL" / "2019.parquet"
    atomic_write_parquet(frame(), path)
    assert list(path.parent.iterdir()) == [path]


def test_a_crash_mid_write_leaves_no_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The error still propagates, and there is nothing on disk for `pending` to trust."""
    path = tmp_path / "AAPL" / "2019.parquet"
    crash_halfway(monkeypatch)
    with pytest.raises(OSError, match="disk yanked"):
        atomic_write_parquet(frame(), path)
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


def test_a_crash_mid_rewrite_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "AAPL" / "2019.parquet"
    atomic_write_parquet(frame(), path)
    before = path.read_bytes()
    crash_halfway(monkeypatch)
    with pytest.raises(OSError):
        atomic_write_parquet(frame(), path)
    assert path.read_bytes() == before


def test_why_it_matters_pending_trusts_any_file(tmp_path: Path) -> None:
    """Passes as soon as `pending` works, whatever storage.py does -- it documents the trap.

    A half-written file at the final path is indistinguishable from a finished
    one as far as `pending` is concerned. That year is skipped on every future run.
    """
    job = Job("AAPL", date(2019, 1, 1), date(2019, 12, 31))
    job.path(tmp_path).parent.mkdir(parents=True)
    job.path(tmp_path).write_bytes(b"PAR1 truncated")
    assert list(pending([job], tmp_path)) == []


# --- reading back -----------------------------------------------------------------


def test_read_all_concatenates_and_sorts(tmp_path: Path) -> None:
    for symbol in ("MSFT", "AAPL"):
        for year in (2020, 2019):
            atomic_write_parquet(frame(symbol, year), tmp_path / symbol / f"{year}.parquet")
    everything = read_all(tmp_path)
    weekdays = len(pd.bdate_range("2019-01-01", "2020-12-31"))
    assert len(everything) == 2 * weekdays
    assert everything["ticker"].tolist() == sorted(everything["ticker"].tolist())
    assert everything.groupby("ticker")["date"].is_monotonic_increasing.all()
    assert not everything.duplicated(["ticker", "date"]).any()
    assert everything.index.tolist() == list(range(len(everything)))


def test_read_all_of_nothing_is_an_empty_frame_with_the_schema(tmp_path: Path) -> None:
    result = read_all(tmp_path)
    assert result.empty
    assert result.dtypes.equals(empty_frame().dtypes)


def test_read_all_ignores_leftover_temp_files(tmp_path: Path) -> None:
    atomic_write_parquet(frame(), tmp_path / "AAPL" / "2019.parquet")
    (tmp_path / "AAPL" / "2020.parquet.tmp").write_bytes(b"PAR1 junk")
    assert len(read_all(tmp_path)) == len(frame())


# --- the idempotency fingerprint ---------------------------------------------------


def test_snapshot_keys_are_relative_posix_paths(tmp_path: Path) -> None:
    atomic_write_parquet(frame(), tmp_path / "AAPL" / "2019.parquet")
    assert list(snapshot(tmp_path)) == ["AAPL/2019.parquet"]


def test_snapshot_compares_directories_by_content(tmp_path: Path) -> None:
    for root in (tmp_path / "a", tmp_path / "b"):
        atomic_write_parquet(frame(), root / "AAPL" / "2019.parquet")
    assert snapshot(tmp_path / "a") == snapshot(tmp_path / "b")
    (tmp_path / "b" / "AAPL" / "2019.parquet").write_bytes(b"different")
    assert snapshot(tmp_path / "a") != snapshot(tmp_path / "b")

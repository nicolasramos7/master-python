"""Sitting 2, step 3. The command line: parse at the boundary, report with exit codes."""

from datetime import date
from pathlib import Path

import pytest

from ingest import cli
from ingest.cli import BenchArgs, FetchArgs, main, parse_args
from ingest.client import API_KEY_VAR
from ingest.fake_api import API_KEY, LAST_DATE, FakeMarketAPI

# --- parse_args -------------------------------------------------------------------


def test_parses_a_fetch() -> None:
    args = parse_args(
        ["fetch", "--tickers", "aapl, msft,,AAPL", "--start", "2015-01-01", "--end", "2016-06-30",
         "--out", "somewhere", "--concurrency", "3", "-v"]
    )  # fmt: skip
    assert args == FetchArgs(
        tickers=("AAPL", "MSFT"),
        start=date(2015, 1, 1),
        end=date(2016, 6, 30),
        out=Path("somewhere"),
        concurrency=3,
        verbose=True,
    )


def test_fetch_defaults() -> None:
    args = parse_args(["fetch", "--tickers", "SPY", "--start", "2020-01-01"])
    assert isinstance(args, FetchArgs)
    assert (args.end, args.out, args.concurrency, args.verbose) == (
        LAST_DATE,
        Path("data"),
        8,
        False,
    )


def test_parses_a_bench() -> None:
    assert parse_args(["bench"]) == BenchArgs(quick=False)
    assert parse_args(["bench", "--quick"]) == BenchArgs(quick=True)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["fetch", "--start", "2020-01-01"],
        ["fetch", "--tickers", ",, ,", "--start", "2020-01-01"],
        ["fetch", "--tickers", "SPY", "--start", "01/02/2020"],
        ["fetch", "--tickers", "SPY", "--start", "2020-01-01", "--concurrency", "0"],
        ["fetch", "--tickers", "SPY", "--start", "2020-01-01", "--concurrency", "lots"],
        ["fetch", "--tickers", "SPY", "--start", "2021-01-01", "--end", "2020-01-01"],
    ],
    ids=[
        "no-command",
        "no-tickers",
        "blank-tickers",
        "bad-date",
        "zero",
        "not-a-number",
        "backwards",
    ],
)
def test_bad_arguments_exit_2_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    """argparse's job, via `type=` and `parser.error` -- no traceback, just usage."""
    with pytest.raises(SystemExit) as info:
        parse_args(argv)
    assert info.value.code == 2
    assert "usage:" in capsys.readouterr().err


# --- main ---------------------------------------------------------------------------


@pytest.fixture
def fast_vendor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the CLI's slow, flaky vendor for an instant, reliable one. Key set."""
    monkeypatch.setattr(cli, "make_api", lambda: FakeMarketAPI())
    monkeypatch.setenv(API_KEY_VAR, API_KEY)


@pytest.mark.usefixtures("fast_vendor")
def test_fetch_then_refetch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["fetch", "--tickers", "AAPL,SPY", "--start", "2020-01-01", "--end", "2020-12-31"]
    argv += ["--out", str(tmp_path)]
    assert main(argv) == 0
    assert "fetched 2, skipped 0, failed 0" in capsys.readouterr().out
    assert main(argv) == 0
    assert "fetched 0, skipped 2, failed 0" in capsys.readouterr().out


@pytest.mark.usefixtures("fast_vendor")
def test_failed_jobs_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["fetch", "--tickers", "AAPL,ENRN", "--start", "2020-06-01", "--end", "2020-12-31"]
    argv += ["--out", str(tmp_path)]
    assert main(argv) == 1
    assert "ENRN" in capsys.readouterr().out


def test_missing_key_exits_2_and_says_which_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    argv = ["fetch", "--tickers", "AAPL", "--start", "2020-01-01", "--out", str(tmp_path)]
    assert main(argv) == 2
    assert API_KEY_VAR in capsys.readouterr().err
    assert not tmp_path.joinpath("AAPL").exists()

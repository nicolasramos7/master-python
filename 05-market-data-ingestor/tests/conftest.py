"""Shared helpers. The async boundary lives here, in one place.

These tests don't use a pytest plugin for async. A test that needs the event
loop calls `asyncio.run(...)` itself -- exactly what `cli.main` does. It is a
little more typing, and it keeps the one fact that matters in plain sight:
async code only runs inside an event loop, and something synchronous has to
start one.

`client` and `run` are imported inside the helpers, not at the top. Importing
`client.py` applies your `@retry` decorator, and until `backoff.py` works that
import fails -- which would take every test file down with it, including the
ones for modules that don't use it.
"""

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ingest.fake_api import API_KEY, FakeMarketAPI
from ingest.jobs import Job

if TYPE_CHECKING:
    from ingest.bars import Payload
    from ingest.run import Report


@pytest.fixture
def api() -> FakeMarketAPI:
    """A fresh, fast, well-behaved vendor. Counters start at zero."""
    return FakeMarketAPI()


def fetch(api: FakeMarketAPI, job: Job, api_key: str = API_KEY) -> "Payload":
    """`fetch_bars` for one job, from synchronous test code."""
    from ingest.client import fetch_bars, make_client

    async def go() -> "Payload":
        async with make_client(api, api_key) as client:
            return await fetch_bars(client, job)

    return asyncio.run(go())


def ingest(api: FakeMarketAPI, jobs: Iterable[Job], out: Path, concurrency: int = 4) -> "Report":
    """`run` for a list of jobs, from synchronous test code."""
    from ingest.client import make_client
    from ingest.run import run

    async def go() -> "Report":
        async with make_client(api, API_KEY) as client:
            return await run(jobs, client=client, out=out, concurrency=concurrency)

    return asyncio.run(go())

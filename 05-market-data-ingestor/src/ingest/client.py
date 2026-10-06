"""Sitting 2, part 1: one HTTP request, done properly. Your first `async def`.

`fetch_bars` makes one `GET` for one job and turns the response into either a
`Payload` or an exception from `errors.py`. It sorts failures into the two
kinds the retry decorator cares about:

    status                    raise              retried?
    ------                    -----              --------
    200                       (return payload)
    429, 500-599              TransientError     yes
    any other 4xx             PermanentError     no -- 404 won't become 200
    httpx.TransportError      TransientError     yes -- a dropped connection

`raise TransientError(...) from exc` when wrapping an httpx exception, so the
original stays attached to the traceback.

The request: `GET /v1/bars/{ticker}` with query params `start` and `end` as
ISO dates. Pass them with `params={...}`, never by formatting the URL yourself.

**Secrets come from the environment.** `api_key_from_env` reads
`MARKETDATA_API_KEY` from `os.environ`. A missing or empty key raises
`ConfigError` whose message names the *variable* -- and the key itself must
never appear in an error message or a log line. (Logs get shipped, pasted into
tickets, and screenshotted. A key in a log is a leaked key.) The key rides in
an `Authorization: Bearer ...` header, set once on the client.

`make_client` is given: it points an `httpx.AsyncClient` at the fake vendor.
An `AsyncClient` holds a connection pool, so make **one** per run and share it
across every request; use it as `async with make_client(...) as client:` so the
pool is closed even if the run fails.

The Python in it: `async def`, `await`, `httpx.AsyncClient.get`, status codes,
`typing.cast`, exception chaining, `os.environ`, applying your own decorator.
"""

# ^ The imports below are what the reference solution uses -- a hint, not a
#   requirement. Delete the line above when you're done, and ruff will flag any
#   import you ended up not needing.
import os
from typing import cast

import httpx

from .backoff import retry
from .bars import Payload
from .errors import ConfigError, PermanentError, TransientError
from .fake_api import BASE_URL, FakeMarketAPI
from .jobs import Job

API_KEY_VAR = "MARKETDATA_API_KEY"


def api_key_from_env(var: str = API_KEY_VAR) -> str:
    """The API key from the environment.

    Raises:
        ConfigError: if `var` is unset or blank. The message names `var`.
    """
    key = os.environ.get(var, "")
    key = key.strip()
    if not key:
        raise ConfigError(f"set the {var} environment variable to your API key")
    return key


def make_client(api: FakeMarketAPI, api_key: str) -> httpx.AsyncClient:
    """An AsyncClient wired to the fake vendor, with the key in every request. Given."""
    return httpx.AsyncClient(
        transport=api.transport(),
        base_url=BASE_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10.0,
    )


@retry(attempts=5, base_delay=0.01, max_delay=0.5, retry_on=(TransientError,))
async def fetch_bars(client: httpx.AsyncClient, job: Job) -> Payload:
    """GET one job's window. Retried on `TransientError` by the decorator above.

    Raises:
        TransientError: on 429, 5xx, or a transport error (after retries run out).
        PermanentError: on any other non-200 status.
    """
    params = {"start": job.start.isoformat(), "end": job.end.isoformat()}
    try:
        response = await client.get(f"/v1/bars/{job.ticker}", params=params)
    except httpx.TransportError as exc:
        raise TransientError(f"{job.ticker} {job.year}: {exc!r}") from exc

    status_code = response.status_code
    if status_code == 200:
        return cast(Payload, response.json())
    message = f"{job.ticker} {job.year}: HTTP {status_code}"
    if status_code == 429 or status_code >= 500:
        raise TransientError(message)
    raise PermanentError(message)


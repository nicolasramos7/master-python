"""The exception hierarchy. GIVEN -- the same idea as Project 1's, one level up.

The split that matters is transient vs permanent, because it decides what the
retry decorator does:

    IngestError                 anything this package raises on purpose
    ├── TransientError          try again later: 429, 5xx, a dropped connection
    ├── PermanentError          trying again will not help: 401, 404, bad input
    └── ConfigError             the program was started wrong: a missing API key

`run()` catches `IngestError` per job and records it in the report. Anything
that is *not* an `IngestError` is a bug in your code, and must crash the run
rather than be written down as "ticker failed".
"""


class IngestError(Exception):
    """Base class. Catch this, never bare `Exception`."""


class TransientError(IngestError):
    """The request might succeed if repeated. The retry decorator retries these."""


class PermanentError(IngestError):
    """The request will fail the same way every time. Never retried."""


class ConfigError(IngestError):
    """The program was misconfigured. Reported to the user, exit code 2."""

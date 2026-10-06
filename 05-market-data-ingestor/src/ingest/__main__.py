"""Entry point: `uv run python -m ingest fetch ...` or `uv run python -m ingest bench`.

Everything lives in `cli.py`. This file only runs it.

The `if __name__ == "__main__":` guard is load-bearing, not boilerplate. The
benchmark starts a process pool, and on macOS each worker process *imports*
this module to find its work. Without the guard, every worker would run
`main()` again -- start its own pool, which starts its own workers...
"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

# Project 1: Double-Entry Ledger

**Framing question:** Can I build a ledger where money is never created or destroyed, and prove it two independent ways?

## Running it

```bash
uv sync
uv run pytest
uv run mypy src
uv run ruff check .
```

## Design

A `Transaction` is a set of `Posting`s that must sum to zero. A posting is one side of a movement: an account ID and a signed amount. Balances are never stored, only derived by replaying the log.

Deposits from outside the system post against `EXTERNAL`, a boundary account allowed to go negative. Its balance is the total owed to depositors.

Validation is layered: `Transaction` enforces what it can see (at least two postings, sum is zero), `Ledger` enforces what needs context (accounts registered, overdraft rules).

## Findings

Money as `float` is broken, but intermittently, which is worse than being broken consistently. My first balanced test used 0.1 + 0.1 - 0.2 and passed, because doubling a float is exact in binary (only the exponent changes). Switching to 0.1 + 0.2 - 0.3 failed: those are different values, each stored as the nearest representable binary fraction, and the residue was about 5.5e-17.

Three fixes were available. Tolerance comparison (`abs(total) < 1e-9`) was rejected: residues accumulate across millions of rows and balances stop reconciling exactly against a bank statement. Integer minor units and `Decimal` both give exact decimal arithmetic. I chose `Decimal` for readability and explicit rounding control. The zero-sum check itself did not change at all, because `Decimal` implements the same operators. The type was the bug, not the logic.

`Decimal` must be constructed from strings. `Decimal(0.1)` faithfully copies the float's existing error.

## Caveats

- Amounts are annotation-typed only. A caller passing a `float` at runtime goes through silently. I chose to trust the types rather than validate in `__post_init__`.
- `AccountId` is a `NewType`, so it is a plain `str` at runtime. It protects against mypy-visible mistakes only.
- `balance()` is O(n) over every posting in the ledger. Fine at this scale, wrong at ten million rows. Real systems use snapshots or a write-invalidated cache.
- Overdraft rejection is implemented but has no test. Untested code is not working code.
- `statement()` duplicates the iteration in `balance()` on purpose. Refactoring to share it would make the verification hook meaningless, since two paths that call the same function are not independent. DRY is wrong here.
- `open_account` raises on a duplicate rather than overwriting, because silently changing `allow_negative` on an account with history would be worse than an exception.

## Verification hook

`balance(account)` sums all postings for that account. `statement(account)` walks the log accumulating a running total. The test asserts they agree, and also asserts against 120, a value computed by hand. Agreement alone is not enough, since two broken implementations can agree.

## Python used

**Types and data:** `@dataclass`, `frozen=True`, `field`, `__post_init__`, `object.__setattr__` to mutate inside a frozen instance, `typing.NewType`, `decimal.Decimal`, `datetime.date`, generic annotations (`list[T]`, `dict[K, V]`, `tuple[T, ...]`).

**Dunders:** `__init__`, `__repr__`, `__eq__`, `__hash__` (free from frozen dataclasses), `__neg__` on `Decimal`. Implement the dunder, get the syntax.

**Errors:** custom exception hierarchy under one `LedgerError` base, exceptions carrying data as attributes rather than only a message.

**Idioms:** `sum()` with a generator expression and `start=`, nested comprehensions, `dict.get` with a default, tuple unpacking, negative indexing, `_` for unused bindings, `in` for membership.

**Tooling:** `uv` for env and deps, pytest (`pytest.raises`, plain `assert`), mypy `--strict`, ruff.

## What I learned

Bugs that raise are cheap. The expensive ones return a confident wrong number: a float sum passing on one arrangement and failing on another, two classes accidentally sharing a name so the second silently overwrites the first, a transaction half-applied when validation raises mid-loop.

A circular import between `models` and `errors` was not fixable with import tricks. Both needed `AccountId`, so it moved into a third module that neither depends on. Dependencies must flow one direction.

Validate completely, then mutate once. Interleaving checks and mutations produced a ledger holding two postings that did not sum to zero, with no traceback.
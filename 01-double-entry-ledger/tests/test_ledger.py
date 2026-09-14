from datetime import date
from decimal import Decimal

import pytest

from ledger.errors import UnbalancedTransaction
from ledger.ids import AccountId
from ledger.models import Account, Ledger, Posting, Transaction


def test_unbalanced_postings_are_rejected() -> None:
    # two postings that dont add to zero
    p1 = Posting(AccountId("ID1"), amount=Decimal("100"))
    p2 = Posting(AccountId("ID2"), amount=Decimal("-90"))

    sample_date = date(2026, 8, 26)

    sample_description = "Sample Transaction"

    with pytest.raises(UnbalancedTransaction):
        Transaction(postings=(p1, p2), value_date=sample_date, description=sample_description)


def test_balanced_transaction_is_accepted() -> None:
    p1 = Posting(AccountId("ID1"), amount=Decimal("0.1"))
    p2 = Posting(AccountId("ID2"), amount=Decimal("0.1"))
    p3 = Posting(AccountId("ID3"), amount=Decimal("-0.2"))

    t = Transaction(postings=(p1, p2, p3), value_date=date(2026, 8, 26), description="Split")

    assert isinstance(t.postings, tuple)
    assert t.description == "Split"
    assert t.postings == (p1, p2, p3)


def test_balanced_transaction_is_accepted_decimal_friendly() -> None:
    p1 = Posting(AccountId("ID1"), amount=Decimal("0.1"))
    p2 = Posting(AccountId("ID2"), amount=Decimal("0.2"))
    p3 = Posting(AccountId("ID3"), amount=Decimal("-0.3"))

    t = Transaction(postings=(p1, p2, p3), value_date=date(2026, 8, 26), description="Split")

    assert isinstance(t.postings, tuple)
    assert t.description == "Split"
    assert t.postings == (p1, p2, p3)


def test_deposit_credits_customer_and_debits_external() -> None:
    ledger = Ledger()
    ledger.open_account(Account(AccountId("EXTERNAL"), allow_negative=True))
    ledger.open_account(Account(AccountId("ALICE"), allow_negative=False))

    t = Transaction(
        postings=(
            Posting(AccountId("EXTERNAL"), Decimal("-100")),
            Posting(AccountId("ALICE"), Decimal("100")),
        ),
        value_date=date(2026, 8, 31),
        description="Deposit",
    )
    ledger.post(t)

    assert ledger.balance(AccountId("ALICE")) == Decimal("100")
    assert ledger.balance(AccountId("EXTERNAL")) == Decimal("-100")


def test_statement() -> None:
    ledger = Ledger()
    ledger.open_account(Account(AccountId("EXTERNAL"), allow_negative=True))
    ledger.open_account(Account(AccountId("ALICE"), allow_negative=False))

    external = AccountId("EXTERNAL")
    alice = AccountId("ALICE")

    ledger.post(transfer(external, alice, Decimal("100"), "Deposit"))
    ledger.post(transfer(alice, external, Decimal("30"), "Withdrawal"))
    ledger.post(transfer(external, alice, Decimal("50"), "Deposit"))

    _, last_running = ledger.statement(alice)[-1]

    assert len(ledger.statement(alice)) == 3
    assert last_running == ledger.balance(alice)
    assert last_running == Decimal("120")

    total = sum(
        (p.amount for t in ledger.transactions for p in t.postings),
        start=Decimal("0"),
    )
    assert total == Decimal("0")


def transfer(frm: AccountId, to: AccountId, amount: Decimal, description: str) -> Transaction:
    return Transaction(
        postings=(
            Posting(frm, -amount),
            Posting(to, amount),
        ),
        value_date=date(2026, 9, 5),
        description=description,
    )

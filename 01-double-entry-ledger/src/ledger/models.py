from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from ledger.errors import (
    AccountAlreadyExists,
    InsufficientFunds,
    InsufficientPostings,
    UnbalancedTransaction,
    UnknownAccount,
)
from ledger.ids import AccountId


@dataclass(frozen=True)
class Posting:
    account: AccountId
    amount: Decimal


@dataclass(frozen=True)
class Transaction:
    postings: tuple[Posting, ...]
    value_date: date
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "postings", tuple(self.postings))

        if len(self.postings) < 2:
            raise InsufficientPostings(len(self.postings))

        total = sum(p.amount for p in self.postings)
        if total != 0:
            raise UnbalancedTransaction(total)


@dataclass(frozen=True)
class Account:
    id: AccountId
    allow_negative: bool = False


class Ledger:
    def __init__(self) -> None:
        self.transactions: list[Transaction] = []
        self.accounts: dict[AccountId, Account] = {}

    def open_account(self, account: Account) -> None:
        if account.id in self.accounts:
            raise AccountAlreadyExists(account.id)
        self.accounts[account.id] = account

    def post(self, transaction: Transaction) -> None:
        for p in transaction.postings:
            if p.account not in self.accounts:
                raise UnknownAccount(p.account)

        net_change: dict[AccountId, Decimal] = {}

        for p in transaction.postings:
            net_change[p.account] = net_change.get(p.account, Decimal("0")) + p.amount

        for account_id, change in net_change.items():
            account = self.accounts[account_id]
            if account.allow_negative:
                continue
            current = self.balance(account_id)
            projected = current + change
            if projected < 0:
                raise InsufficientFunds(account_id, change, current)

        self.transactions.append(transaction)

    def balance(self, account_id: AccountId) -> Decimal:
        if account_id not in self.accounts:
            raise UnknownAccount(account_id)

        total = Decimal("0")
        for transaction in self.transactions:
            for posting in transaction.postings:
                if posting.account == account_id:
                    total += posting.amount
        return total

    def statement(self, account_id: AccountId) -> list[tuple[Posting, Decimal]]:
        if account_id not in self.accounts:
            raise UnknownAccount(account_id)

        result: list[tuple[Posting, Decimal]] = []
        running = Decimal("0")

        for t in self.transactions:
            for p in t.postings:
                if p.account == account_id:
                    running = running + p.amount
                    result.append((p, running))
        return result

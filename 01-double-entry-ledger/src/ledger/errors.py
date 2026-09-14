from decimal import Decimal


class LedgerError(Exception):
    """Base for every error this package raises."""


class UnbalancedTransaction(LedgerError):
    def __init__(self, total: Decimal) -> None:
        super().__init__(f"Postings must sum to zero, got {total}")
        self.total = total


class InsufficientPostings(LedgerError):
    def __init__(self, number_of_postings: int) -> None:
        super().__init__(f"Insufficient postings, only got {number_of_postings}")
        self.count = number_of_postings


class UnknownAccount(LedgerError):
    def __init__(self, account: str) -> None:
        super().__init__(f"Account not registered: {account}")
        self.account = account


class InsufficientFunds(LedgerError):
    def __init__(self, account: str, requested: Decimal, available: Decimal) -> None:
        super().__init__(f"Account {account} has {available}, cannot post {requested}")
        self.account = account
        self.requested = requested
        self.available = available


class AccountAlreadyExists(LedgerError):
    def __init__(self, account: str) -> None:
        super().__init__(f"Account already registered: {account}")
        self.account = account

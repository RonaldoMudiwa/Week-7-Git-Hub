"""A customer complaint.

Each one is what a customer wrote to a bank, plus the labels the database
already gives it: which product it's about (like "Mortgage") and which
issue (like "Trouble during payment process"). The labels are what lets us
check later whether our program sorts complaints correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .exceptions import ValidationError


@dataclass(frozen=True)
class Complaint:
    complaint_id: str
    received: date
    product: str
    issue: str
    narrative: str          # what the customer actually wrote
    company: str = ""
    state: str = ""

    def __post_init__(self) -> None:
        for name in ("complaint_id", "product", "narrative"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValidationError(f"A complaint needs a {name.replace('_', ' ')}.")
        if not isinstance(self.received, date):
            raise ValidationError("received must be a date.")
        # Frozen, so tidied values are set this way.
        for name in ("complaint_id", "product", "issue", "narrative", "company", "state"):
            object.__setattr__(self, name, (getattr(self, name) or "").strip())

    @property
    def word_count(self) -> int:
        return len(self.narrative.split())

    def short(self, length: int = 60) -> str:
        """The start of the narrative, for showing in a list."""
        if len(self.narrative) <= length:
            return self.narrative
        return self.narrative[:length - 3].rstrip() + "..."

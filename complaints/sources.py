"""Where complaints come from.

ComplaintSource is the plug. CsvComplaintSource reads the CFPB's CSV file
(or our made up sample, which has the same columns). A source for a
database or a live feed could be added as another class later, and the
rest of the program wouldn't notice.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator

from .exceptions import DataError, ValidationError
from .models import Complaint

# The CFPB file's column names -> our field names.
COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "received",
    "Product": "product",
    "Issue": "issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
}
REQUIRED = ("Complaint ID", "Date received", "Product", "Consumer complaint narrative")


class ComplaintSource(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Complaint]:
        """Complaints one at a time, so a huge file never has to fit in memory."""

    def load(self, limit: int | None = None) -> list[Complaint]:
        """Read complaints into a list. limit stops early, handy with big files."""
        complaints = []
        for complaint in self:
            if limit is not None and len(complaints) >= limit:
                break
            complaints.append(complaint)
        return complaints


class ListSource(ComplaintSource):
    """Complaints already in memory. Mostly for tests."""

    def __init__(self, complaints: Iterable[Complaint]) -> None:
        self._complaints = list(complaints)

    def __iter__(self) -> Iterator[Complaint]:
        return iter(self._complaints)


@dataclass
class LoadReport:
    """What happened while reading a file, so nothing is skipped silently."""

    read: int = 0
    skipped_no_text: int = 0
    skipped_bad: int = 0


class CsvComplaintSource(ComplaintSource):
    """Reads a CFPB style CSV file.

    Most rows in the real database have no narrative, because customers
    can choose not to share what they wrote. Those rows are skipped and
    counted in .report.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.report = LoadReport()

    def __iter__(self) -> Iterator[Complaint]:
        self.report = LoadReport()
        try:
            # utf-8-sig quietly removes the invisible mark Excel adds to the start.
            with self.path.open(newline="", encoding="utf-8-sig") as file:
                reader = csv.DictReader(file)
                self._check_columns(reader.fieldnames or [])
                for row in reader:
                    complaint = self._to_complaint(row)
                    if complaint is not None:
                        self.report.read += 1
                        yield complaint
        except FileNotFoundError:
            raise DataError(f"No file at {self.path}.") from None
        except UnicodeDecodeError as error:
            raise DataError(f"{self.path} isn't a UTF-8 text file: {error}") from None

    def _check_columns(self, found: list[str]) -> None:
        missing = [name for name in REQUIRED if name not in found]
        if missing:
            raise DataError(f"{self.path} is missing columns: {', '.join(missing)}.")

    def _to_complaint(self, row: dict[str, str]) -> Complaint | None:
        values = {ours: (row.get(theirs) or "").strip() for theirs, ours in COLUMNS.items()}
        if not values["narrative"]:
            self.report.skipped_no_text += 1
            return None
        try:
            values["received"] = parse_date(values["received"])
            return Complaint(**values)
        except ValidationError:
            self.report.skipped_bad += 1
            return None


def parse_date(text: str) -> date:
    """The CFPB website and API use 2026-10-28, their bulk download uses 10/28/26."""
    for pattern in ("%Y-%m-%d", "%m/%d/%y", "%m/%d/%Y"):
        try:
            return datetime.strptime(text.strip()[:10], pattern).date()
        except ValueError:
            continue
    raise ValidationError(f"{text!r} isn't a date.")

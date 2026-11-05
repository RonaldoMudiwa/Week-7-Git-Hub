"""Shared test helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from complaints import Complaint

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_complaints.csv"


def make(narrative: str = "I was charged a late fee", product: str = "Credit card",
         issue: str = "Fees or interest", complaint_id: str = "1") -> Complaint:
    return Complaint(complaint_id, date(2026, 11, 5), product, issue, narrative,
                     "Northgate Bank", "CA")


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE

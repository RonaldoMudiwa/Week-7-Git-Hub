"""Tests for the Complaint model."""

from __future__ import annotations

from datetime import date

import pytest

from complaints import Complaint, ValidationError

from .conftest import make


def test_values_are_tidied():
    complaint = Complaint(" 42 ", date(2026, 11, 5), " Mortgage ", " Late ", "  Help me  ",
                          " Bank ", " TX ")
    assert (complaint.complaint_id, complaint.product, complaint.narrative) == ("42", "Mortgage",
                                                                                "Help me")
    assert (complaint.company, complaint.state) == ("Bank", "TX")


@pytest.mark.parametrize("field", ["complaint_id", "product", "narrative"])
@pytest.mark.parametrize("bad", ["", "   ", None])
def test_required_fields(field, bad):
    values = dict(complaint_id="1", received=date(2026, 11, 5), product="Mortgage",
                  issue="", narrative="text")
    values[field] = bad
    with pytest.raises(ValidationError, match="needs a"):
        Complaint(**values)


def test_received_must_be_a_date():
    with pytest.raises(ValidationError, match="must be a date"):
        Complaint("1", "2026-11-05", "Mortgage", "", "text")


def test_issue_company_and_state_can_be_empty():
    complaint = Complaint("1", date(2026, 11, 5), "Mortgage", None, "text")
    assert complaint.issue == "" and complaint.company == ""


def test_word_count_and_short():
    complaint = make("one two three four five six seven eight nine ten eleven twelve")
    assert complaint.word_count == 12
    assert complaint.short(20) == "one two three fou..."
    assert make("short").short() == "short"


def test_frozen():
    with pytest.raises(AttributeError):
        make().product = "Mortgage"  # type: ignore[misc]

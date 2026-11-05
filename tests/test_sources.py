"""Tests for reading complaints from files."""

from __future__ import annotations

from datetime import date

import pytest

from complaints import (
    ComplaintSource,
    CsvComplaintSource,
    DataError,
    ListSource,
    ValidationError,
    parse_date,
)

from .conftest import make

HEADER = "Date received,Product,Issue,Consumer complaint narrative,Company,State,Complaint ID\n"


def write(tmp_path, text: str, encoding: str = "utf-8"):
    path = tmp_path / "complaints.csv"
    path.write_text(text, encoding=encoding)
    return path


def test_reads_the_sample(sample_path):
    source = CsvComplaintSource(sample_path)
    complaints = source.load()
    assert len(complaints) == 300
    assert source.report.read == 300
    assert {c.product for c in complaints} == {
        "Credit card", "Checking or savings account", "Mortgage", "Debt collection",
        "Credit reporting", "Money transfer"}
    assert all(c.narrative and c.received.year == 2026 for c in complaints)


def test_rows_without_a_narrative_are_skipped_and_counted(tmp_path):
    path = write(tmp_path, HEADER
                 + "2026-11-01,Mortgage,Late,My payment was lost,Bank,TX,1\n"
                 + "2026-11-01,Mortgage,Late,,Bank,TX,2\n"
                 + "not a date,Mortgage,Late,Some text,Bank,TX,3\n")
    source = CsvComplaintSource(path)
    complaints = source.load()
    assert [c.complaint_id for c in complaints] == ["1"]
    assert (source.report.read, source.report.skipped_no_text, source.report.skipped_bad) == (1, 1, 1)


def test_bulk_download_dates_and_excel_marker(tmp_path):
    # The CFPB bulk file uses 10/28/26 dates, and Excel adds an invisible mark at the start.
    path = write(tmp_path, HEADER + "10/28/26,Mortgage,Late,Lost payment,Bank,TX,1\n",
                 encoding="utf-8-sig")
    assert CsvComplaintSource(path).load()[0].received == date(2026, 10, 28)


def test_quotes_and_commas_in_the_text(tmp_path):
    path = write(tmp_path, HEADER + '2026-11-01,Mortgage,Late,"They said ""no"", twice",Bank,TX,1\n')
    assert CsvComplaintSource(path).load()[0].narrative == 'They said "no", twice'


def test_limit_stops_early(sample_path):
    assert len(CsvComplaintSource(sample_path).load(limit=10)) == 10


def test_missing_file(tmp_path):
    with pytest.raises(DataError, match="No file at"):
        CsvComplaintSource(tmp_path / "nope.csv").load()


def test_missing_columns(tmp_path):
    path = write(tmp_path, "Product,Issue\nMortgage,Late\n")
    with pytest.raises(DataError, match="missing columns: Complaint ID, Date received"):
        CsvComplaintSource(path).load()


def test_not_a_text_file(tmp_path):
    path = tmp_path / "complaints.csv"
    path.write_bytes(b"\xff\xfe\x00\x81" * 50)
    with pytest.raises(DataError, match="isn't a UTF-8 text file"):
        CsvComplaintSource(path).load()


def test_list_source():
    source = ListSource([make(complaint_id="1"), make(complaint_id="2")])
    assert isinstance(source, ComplaintSource)
    assert [c.complaint_id for c in source.load()] == ["1", "2"]


@pytest.mark.parametrize("text, expected", [
    ("2026-10-28", date(2026, 10, 28)), ("10/28/26", date(2026, 10, 28)),
    ("10/28/2026", date(2026, 10, 28)), ("2026-10-28T00:00:00", date(2026, 10, 28)),
])
def test_parse_date(text, expected):
    assert parse_date(text) == expected


def test_parse_date_refuses_nonsense():
    with pytest.raises(ValidationError):
        parse_date("yesterday")

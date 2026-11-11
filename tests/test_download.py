"""Tests for downloading from the CFPB. None of them go online."""

from __future__ import annotations

import io
import json
import urllib.error
from urllib.parse import parse_qs, urlparse

import pytest

from complaints import CfpbDownloader, CsvComplaintSource, DataError
from complaints import download as download_module

ITEM = {"date_received": "2026-11-02T12:00:00-05:00", "product": "Mortgage",
        "issue": "Trouble during payment process",
        "complaint_what_happened": "My payment was not applied.",
        "company": "Big Bank", "state": "TX", "complaint_id": 123}
NO_TEXT = dict(ITEM, complaint_id=124, complaint_what_happened="")


class FakeFetch:
    def __init__(self, answer) -> None:
        self.answer = answer
        self.urls: list[str] = []

    def __call__(self, url: str):
        self.urls.append(url)
        return self.answer


def test_url_asks_for_complaints_with_text():
    query = parse_qs(urlparse(CfpbDownloader().build_url(50, "Mortgage")).query)
    assert query["size"] == ["50"] and query["has_narrative"] == ["true"]
    assert query["product"] == ["Mortgage"] and query["format"] == ["json"]
    assert "product" not in parse_qs(urlparse(CfpbDownloader().build_url(5)).query)


@pytest.mark.parametrize("answer", [
    [ITEM, NO_TEXT],                                                   # a plain list
    {"hits": {"hits": [{"_source": ITEM}, {"_source": NO_TEXT}]}},     # search engine style
])
def test_download_reads_both_answer_shapes(answer):
    rows = CfpbDownloader(FakeFetch(answer)).download(2)
    assert len(rows) == 1                                    # the one without text is dropped
    assert rows[0]["Consumer complaint narrative"] == "My payment was not applied."
    assert rows[0]["Complaint ID"] == "123"


def test_zero_and_missing_values():
    rows = CfpbDownloader(FakeFetch([dict(ITEM, complaint_id=0, state=None)])).download(1)
    assert rows[0]["Complaint ID"] == "0" and rows[0]["State"] == ""


def test_saved_file_reads_back_like_the_sample(tmp_path):
    downloader = CfpbDownloader(FakeFetch([ITEM]))
    path = downloader.save(downloader.download(1), tmp_path / "out" / "cfpb.csv")
    complaint = CsvComplaintSource(path).load()[0]
    assert (complaint.product, complaint.state) == ("Mortgage", "TX")
    assert complaint.received.isoformat() == "2026-11-02"


@pytest.mark.parametrize("count", [0, 10_001])
def test_count_limits(count):
    with pytest.raises(ValueError):
        CfpbDownloader(FakeFetch([])).download(count)


def test_strange_answer():
    with pytest.raises(DataError, match="shape"):
        CfpbDownloader(FakeFetch({"error": "nope"})).download(1)


class FakeAnswer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_fetch_json(monkeypatch):
    monkeypatch.setattr(download_module.urllib.request, "urlopen",
                        lambda request, timeout: FakeAnswer(json.dumps([ITEM]).encode()))
    assert download_module.fetch_json("https://example.test") == [ITEM]


def test_fetch_json_offline(monkeypatch):
    def offline(request, timeout):
        raise urllib.error.URLError("no internet")

    monkeypatch.setattr(download_module.urllib.request, "urlopen", offline)
    with pytest.raises(DataError, match="Couldn't reach the CFPB"):
        download_module.fetch_json("https://example.test")


def test_fetch_json_not_json(monkeypatch):
    monkeypatch.setattr(download_module.urllib.request, "urlopen",
                        lambda request, timeout: FakeAnswer(b"<html>busy</html>"))
    with pytest.raises(DataError, match="isn't JSON"):
        download_module.fetch_json("https://example.test")

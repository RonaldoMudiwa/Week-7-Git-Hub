"""Downloading real complaints from the CFPB.

The CFPB has a free public search API (no key needed). This asks it for
complaints that have a narrative and saves them in the same CSV columns as
the sample, so the rest of the program reads them the same way.

Note: this couldn't be tested against the real service from where it was
written (the network there blocks it), so the tests use saved answers in
the shape the API documents. The first real run is on your own computer.
"""

from __future__ import annotations

import csv
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .exceptions import DataError
from .sources import COLUMNS

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
MAX_COUNT = 10_000

# The API's field names -> the CSV column names the rest of the program reads.
API_FIELDS = {
    "date_received": "Date received",
    "product": "Product",
    "issue": "Issue",
    "complaint_what_happened": "Consumer complaint narrative",
    "company": "Company",
    "state": "State",
    "complaint_id": "Complaint ID",
}

Fetch = Callable[[str], Any]     # takes a URL, returns the decoded JSON


def fetch_json(url: str, timeout: float = 60) -> Any:
    """Ask the internet for a URL and read the JSON answer."""
    request = urllib.request.Request(url, headers={"User-Agent": "complaint-analyzer"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return json.loads(answer.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as error:
        raise DataError(f"Couldn't reach the CFPB: {error}") from None
    except ValueError as error:
        raise DataError(f"The CFPB sent something that isn't JSON: {error}") from None


class CfpbDownloader:
    def __init__(self, fetch: Fetch = fetch_json, url: str = API_URL) -> None:
        # The fetch function can be swapped for a fake one, so tests never go online.
        self._fetch = fetch
        self.url = url

    def build_url(self, count: int, product: str | None = None) -> str:
        params = {"size": count, "has_narrative": "true", "no_aggs": "true",
                  "sort": "created_date_desc", "format": "json"}
        if product:
            params["product"] = product
        return f"{self.url}?{urllib.parse.urlencode(params)}"

    def download(self, count: int = 1000, product: str | None = None) -> list[dict[str, str]]:
        """Complaints as rows with the CSV column names."""
        if not 1 <= count <= MAX_COUNT:
            raise ValueError(f"count must be between 1 and {MAX_COUNT}.")
        data = self._fetch(self.build_url(count, product))
        return [row for row in (self._to_row(item) for item in _records(data)) if row]

    def save(self, rows: list[dict[str, str]], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = list(COLUMNS)    # same order as the sample file's reader expects
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return path

    @staticmethod
    def _to_row(item: dict[str, Any]) -> dict[str, str] | None:
        # "is None" rather than "or", so a value like 0 isn't mistaken for missing.
        row = {column: "" if item.get(field) is None else str(item[field]).strip()
               for field, column in API_FIELDS.items()}
        return row if row["Consumer complaint narrative"] else None


def _records(data: Any) -> list[dict[str, Any]]:
    """The API can answer as a plain list of complaints, or wrapped the way
    its search engine (Elasticsearch) does: {"hits": {"hits": [{"_source": {...}}]}}."""
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and isinstance(data.get("hits"), dict):
        items = data["hits"].get("hits", [])
    else:
        raise DataError("The CFPB answer wasn't in a shape this program understands.")
    return [item.get("_source", item) for item in items if isinstance(item, dict)]

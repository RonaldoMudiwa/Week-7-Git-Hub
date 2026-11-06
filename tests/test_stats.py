"""Tests for word statistics."""

from __future__ import annotations

import pytest

from complaints import CsvComplaintSource, TextCleaner, WordStats, counts_by, top_issues

from .conftest import make

SMALL = [
    make("late fee charged on my card", "Credit card", "Fees", "1"),
    make("another late fee on the card", "Credit card", "Fees", "2"),
    make("card charged twice by a shop", "Credit card", "Purchase", "3"),
    make("escrow payment went up", "Mortgage", "Payment", "4"),
    make("my escrow is wrong and the fee too", "Mortgage", "Payment", "5"),
]


@pytest.fixture
def stats() -> WordStats:
    return WordStats()


def test_top_words(stats):
    assert stats.top_words(SMALL, 3) == [("card", 3), ("fee", 3), ("charged", 2)]


def test_top_phrases(stats):
    assert stats.top_phrases(SMALL, 2, 2) == [("late fee", 2), ("another late", 1)]


def test_word_counts_with_own_cleaner():
    stats = WordStats(TextCleaner(remove_stop_words=False))
    assert stats.word_counts([make("the the fee")])["the"] == 2


def test_distinctive_words(stats):
    card = dict(stats.distinctive_words(SMALL, "Credit card"))
    mortgage = [w for w, _ in stats.distinctive_words(SMALL, "Mortgage", 2)]
    assert mortgage == ["escrow", "payment"]
    assert card["card"] > card.get("fee", 0)       # "fee" also shows up in a mortgage complaint
    assert "escrow" not in card


def test_distinctive_words_by_issue(stats):
    words = [w for w, _ in stats.distinctive_words(SMALL, "Purchase", 1, key=lambda c: c.issue)]
    assert words == ["shop"]


def test_distinctive_words_for_an_unknown_label(stats):
    assert stats.distinctive_words(SMALL, "Pets") == []


def test_counts_by_product_and_issue():
    assert counts_by(SMALL) == [("Credit card", 3), ("Mortgage", 2)]
    assert counts_by(SMALL, lambda c: c.issue) == [("Fees", 2), ("Payment", 2), ("Purchase", 1)]


def test_top_issues():
    assert top_issues(SMALL, "Credit card") == [("Fees", 2), ("Purchase", 1)]
    assert top_issues(SMALL, "Pets") == []


def test_sample_has_sensible_clue_words(stats, sample_path):
    complaints = CsvComplaintSource(sample_path).load()
    clues = {product: [w for w, _ in stats.distinctive_words(complaints, product, 15)]
             for product, _ in counts_by(complaints)}
    assert "escrow" in clues["Mortgage"]
    assert "collector" in clues["Debt collection"]
    assert "transfer" in clues["Money transfer"]

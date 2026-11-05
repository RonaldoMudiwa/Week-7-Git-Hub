"""Tests for cleaning text and making tokens."""

from __future__ import annotations

import pytest

from complaints import STOP_WORDS, TextCleaner, ngrams


@pytest.fixture
def cleaner() -> TextCleaner:
    return TextCleaner()


def test_clean(cleaner):
    assert cleaner.clean("I was CHARGED a late-fee!!  Why?") == "i was charged a late fee why"


def test_hidden_details_are_removed(cleaner):
    text = "On XX/XX/2026 I called XXXX about account XXXX XXXX."
    assert cleaner.clean(text) == "on i called about account"


def test_amounts_become_the_word_money(cleaner):
    assert cleaner.tokens("They took {$1,250.00} and {$35.00}") == ["took", "money", "money"]


def test_apostrophes_join_words(cleaner):
    assert cleaner.clean("I don't know, they won't say") == "i dont know they wont say"


def test_tokens_drop_stop_words_and_tiny_words(cleaner):
    assert cleaner.tokens("I was charged a late fee of {$35.00} on my card x") == [
        "charged", "late", "fee", "money", "card"]


def test_stop_words_can_be_kept():
    assert TextCleaner(remove_stop_words=False).tokens("I was charged") == ["was", "charged"]


def test_custom_stop_words():
    cleaner = TextCleaner(stop_words=frozenset({"bank"}))
    assert cleaner.tokens("the bank charged me") == ["the", "charged", "me"]


def test_empty_text(cleaner):
    assert cleaner.tokens("") == [] and cleaner.tokens(None) == []


def test_stop_words_are_lower_case():
    assert "the" in STOP_WORDS and all(w == w.lower() for w in STOP_WORDS)


def test_ngrams():
    words = ["late", "fee", "charged", "twice"]
    assert ngrams(words, 1) == words
    assert ngrams(words, 2) == ["late fee", "fee charged", "charged twice"]
    assert ngrams(words, 3) == ["late fee charged", "fee charged twice"]
    assert ngrams(words, 5) == []


def test_ngram_size_must_be_positive():
    with pytest.raises(ValueError):
        ngrams(["a"], 0)

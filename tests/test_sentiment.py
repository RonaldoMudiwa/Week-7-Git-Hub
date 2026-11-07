"""Tests for the mood score."""

from __future__ import annotations

import pytest

from complaints import LexiconSentimentScorer, Mood, SentimentScorer


@pytest.fixture
def scorer() -> LexiconSentimentScorer:
    return LexiconSentimentScorer()


def test_it_is_a_scorer(scorer):
    assert isinstance(scorer, SentimentScorer)


def test_no_feeling_words_is_calm(scorer):
    sentiment = scorer.score("The interest rate went up")
    assert (sentiment.score, sentiment.words, sentiment.mood) == (0.0, (), Mood.CALM)


def test_angry_complaint(scorer):
    sentiment = scorer.score("This is a scam and totally unacceptable. The staff were rude.")
    assert sentiment.mood is Mood.ANGRY
    assert sentiment.words == ("scam", "unacceptable", "rude")


def test_happy_text(scorer):
    sentiment = scorer.score("Thank you, it was resolved and staff were helpful")
    assert sentiment.score > 0.3 and sentiment.mood is Mood.CALM


def test_not_flips_the_meaning(scorer):
    assert scorer.score("I am happy").score > 0
    assert scorer.score("I am not happy").score < 0
    assert scorer.score("I am not at all happy").score < 0    # "not" three words back
    assert scorer.score("not that I mind, I am happy").score > 0   # too far back to count


def test_very_makes_it_stronger(scorer):
    assert scorer.score("very rude").score < scorer.score("rude").score


def test_more_angry_words_score_lower(scorer):
    one = scorer.score("rude").score
    three = scorer.score("rude unfair terrible").score
    assert three < one


def test_score_stays_between_minus_1_and_1(scorer):
    text = " ".join(["extremely unacceptable"] * 50)
    assert scorer.score(text).score == -1.0


def test_hidden_details_and_money_are_ignored(scorer):
    assert scorer.score("XXXX {$500.00} XX/XX/2026").score == 0.0


@pytest.mark.parametrize("score, mood", [
    (0.3, Mood.CALM), (-0.1, Mood.CALM), (-0.2, Mood.UNHAPPY), (-0.5, Mood.UNHAPPY),
    (-0.51, Mood.ANGRY),
])
def test_mood_bands(score, mood):
    assert Mood.from_score(score) is mood


def test_own_word_list():
    scorer = LexiconSentimentScorer({"brilliant": 3})
    assert scorer.score("brilliant").mood is Mood.CALM
    assert scorer.score("rude").score == 0.0

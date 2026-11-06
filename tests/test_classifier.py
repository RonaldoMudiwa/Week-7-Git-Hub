"""Tests for the keyword classifier and for marking classifiers fairly."""

from __future__ import annotations

import pytest

from complaints import (
    Classifier,
    CsvComplaintSource,
    KeywordClassifier,
    NotTrainedError,
    Prediction,
    evaluate,
    train_test_split,
)

from .conftest import make
from .test_stats import SMALL


@pytest.fixture(scope="module")
def sample():
    from .conftest import SAMPLE

    return CsvComplaintSource(SAMPLE).load()


def test_it_is_a_classifier():
    assert isinstance(KeywordClassifier(), Classifier)


def test_learns_clue_words():
    classifier = KeywordClassifier(keywords_per_label=2).fit(SMALL)
    assert classifier.labels == ["Credit card", "Mortgage"]
    assert classifier.keywords("Mortgage") == ["escrow", "payment"]
    assert classifier.keywords("Pets") == []


def test_predicts_from_clue_words():
    classifier = KeywordClassifier().fit(SMALL)
    assert classifier.predict("my ESCROW payment is wrong").label == "Mortgage"
    guess = classifier.predict("a late fee on my card")
    assert guess.label == "Credit card" and 0.5 < guess.confidence <= 1


def test_no_clue_words_means_the_most_common_product_with_no_confidence():
    classifier = KeywordClassifier().fit(SMALL)
    assert classifier.predict("hello there") == Prediction("Credit card", 0.0)


def test_predict_many():
    classifier = KeywordClassifier().fit(SMALL)
    assert [p.label for p in classifier.predict_many(["escrow", "card"])] == ["Mortgage",
                                                                             "Credit card"]


def test_must_be_trained_first():
    with pytest.raises(NotTrainedError):
        KeywordClassifier().predict("anything")


def test_bad_settings():
    with pytest.raises(ValueError):
        KeywordClassifier(keywords_per_label=0)
    with pytest.raises(ValueError):
        KeywordClassifier().fit([])


# ----------------------------------------------------------------------
# Splitting and marking
# ----------------------------------------------------------------------

def test_split_keeps_every_product_on_both_sides(sample):
    train, test = train_test_split(sample, test_share=0.2)
    assert len(train) + len(test) == len(sample)
    assert {c.product for c in train} == {c.product for c in test}
    assert not {c.complaint_id for c in train} & {c.complaint_id for c in test}


def test_split_is_the_same_every_time(sample):
    first = [c.complaint_id for c in train_test_split(sample, seed=1)[1]]
    again = [c.complaint_id for c in train_test_split(sample, seed=1)[1]]
    other = [c.complaint_id for c in train_test_split(sample, seed=2)[1]]
    assert first == again and first != other


@pytest.mark.parametrize("share", [0, 1, -0.5, 1.5])
def test_split_share_must_be_between_0_and_1(sample, share):
    with pytest.raises(ValueError):
        train_test_split(sample, share)


def test_evaluate_counts_right_and_wrong():
    classifier = KeywordClassifier().fit(SMALL)
    test = [make("escrow went up", "Mortgage", complaint_id="a"),
            make("hello", "Mortgage", complaint_id="b"),          # falls back to Credit card: wrong
            make("late fee on card", "Credit card", complaint_id="c")]
    result = evaluate(classifier, test)
    assert (result.total, result.correct, result.accuracy) == (3, 2, 0.667)
    assert result.per_label == {"Credit card": (1, 1), "Mortgage": (1, 2)}
    assert result.label_accuracy("Mortgage") == 0.5
    assert result.label_accuracy("Pets") == 0.0


def test_evaluate_nothing():
    assert evaluate(KeywordClassifier().fit(SMALL), []).accuracy == 0.0


def test_keyword_classifier_on_the_sample(sample):
    # The sample is made up and tidy, so this is easier than real data.
    # The test is here to catch the score suddenly dropping after a change.
    train, test = train_test_split(sample)
    result = evaluate(KeywordClassifier().fit(train), test)
    assert result.accuracy >= 0.85

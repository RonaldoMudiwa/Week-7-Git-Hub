"""Tests for the machine learning classifier and the confusion matrix."""

from __future__ import annotations

import pytest

from complaints import (
    Classifier,
    ConfusionMatrix,
    CsvComplaintSource,
    KeywordClassifier,
    MLClassifier,
    NotTrainedError,
    compare,
    confusion_matrix,
    train_test_split,
)

from .conftest import SAMPLE, make
from .test_stats import SMALL


@pytest.fixture(scope="module")
def split():
    return train_test_split(CsvComplaintSource(SAMPLE).load())


@pytest.fixture(scope="module")
def trained(split) -> MLClassifier:
    return MLClassifier().fit(split[0])


def test_it_is_a_classifier():
    assert isinstance(MLClassifier(), Classifier)


def test_learns_every_product(trained):
    assert trained.labels == ["Checking or savings account", "Credit card", "Credit reporting",
                              "Debt collection", "Money transfer", "Mortgage"]


@pytest.mark.parametrize("text, product", [
    ("The debt collector keeps calling me at work", "Debt collection"),
    ("My escrow payment went up and the servicer will not explain", "Mortgage"),
    ("A charge on my card statement I never made", "Credit card"),
    ("My wire transfer never arrived", "Money transfer"),
])
def test_predicts_new_text(trained, text, product):
    guess = trained.predict(text)
    assert guess.label == product
    assert 0 < guess.confidence <= 1


def test_predict_many_matches_predict(trained):
    texts = ["escrow payment", "debt collector"]
    assert trained.predict_many(texts) == [trained.predict(t) for t in texts]
    assert trained.predict_many([]) == []


def test_top_words_show_what_it_learned(trained):
    words = [w for w, _ in trained.top_words("Money transfer", 10)]
    assert "transfer" in words
    assert trained.top_words("Pets") == []


def test_two_products_only():
    classifier = MLClassifier().fit(SMALL)
    assert classifier.predict("escrow is wrong").label == "Mortgage"
    assert classifier.top_words("Mortgage", 1)[0][0] == "escrow"
    assert classifier.top_words("Credit card", 1)[0][1] > 0


def test_must_be_trained_first():
    with pytest.raises(NotTrainedError):
        MLClassifier().predict("anything")
    with pytest.raises(NotTrainedError):
        MLClassifier().top_words("Mortgage")
    assert MLClassifier().labels == []


def test_needs_two_products():
    with pytest.raises(ValueError, match="at least two products"):
        MLClassifier().fit([make(complaint_id="1"), make(complaint_id="2")])


def test_ml_beats_or_matches_the_keyword_baseline(split):
    # On the tidy made up sample both do well. What matters is the order.
    scores = compare({"keyword": KeywordClassifier(), "ml": MLClassifier()}, *split)
    assert scores["ml"].accuracy >= scores["keyword"].accuracy
    assert scores["keyword"].accuracy >= 0.85


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------

def matrix() -> ConfusionMatrix:
    # 3 real mortgages: 2 guessed right, 1 called debt collection.
    # 2 real debt: both right. So "debt" was called 3 times, right twice.
    counts = {("Mortgage", "Mortgage"): 2, ("Mortgage", "Debt"): 1, ("Debt", "Debt"): 2}
    return ConfusionMatrix(("Debt", "Mortgage"), counts)


def test_counts_precision_and_recall():
    m = matrix()
    assert m.count("Mortgage", "Debt") == 1 and m.count("Debt", "Mortgage") == 0
    assert m.precision("Debt") == 0.667      # one false alarm
    assert m.recall("Debt") == 1.0           # found every real one
    assert m.precision("Mortgage") == 1.0
    assert m.recall("Mortgage") == 0.667     # missed one
    assert m.precision("Pets") == 0.0 and m.recall("Pets") == 0.0


def test_table():
    lines = matrix().table(width=8).splitlines()
    assert lines[0] == "real / guess      Debt Mortgage"
    assert lines[2] == "Mortgage" + " " * 6 + "       1        2"
    assert len({len(line) for line in lines}) == 1      # every row lines up


def test_confusion_matrix_from_a_classifier():
    classifier = KeywordClassifier().fit(SMALL)
    test = [make("escrow", "Mortgage", complaint_id="a"), make("hello", "Mortgage", complaint_id="b")]
    m = confusion_matrix(classifier, test)
    assert m.labels == ("Credit card", "Mortgage")       # a guessed label is included too
    assert m.count("Mortgage", "Credit card") == 1

"""Classifiers: programs that guess what a complaint is about.

Classifier is the plug. KeywordClassifier is the simple one: it learns a
short list of clue words for each product and counts how many appear. On
Day 4 a machine learning classifier will plug into the same place, and the
two can be compared fairly. Starting with something simple matters: if the
clever version can't beat it, the clever version isn't worth having.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .exceptions import ComplaintError
from .models import Complaint
from .stats import WordStats
from .text import TextCleaner


class NotTrainedError(ComplaintError):
    """predict() was called before fit()."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float      # 0 to 1. How sure the classifier is.


class Classifier(ABC):
    """Learns from labelled complaints, then labels new ones."""

    @abstractmethod
    def fit(self, complaints: Iterable[Complaint]) -> Classifier:
        """Learn from complaints that already have a product. Returns itself,
        so you can write KeywordClassifier().fit(training)."""

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Guess the product for some complaint text."""

    @property
    @abstractmethod
    def labels(self) -> list[str]:
        """Every product it knows about. Empty until it's trained."""

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        return [self.predict(text) for text in texts]


class KeywordClassifier(Classifier):
    """Picks the product whose clue words appear most in the text.

    Like a post room worker with a cheat sheet: letters mentioning
    "escrow" or "refinance" go in the mortgage pile. The cheat sheet is
    learned from the training complaints, using WordStats.distinctive_words.
    """

    def __init__(self, keywords_per_label: int = 15, cleaner: TextCleaner | None = None) -> None:
        if keywords_per_label < 1:
            raise ValueError("keywords_per_label must be 1 or more.")
        self.keywords_per_label = keywords_per_label
        self.cleaner = cleaner or TextCleaner()
        self._keywords: dict[str, dict[str, float]] = {}
        self._fallback = ""

    def fit(self, complaints: Iterable[Complaint]) -> KeywordClassifier:
        complaints = list(complaints)
        if not complaints:
            raise ValueError("Can't learn from no complaints.")
        stats = WordStats(self.cleaner)
        labels = Counter(c.product for c in complaints)
        self._keywords = {
            label: dict(stats.distinctive_words(complaints, label, self.keywords_per_label))
            for label in sorted(labels)
        }
        # With no clue words at all, the safest guess is the most common product.
        self._fallback = labels.most_common(1)[0][0]
        return self

    @property
    def labels(self) -> list[str]:
        return list(self._keywords)

    def keywords(self, label: str) -> list[str]:
        """The clue words learned for one product, best first."""
        return list(self._keywords.get(label, {}))

    def predict(self, text: str) -> Prediction:
        if not self._keywords:
            raise NotTrainedError("Call fit() with some labelled complaints first.")
        words = set(self.cleaner.tokens(text))
        scores = {label: sum(weight for word, weight in clues.items() if word in words)
                  for label, clues in self._keywords.items()}
        total = sum(scores.values())
        if total == 0:
            return Prediction(self._fallback, 0.0)
        best = max(scores, key=lambda label: (scores[label], label == self._fallback, label))
        return Prediction(best, round(scores[best] / total, 3))

"""Checking how good a classifier is, fairly.

You can't mark a student on the exact questions they revised from, or
they'd get 100% just by remembering. So the complaints are split in two:
the classifier learns from the "train" part and is marked on the "test"
part it has never seen.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .classifier import Classifier
from .models import Complaint


def train_test_split(complaints: Sequence[Complaint], test_share: float = 0.2,
                     seed: int = 7) -> tuple[list[Complaint], list[Complaint]]:
    """Shuffle, then keep test_share of each product back for testing.

    Splitting each product separately (called stratifying) means a rare
    product can't end up only in training or only in testing by bad luck.
    The seed makes the shuffle the same every run, so results can be compared.
    """
    if not 0 < test_share < 1:
        raise ValueError("test_share must be between 0 and 1.")
    groups: dict[str, list[Complaint]] = defaultdict(list)
    for complaint in complaints:
        groups[complaint.product].append(complaint)
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(groups):
        group = groups[label][:]
        rng.shuffle(group)
        cut = round(len(group) * test_share)
        test += group[:cut]
        train += group[cut:]
    return train, test


@dataclass(frozen=True)
class Evaluation:
    total: int
    correct: int
    # label -> (right, out of)
    per_label: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """The share of guesses that were right, 0 to 1."""
        return round(self.correct / self.total, 3) if self.total else 0.0

    def label_accuracy(self, label: str) -> float:
        right, out_of = self.per_label.get(label, (0, 0))
        return round(right / out_of, 3) if out_of else 0.0


def evaluate(classifier: Classifier, test: Sequence[Complaint]) -> Evaluation:
    """Mark the classifier on complaints whose real product we know."""
    per_label: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for complaint, guess in zip(test, classifier.predict_many(c.narrative for c in test)):
        right = guess.label == complaint.product
        correct += right
        per_label[complaint.product][0] += right
        per_label[complaint.product][1] += 1
    return Evaluation(len(test), correct,
                      {label: (r, n) for label, (r, n) in sorted(per_label.items())})


@dataclass(frozen=True)
class ConfusionMatrix:
    """A table of real product (rows) against guessed product (columns).

    Everything on the diagonal is right. Anything off it shows exactly
    which products get mixed up with which, which a single accuracy
    number hides.
    """

    labels: tuple[str, ...]
    counts: dict[tuple[str, str], int]       # (real, guessed) -> how many

    def count(self, real: str, guessed: str) -> int:
        return self.counts.get((real, guessed), 0)

    def precision(self, label: str) -> float:
        """Of everything it CALLED this product, the share that really was.
        Low precision means lots of false alarms."""
        called = sum(self.count(real, label) for real in self.labels)
        return round(self.count(label, label) / called, 3) if called else 0.0

    def recall(self, label: str) -> float:
        """Of everything that really WAS this product, the share it found.
        Low recall means it misses lots of them."""
        actual = sum(self.count(label, guessed) for guessed in self.labels)
        return round(self.count(label, label) / actual, 3) if actual else 0.0

    def table(self, width: int = 10) -> str:
        """The matrix as text, with names cut short so it fits a terminal."""
        short = [label[:width] for label in self.labels]
        first = max(width, 12) + 2      # room for the "real / guess" heading
        lines = ["real / guess".ljust(first) + " ".join(s.rjust(width) for s in short)]
        for label, name in zip(self.labels, short):
            cells = " ".join(str(self.count(label, g)).rjust(width) for g in self.labels)
            lines.append(name.ljust(first) + cells)
        return "\n".join(lines)


def confusion_matrix(classifier: Classifier, test: Sequence[Complaint]) -> ConfusionMatrix:
    guesses = classifier.predict_many(c.narrative for c in test)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for complaint, guess in zip(test, guesses):
        counts[(complaint.product, guess.label)] += 1
    labels = sorted({c.product for c in test} | {g.label for g in guesses})
    return ConfusionMatrix(tuple(labels), dict(counts))


def compare(classifiers: dict[str, Classifier], train: Sequence[Complaint],
            test: Sequence[Complaint]) -> dict[str, Evaluation]:
    """Train each classifier on the same complaints and mark them on the same
    test, so the scores can be compared fairly."""
    return {name: evaluate(classifier.fit(train), test)
            for name, classifier in classifiers.items()}

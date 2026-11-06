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

"""Counting words: which ones come up most, and which ones belong to which
kind of complaint.

The most common words overall show what customers complain about in
general. The "distinctive" words for a product are the ones that come up a
lot in, say, mortgage complaints but hardly anywhere else, like "escrow".
Those are the clues a classifier can use.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable

from .models import Complaint
from .text import TextCleaner, ngrams


class WordStats:
    def __init__(self, cleaner: TextCleaner | None = None) -> None:
        self.cleaner = cleaner or TextCleaner()

    def word_counts(self, complaints: Iterable[Complaint], size: int = 1) -> Counter[str]:
        """How often each word (size 1) or phrase (size 2, 3...) appears."""
        counts: Counter[str] = Counter()
        for complaint in complaints:
            counts.update(ngrams(self.cleaner.tokens(complaint.narrative), size))
        return counts

    def top_words(self, complaints: Iterable[Complaint], n: int = 10) -> list[tuple[str, int]]:
        return _top(self.word_counts(complaints), n)

    def top_phrases(self, complaints: Iterable[Complaint], size: int = 2,
                    n: int = 10) -> list[tuple[str, int]]:
        return _top(self.word_counts(complaints, size), n)

    def distinctive_words(self, complaints: Iterable[Complaint], label: str, n: int = 10,
                          key: Callable[[Complaint], str] = lambda c: c.product,
                          ) -> list[tuple[str, float]]:
        """Words much more common in one group than in all the others.

        The score is how many times more often the word is used inside the
        group than outside it. A word used equally everywhere scores about 1.
        Adding 1 to every count means a word never seen outside the group
        gets a big score instead of dividing by zero.
        """
        inside: Counter[str] = Counter()
        outside: Counter[str] = Counter()
        for complaint in complaints:
            # set() so a word said five times in one complaint still counts once.
            words = set(self.cleaner.tokens(complaint.narrative))
            (inside if key(complaint) == label else outside).update(words)
        inside_total = sum(inside.values()) or 1
        outside_total = sum(outside.values()) or 1
        scores = {word: ((count + 1) / inside_total) / ((outside[word] + 1) / outside_total)
                  for word, count in inside.items()}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(word, round(score, 2)) for word, score in ranked[:n]]


def _top(counts: Counter[str], n: int) -> list[tuple[str, int]]:
    """Most common first. Ties go in A to Z order, so the answer is always the same."""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:n]


def counts_by(complaints: Iterable[Complaint],
              key: Callable[[Complaint], str] = lambda c: c.product) -> list[tuple[str, int]]:
    """How many complaints in each group, biggest first.

        counts_by(complaints)                       # by product
        counts_by(complaints, lambda c: c.issue)    # by issue
    """
    counts = Counter(key(c) for c in complaints)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def top_issues(complaints: Iterable[Complaint], product: str, n: int = 5) -> list[tuple[str, int]]:
    """The most common issues for one product."""
    return counts_by((c for c in complaints if c.product == product), lambda c: c.issue)[:n]

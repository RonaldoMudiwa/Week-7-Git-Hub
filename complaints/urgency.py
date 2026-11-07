"""How urgent is a complaint?

Each rule looks for one kind of emergency, like fraud or being locked out
of an account, and adds points if it finds it. UrgencyAssessor adds the
points to a mood score and turns the total into a priority from 1 (can
wait) to 5 (deal with it today). Like a hospital A&E: a broken arm is seen
before a sore throat, whatever order they arrived in.

A new kind of emergency is a new rule class added to the list. Nothing
else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

from .sentiment import LexiconSentimentScorer, Mood, Sentiment, SentimentScorer
from .text import TextCleaner, ngrams

MIN_PRIORITY, MAX_PRIORITY = 1, 5


@dataclass(frozen=True)
class Finding:
    rule: str
    points: int
    reason: str


class UrgencyRule(ABC):
    """Looks at the cleaned words of a complaint and maybe finds something."""

    name = ""

    @abstractmethod
    def check(self, words: Sequence[str]) -> Finding | None:
        """A finding, or None if this rule has nothing to say."""


class PhraseRule(UrgencyRule):
    """Finds any of a list of words or phrases of up to three words. Most
    rules are just this."""

    def __init__(self, name: str, phrases: Iterable[str], points: int, reason: str) -> None:
        self.name = name
        self.phrases = frozenset(phrases)
        self.points = points
        self.reason = reason

    def check(self, words: Sequence[str]) -> Finding | None:
        words = list(words)
        seen = set(words) | set(ngrams(words, 2)) | set(ngrams(words, 3))
        found = sorted(self.phrases & seen)
        if not found:
            return None
        return Finding(self.name, self.points, f"{self.reason} ({', '.join(found)})")


class FraudRule(PhraseRule):
    def __init__(self) -> None:
        super().__init__("fraud", {"fraud", "scam", "scammer", "hacked", "stolen", "tricked",
                                   "unauthorized", "identity theft"},
                         3, "Possible fraud: money may still be at risk")


class LockedOutRule(PhraseRule):
    def __init__(self) -> None:
        super().__init__("locked_out", {"frozen", "locked out", "cannot get", "cant access",
                                        "closed my", "cannot access"},
                         2, "Customer can't get to their money")


class HarassmentRule(PhraseRule):
    def __init__(self) -> None:
        super().__init__("harassment", {"threatens", "threatened", "harass", "harassed",
                                        "late at night", "contacted my employer"},
                         2, "Customer is being pressured or threatened")


class RepeatContactRule(PhraseRule):
    def __init__(self) -> None:
        super().__init__("repeat_contact", {"many times", "several times", "times a day",
                                            "nobody helps", "nobody will listen", "no update"},
                         1, "Customer has already tried to get help")


def default_rules() -> list[UrgencyRule]:
    return [FraudRule(), LockedOutRule(), HarassmentRule(), RepeatContactRule()]


@dataclass(frozen=True)
class Urgency:
    priority: int                       # 1 can wait, 5 deal with it today
    sentiment: Sentiment
    findings: tuple[Finding, ...]

    @property
    def reasons(self) -> list[str]:
        reasons = [f.reason for f in self.findings]
        if self.sentiment.mood is Mood.ANGRY:
            reasons.append("Customer is angry")
        return reasons


class UrgencyAssessor:
    def __init__(self, rules: Iterable[UrgencyRule] | None = None,
                 scorer: SentimentScorer | None = None) -> None:
        self.rules = tuple(rules) if rules is not None else tuple(default_rules())
        self.scorer = scorer or LexiconSentimentScorer()
        # Stop words are kept, so phrases like "locked out" and "times a day" match.
        self.cleaner = TextCleaner(remove_stop_words=False, min_length=1)

    def assess(self, text: str) -> Urgency:
        words = self.cleaner.tokens(text)
        findings = tuple(f for f in (rule.check(words) for rule in self.rules) if f)
        sentiment = self.scorer.score(text)
        mood_points = {Mood.CALM: 0, Mood.UNHAPPY: 0, Mood.ANGRY: 1}[sentiment.mood]
        points = sum(f.points for f in findings) + mood_points
        priority = max(MIN_PRIORITY, min(MAX_PRIORITY, MIN_PRIORITY + points))
        return Urgency(priority, sentiment, findings)

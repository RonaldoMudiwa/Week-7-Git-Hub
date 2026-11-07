"""How upset is the customer?

LexiconSentimentScorer uses a word list (a lexicon) where each word has a
score: "unacceptable" is very negative, "thanks" is a bit positive. It
adds up the scores of the words it finds. Like a teacher marking tone of
voice: a few angry words and the whole letter reads as angry.

Two small rules make it smarter than just counting:
- "not" flips the next word, so "not happy" counts as negative.
- "very" and similar words make the next word count for more.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from .text import TextCleaner

# -3 is furious, +3 is delighted. Only words that matter for complaints.
LEXICON: Mapping[str, float] = MappingProxyType({
    # very negative
    "unacceptable": -3, "scam": -3, "fraud": -3, "furious": -3, "disgusting": -3,
    "threatens": -3, "threatened": -3, "stolen": -3, "harass": -3, "harassed": -3,
    "nightmare": -3, "illegal": -3,
    # negative
    "rude": -2, "stressful": -2, "wrong": -2, "refused": -2, "frozen": -2, "hacked": -2,
    "tricked": -2, "lost": -2, "angry": -2, "upset": -2, "unfair": -2, "worst": -2,
    "terrible": -2, "ignored": -2, "lied": -2, "never": -1.5, "nobody": -1.5,
    # a bit negative
    "late": -1, "delayed": -1, "problem": -1, "error": -1, "charged": -1, "fee": -1,
    "fees": -1, "disputed": -1, "mistake": -1, "locked": -1, "stop": -1, "help": -0.5,
    "waiting": -1, "twice": -1, "negative": -1,
    # positive
    "thanks": 1, "thank": 1, "please": 0.5, "resolved": 2, "helpful": 2, "happy": 2,
    "good": 1.5, "great": 2, "fixed": 1, "appreciate": 1.5, "kind": 1.5,
})

NEGATIONS = frozenset({"not", "no", "never", "dont", "didnt", "doesnt", "wont", "cant",
                       "cannot", "isnt", "wasnt", "nobody", "nothing"})
BOOSTERS: Mapping[str, float] = MappingProxyType({
    "very": 1.5, "really": 1.5, "extremely": 2, "so": 1.3, "totally": 1.5, "completely": 1.5})


class Mood(str, Enum):
    CALM = "calm"            # score above -0.1
    UNHAPPY = "unhappy"      # -0.1 to -0.5
    ANGRY = "angry"          # below -0.5

    @classmethod
    def from_score(cls, score: float) -> Mood:
        if score < -0.5:
            return cls.ANGRY
        if score < -0.1:
            return cls.UNHAPPY
        return cls.CALM


@dataclass(frozen=True)
class Sentiment:
    score: float               # -1 (furious) to +1 (happy)
    words: tuple[str, ...]     # the words that changed the score, to explain it

    @property
    def mood(self) -> Mood:
        return Mood.from_score(self.score)


class SentimentScorer(ABC):
    @abstractmethod
    def score(self, text: str) -> Sentiment:
        """How positive or negative the text is."""


class LexiconSentimentScorer(SentimentScorer):
    def __init__(self, lexicon: Mapping[str, float] = LEXICON) -> None:
        self.lexicon = lexicon
        # Stop words must stay in here, because "not" is one and it matters.
        self.cleaner = TextCleaner(remove_stop_words=False, min_length=1)

    def score(self, text: str) -> Sentiment:
        words = self.cleaner.tokens(text)
        total, hits = 0.0, []
        for i, word in enumerate(words):
            value = self.lexicon.get(word)
            if value is None:
                continue
            before = words[max(i - 3, 0):i]     # look back up to three words, for "not at all"
            if before and before[-1] in BOOSTERS:
                value *= BOOSTERS[before[-1]]
            if any(w in NEGATIONS for w in before):
                value *= -0.5   # "not happy" is negative, but softer than "unhappy"
            total += value
            hits.append(word)
        # Squash the total into -1..1. Dividing by the hit count plus one
        # means one strong word counts, but a long rant counts for more.
        score = total / (3 * (len(hits) + 1)) if hits else 0.0
        return Sentiment(round(max(-1.0, min(1.0, score)), 3), tuple(hits))

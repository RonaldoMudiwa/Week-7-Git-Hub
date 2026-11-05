"""Cleaning text and splitting it into words.

A computer can't read. Before it can count or compare words, the text has
to be made tidy: all lower case, no punctuation, no hidden-out details, and
none of the little words like "the" and "and" that appear in every
complaint and tell us nothing. Like sorting post: you look at the address,
not the stamp.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Words so common they say nothing about what a complaint is about.
STOP_WORDS = frozenset("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its
itself just me more most my myself no nor not now of off on once only or other our ours
out over own same she should so some such than that the their theirs them then there
these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours also get got one two us
""".split())

# The CFPB hides private details: names and numbers become XXXX, dates XX/XX/2026.
REDACTED = re.compile(r"\b[xX]{2,}(?:/[xX]{2,})*(?:/\d{2,4})?\b")
# Amounts are written like {$120.00}. The amount itself isn't useful, but
# knowing money was mentioned is, so it becomes the word "money".
MONEY = re.compile(r"\{\$[\d,]*\.?\d*\}")
NOT_A_LETTER = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")


@dataclass(frozen=True)
class TextCleaner:
    """Turns a narrative into a list of useful words (called tokens).

        TextCleaner().tokens("I was charged a LATE fee of {$35.00}!")
        -> ['charged', 'late', 'fee', 'money']
    """

    remove_stop_words: bool = True
    min_length: int = 2
    stop_words: frozenset[str] = field(default=STOP_WORDS)

    def clean(self, text: str) -> str:
        """Lower case, hidden details removed, letters and single spaces only."""
        text = MONEY.sub(" money ", text or "")
        text = REDACTED.sub(" ", text)
        text = text.lower().replace("'", "")    # "don't" -> "dont", not "don t"
        text = NOT_A_LETTER.sub(" ", text)
        return SPACES.sub(" ", text).strip()

    def tokens(self, text: str) -> list[str]:
        words = self.clean(text).split()
        return [w for w in words
                if len(w) >= self.min_length
                and not (self.remove_stop_words and w in self.stop_words)]


def ngrams(tokens: list[str], size: int) -> list[str]:
    """Groups of words that sit next to each other.

        ngrams(['late', 'fee', 'charged'], 2) -> ['late fee', 'fee charged']

    "late fee" means much more than "late" and "fee" on their own, so pairs
    and triples of words are often better clues than single words.
    """
    if size < 1:
        raise ValueError("size must be 1 or more.")
    return [" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

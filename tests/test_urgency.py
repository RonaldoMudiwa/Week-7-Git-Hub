"""Tests for urgency rules and priorities."""

from __future__ import annotations

import pytest

from complaints import (
    CsvComplaintSource,
    FraudRule,
    HarassmentRule,
    LexiconSentimentScorer,
    LockedOutRule,
    PhraseRule,
    RepeatContactRule,
    UrgencyAssessor,
    UrgencyRule,
    default_rules,
)
from complaints.text import TextCleaner

WORDS = TextCleaner(remove_stop_words=False, min_length=1)


def words(text: str) -> list[str]:
    return WORDS.tokens(text)


@pytest.fixture
def assessor() -> UrgencyAssessor:
    return UrgencyAssessor()


@pytest.mark.parametrize("rule, text, found", [
    (FraudRule(), "someone hacked my account", "hacked"),
    (FraudRule(), "this was identity theft", "identity theft"),
    (LockedOutRule(), "I have been locked out of online banking", "locked out"),
    (LockedOutRule(), "I cannot get to my money", "cannot get"),
    (HarassmentRule(), "they call late at night", "late at night"),
    (RepeatContactRule(), "I called several times", "several times"),
    (RepeatContactRule(), "they call three times a day", "times a day"),
])
def test_rules_find_their_words_and_phrases(rule, text, found):
    finding = rule.check(words(text))
    assert finding is not None and found in finding.reason
    assert finding.rule == rule.name and finding.points > 0


@pytest.mark.parametrize("rule", default_rules(), ids=lambda r: r.name)
def test_rules_stay_quiet_on_ordinary_text(rule):
    assert rule.check(words("the interest rate on my card went up")) is None


def test_every_default_rule_is_a_rule():
    assert all(isinstance(rule, UrgencyRule) for rule in default_rules())
    assert len({rule.name for rule in default_rules()}) == 4


def test_ordinary_complaint_is_priority_1(assessor):
    urgency = assessor.assess("The interest rate on my credit card went up.")
    assert (urgency.priority, urgency.findings, urgency.reasons) == (1, (), [])


def test_fraud_and_anger_is_priority_5(assessor):
    urgency = assessor.assess("Someone hacked my account and stole {$500.00}. "
                              "This is unacceptable and a scam.")
    assert urgency.priority == 5
    assert "Customer is angry" in urgency.reasons
    assert urgency.reasons[0].startswith("Possible fraud")


def test_points_add_up(assessor):
    urgency = assessor.assess("My account was frozen. I have called many times.")
    assert [f.rule for f in urgency.findings] == ["locked_out", "repeat_contact"]
    assert urgency.priority == 1 + 2 + 1


def test_priority_never_goes_over_5(assessor):
    text = ("Fraud! I was hacked, locked out, threatened late at night and "
            "called many times. Unacceptable, terrible, a scam.")
    assert assessor.assess(text).priority == 5


def test_own_rules_and_scorer():
    rule = PhraseRule("pets", {"dog"}, 2, "Mentions a dog")
    assessor = UrgencyAssessor([rule], LexiconSentimentScorer({}))
    assert assessor.assess("my dog ate the card").priority == 3
    assert UrgencyAssessor([]).assess("hacked").priority == 1


def test_sample_priorities_are_spread_out(assessor):
    from .conftest import SAMPLE

    priorities = [assessor.assess(c.narrative).priority for c in CsvComplaintSource(SAMPLE).load()]
    assert set(priorities) == {1, 2, 3, 4, 5}
    assert priorities.count(1) > priorities.count(5)     # most complaints aren't emergencies

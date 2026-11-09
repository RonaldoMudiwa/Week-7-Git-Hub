"""Putting it all together: what is a complaint about, and how urgent is it?

ComplaintAnalyzer is handed a trained classifier and an urgency assessor,
and runs both over each complaint. It doesn't care which classifier it
gets, keyword or machine learning, as long as it's a Classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from .classifier import Classifier, Prediction
from .models import Complaint
from .sentiment import Mood
from .urgency import Urgency, UrgencyAssessor


@dataclass(frozen=True)
class AnalysisResult:
    complaint: Complaint
    prediction: Prediction
    urgency: Urgency

    @property
    def priority(self) -> int:
        return self.urgency.priority

    @property
    def mood(self) -> Mood:
        return self.urgency.sentiment.mood

    @property
    def guessed_right(self) -> bool | None:
        """True or False when the real product is known. None for typed in text."""
        if self.complaint.product == UNKNOWN:
            return None
        return self.prediction.label == self.complaint.product


UNKNOWN = "Unknown"


class ComplaintAnalyzer:
    def __init__(self, classifier: Classifier, assessor: UrgencyAssessor | None = None) -> None:
        self.classifier = classifier
        self.assessor = assessor or UrgencyAssessor()

    def analyse(self, complaint: Complaint) -> AnalysisResult:
        return self.analyse_all([complaint])[0]

    def analyse_text(self, text: str) -> AnalysisResult:
        """For a complaint typed in by hand, with no product label."""
        complaint = Complaint("typed", date.today(), UNKNOWN, "", text)
        return self.analyse(complaint)

    def analyse_all(self, complaints: Iterable[Complaint]) -> list[AnalysisResult]:
        """Every complaint, most urgent first. Same priority? Oldest first,
        because it has been waiting longest."""
        complaints = list(complaints)
        guesses = self.classifier.predict_many(c.narrative for c in complaints)
        results = [AnalysisResult(c, guess, self.assessor.assess(c.narrative))
                   for c, guess in zip(complaints, guesses)]
        return sorted(results, key=lambda r: (-r.priority, r.complaint.received,
                                              r.complaint.complaint_id))

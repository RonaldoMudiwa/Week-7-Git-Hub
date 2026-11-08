"""A machine learning classifier, built with scikit-learn.

Two steps, joined into a pipeline:

1. TF-IDF turns each complaint into a list of numbers, one per word. A word
   gets a high number if it's used a lot in this complaint but is rare
   across all complaints. "the" scores almost nothing, "escrow" scores high.
   (TF = term frequency, how often here. IDF = inverse document frequency,
   how rare everywhere.)

2. Logistic regression learns a weight for every word and product: how
   much seeing that word should push the guess towards that product. It's
   like the keyword classifier's cheat sheet, but it learns every weight
   itself, including negative ones ("this word means it's NOT a mortgage").
"""

from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .classifier import Classifier, NotTrainedError, Prediction
from .models import Complaint
from .text import TextCleaner


class MLClassifier(Classifier):
    def __init__(self, cleaner: TextCleaner | None = None, max_words: int = 20_000,
                 strength: float = 5.0) -> None:
        """strength is how closely it fits the training data. Too high and it
        learns the training complaints off by heart and does worse on new
        ones (called overfitting), like a student who memorised the answers
        instead of learning the topic."""
        self.cleaner = cleaner or TextCleaner()
        self.max_words = max_words
        self.strength = strength
        self._pipeline: Pipeline | None = None

    def _new_pipeline(self) -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(
                tokenizer=self.cleaner.tokens,     # our own cleaning, same as everywhere else
                lowercase=False, token_pattern=None,
                ngram_range=(1, 2),               # single words and pairs like "late fee"
                max_features=self.max_words,
                sublinear_tf=True,                # saying a word 10 times isn't 10 times stronger
            )),
            ("model", LogisticRegression(
                C=self.strength, max_iter=1000,
                class_weight="balanced",          # rare products aren't drowned out by common ones
            )),
        ])

    def fit(self, complaints: Iterable[Complaint]) -> MLClassifier:
        complaints = list(complaints)
        labels = [c.product for c in complaints]
        if len(set(labels)) < 2:
            raise ValueError("Needs complaints from at least two products to learn the difference.")
        pipeline = self._new_pipeline()
        pipeline.fit([c.narrative for c in complaints], labels)
        self._pipeline = pipeline
        return self

    @property
    def labels(self) -> list[str]:
        return list(self._pipeline.classes_) if self._pipeline is not None else []

    def predict(self, text: str) -> Prediction:
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[str]) -> list[Prediction]:
        """Faster than one at a time, because the numbers are worked out together."""
        if self._pipeline is None:
            raise NotTrainedError("Call fit() with some labelled complaints first.")
        texts = list(texts)
        if not texts:
            return []
        chances = self._pipeline.predict_proba(texts)
        labels = self._pipeline.classes_
        return [Prediction(str(labels[row.argmax()]), round(float(row.max()), 3))
                for row in chances]

    def top_words(self, label: str, n: int = 10) -> list[tuple[str, float]]:
        """The words that push hardest towards a product. Lets you see what it
        learned, instead of trusting a black box."""
        if self._pipeline is None:
            raise NotTrainedError("Call fit() with some labelled complaints first.")
        if label not in self.labels:
            return []
        vocabulary = self._pipeline.named_steps["tfidf"].get_feature_names_out()
        model = self._pipeline.named_steps["model"]
        index = list(model.classes_).index(label)
        if model.coef_.shape[0] == 1:
            # With only two products sklearn keeps one row of weights for the
            # second product. The first product's weights are the opposite.
            weights = model.coef_[0] if index == 1 else -model.coef_[0]
        else:
            weights = model.coef_[index]
        best = weights.argsort()[::-1][:n]
        return [(str(vocabulary[i]), round(float(weights[i]), 3)) for i in best]

"""Tests for the analyzer, saving models and reports."""

from __future__ import annotations

import csv
import pickle
from datetime import date

import pytest

from complaints import (
    UNKNOWN,
    ComplaintAnalyzer,
    CsvComplaintSource,
    CsvExport,
    DataError,
    HtmlReport,
    KeywordClassifier,
    MLClassifier,
    Mood,
    ReportBuilder,
    Summary,
    TextReport,
    load_model,
    save_model,
    train_test_split,
)
from complaints.reports import bar, percent

from .conftest import SAMPLE, make
from .test_stats import SMALL


@pytest.fixture(scope="module")
def complaints():
    return CsvComplaintSource(SAMPLE).load()


@pytest.fixture(scope="module")
def results(complaints):
    train, _ = train_test_split(complaints)
    return ComplaintAnalyzer(KeywordClassifier().fit(train)).analyse_all(complaints)


# ----------------------------------------------------------------------
# Analyzer
# ----------------------------------------------------------------------

def test_most_urgent_first_then_oldest(results):
    keys = [(-r.priority, r.complaint.received) for r in results]
    assert keys == sorted(keys)
    assert results[0].priority == 5


def test_result_details(results):
    first = results[0]
    assert first.mood in set(Mood)
    assert first.guessed_right in (True, False)
    assert first.urgency.reasons


def test_typed_text_has_no_real_product():
    analyzer = ComplaintAnalyzer(KeywordClassifier().fit(SMALL))
    result = analyzer.analyse_text("My escrow payment was stolen by a scammer")
    assert result.complaint.product == UNKNOWN
    assert result.guessed_right is None
    assert result.prediction.label == "Mortgage"
    assert result.priority >= 4


def test_analyse_one_complaint():
    analyzer = ComplaintAnalyzer(KeywordClassifier().fit(SMALL))
    result = analyzer.analyse(make("late fee on my card"))
    assert result.guessed_right is True


def test_works_with_either_classifier(complaints):
    train, test = train_test_split(complaints)
    for classifier in (KeywordClassifier(), MLClassifier()):
        assert len(ComplaintAnalyzer(classifier.fit(train)).analyse_all(test)) == len(test)


# ----------------------------------------------------------------------
# Saving models
# ----------------------------------------------------------------------

@pytest.mark.parametrize("classifier", [KeywordClassifier(), MLClassifier()],
                         ids=["keyword", "ml"])
def test_save_and_load_give_the_same_answers(tmp_path, complaints, classifier):
    train, test = train_test_split(complaints)
    classifier.fit(train)
    path = save_model(classifier, tmp_path / "models" / "model.pkl")
    loaded = load_model(path)
    texts = [c.narrative for c in test]
    assert type(loaded) is type(classifier)
    assert loaded.predict_many(texts) == classifier.predict_many(texts)
    assert not (tmp_path / "models" / "model.pkl.tmp").exists()


def test_cant_save_an_untrained_model(tmp_path):
    with pytest.raises(ValueError, match="Train"):
        save_model(KeywordClassifier(), tmp_path / "m.pkl")


def test_load_missing(tmp_path):
    with pytest.raises(DataError, match="No model at"):
        load_model(tmp_path / "nope.pkl")


@pytest.mark.parametrize("content, message", [
    (b"not a pickle at all", "isn't a model this program can read"),
    (pickle.dumps(["a", "list"]), "isn't a complaints model file"),
    (pickle.dumps({"format": "complaints-model", "version": 99}), "different version"),
    (pickle.dumps({"format": "complaints-model", "version": 1, "classifier": "text"}),
     "doesn't hold a classifier"),
])
def test_load_refuses_other_files(tmp_path, content, message):
    path = tmp_path / "bad.pkl"
    path.write_bytes(content)
    with pytest.raises(DataError, match=message):
        load_model(path)


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------

def test_summary(results):
    summary = Summary.of(results, urgent=3)
    assert summary.total == 300
    assert sum(c for _, c in summary.by_product) == 300
    assert list(summary.by_priority) == [5, 4, 3, 2, 1]
    assert sum(summary.by_priority.values()) == 300
    assert set(summary.by_mood) == set(Mood)
    assert len(summary.most_urgent) == 3 and summary.most_urgent[0].priority == 5
    assert 0.8 < summary.accuracy <= 1


def test_summary_of_typed_text_has_no_accuracy():
    analyzer = ComplaintAnalyzer(KeywordClassifier().fit(SMALL))
    summary = Summary.of([analyzer.analyse_text("escrow")])
    assert summary.accuracy is None


def test_summary_of_nothing():
    summary = Summary.of([])
    assert (summary.total, summary.by_product, summary.accuracy) == (0, [], None)
    assert "Complaints analysed: 0" in TextReport().render(summary)


def test_text_report(results):
    text = TextReport().render(Summary.of(results, urgent=2))
    assert "Complaints analysed: 300" in text
    assert "Product guessed right:" in text
    assert "By priority (5 = deal with it today)" in text
    assert text.count("\n  [5] ") == 2
    assert "        - Possible fraud" in text or "        - Customer" in text


def test_html_report(tmp_path, results):
    path = HtmlReport().save(Summary.of(results), tmp_path / "out" / "report.html")
    html = path.read_text(encoding="utf-8")
    assert isinstance(HtmlReport(), ReportBuilder)
    assert html.startswith("<!DOCTYPE html>")
    assert html.count("data:image/png;base64,") == 2
    assert "<script" not in html
    assert html.count("<tr>") == 11      # heading plus the 10 most urgent


def test_html_report_escapes_text():
    analyzer = ComplaintAnalyzer(KeywordClassifier().fit(SMALL))
    result = analyzer.analyse(make("<b>escrow</b> & stuff", "Mortgage"))
    html = HtmlReport().render(Summary.of([result]))
    assert "<b>escrow" not in html and "&lt;b&gt;escrow" in html


def test_csv_export(tmp_path, results):
    path = CsvExport().save(results, tmp_path / "results.csv")
    with path.open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 300
    assert list(rows[0]) == CsvExport.COLUMNS
    assert rows[0]["priority"] == "5"
    assert rows[0]["received"] < "2027"


@pytest.mark.parametrize("count, biggest, width, expected", [
    (10, 10, 4, "####"), (5, 10, 4, "##"), (1, 100, 4, "#"), (0, 10, 4, ""),
])
def test_bar(count, biggest, width, expected):
    assert bar(count, biggest, width) == expected


def test_percent():
    assert percent(1, 4) == "25%" and percent(1, 0) == "0%"


def test_today_is_used_for_typed_text():
    analyzer = ComplaintAnalyzer(KeywordClassifier().fit(SMALL))
    assert analyzer.analyse_text("escrow").complaint.received == date.today()

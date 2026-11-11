"""Whole program tests: every part working together, the way a person
would use it, from a data file all the way to a finished report."""

from __future__ import annotations

import csv
import io

import pytest

from complaints import (
    CfpbDownloader,
    ComplaintAnalyzer,
    CsvComplaintSource,
    HtmlReport,
    KeywordClassifier,
    MLClassifier,
    Summary,
    cli,
    compare,
    load_model,
    save_model,
    train_test_split,
)

from .conftest import SAMPLE
from .test_download import ITEM, FakeFetch


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return cli.main(list(argv), out=out), out.getvalue()


def test_a_week_of_work_from_the_command_line(tmp_path):
    model, report, results = tmp_path / "m.pkl", tmp_path / "r.html", tmp_path / "r.csv"
    base = ["--data", str(SAMPLE), "--model-file", str(model)]

    assert run(*base, "train")[0] == 0
    code, text = run(*base, "analyse", "--top", "5", "--html", str(report), "--csv", str(results))
    assert code == 0

    # The most urgent complaint in the CSV is the same one the report lists first.
    with results.open(encoding="utf-8-sig") as file:
        first = next(csv.DictReader(file))
    assert first["priority"] == "5"
    assert f"[5] {first['complaint_id']}" in text
    assert first["complaint_id"] in report.read_text(encoding="utf-8")


def test_downloaded_data_goes_through_the_whole_program(tmp_path, monkeypatch):
    # Pretend the CFPB sent back a few complaints, then use them like real data.
    items = [dict(ITEM, complaint_id=n, product=p, complaint_what_happened=t) for n, (p, t) in
             enumerate([("Mortgage", "My escrow payment went up"),
                        ("Debt collection", "A debt collector threatens me late at night"),
                        ("Mortgage", "The lender changed my closing costs"),
                        ("Debt collection", "They call about a debt that is not mine")])]
    monkeypatch.setattr(cli, "CfpbDownloader", lambda: CfpbDownloader(FakeFetch(items)))
    data, model = tmp_path / "cfpb.csv", tmp_path / "m.pkl"

    assert run("download", "--count", "4", "--out", str(data))[0] == 0
    assert len(CsvComplaintSource(data).load()) == 4
    code, text = run("--data", str(data), "--model-file", str(model), "analyse")
    assert code == 1 and "No model" in text            # no model yet: a clear message
    classifier = MLClassifier().fit(CsvComplaintSource(data).load())
    save_model(classifier, model)
    code, text = run("--data", str(data), "--model-file", str(model), "analyse")
    assert code == 0 and "Complaints analysed: 4" in text


def test_a_saved_model_gives_the_same_report(tmp_path):
    complaints = CsvComplaintSource(SAMPLE).load()
    classifier = MLClassifier().fit(complaints)
    reloaded = load_model(save_model(classifier, tmp_path / "m.pkl"))

    before = Summary.of(ComplaintAnalyzer(classifier).analyse_all(complaints))
    after = Summary.of(ComplaintAnalyzer(reloaded).analyse_all(complaints))
    assert HtmlReport().render(before) == HtmlReport().render(after)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_machine_learning_beats_the_baseline_on_any_split(seed):
    # Not just one lucky shuffle: the result holds on several different splits.
    train, test = train_test_split(CsvComplaintSource(SAMPLE).load(), seed=seed)
    scores = compare({"keyword": KeywordClassifier(), "ml": MLClassifier()}, train, test)
    assert scores["ml"].accuracy >= scores["keyword"].accuracy

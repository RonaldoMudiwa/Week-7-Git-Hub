"""Tests for the command line. None of them go online."""

from __future__ import annotations

import io
import runpy
import sys
from pathlib import Path

import pytest

from complaints import CfpbDownloader, cli

from .conftest import SAMPLE
from .test_download import ITEM, FakeFetch


@pytest.fixture
def run(tmp_path):
    model = tmp_path / "model.pkl"

    def run(*argv: str) -> tuple[int, str]:
        out = io.StringIO()
        code = cli.main(["--data", str(SAMPLE), "--model-file", str(model), *argv], out=out)
        return code, out.getvalue()

    return run


def test_no_command_shows_help(run):
    code, text = run()
    assert code == 0 and "classify" in text and "download" in text


def test_every_command_has_a_name_and_help():
    names = [c.name for c in cli.COMMANDS]
    assert len(names) == len(set(names)) and all(c.help for c in cli.COMMANDS)


def test_demo(run):
    code, text = run("demo")
    assert code == 0
    assert text.count("Product:") == len(cli.DEMO_TEXTS)
    assert "Priority: 5 of 5" in text and "Possible fraud" in text


@pytest.mark.parametrize("kind, name", [("ml", "machine learning"), ("keyword", "keyword")])
def test_train_then_classify(run, kind, name):
    code, text = run("train", "--kind", kind)
    assert code == 0
    assert f"Trained a {name} classifier on 241 complaints." in text
    assert "Retrained on all 300 and saved to" in text

    code, text = run("classify", "The", "debt", "collector", "keeps", "calling")
    assert code == 0 and "Product:  Debt collection" in text


def test_classify_needs_a_model(run):
    code, text = run("classify", "hello")
    assert code == 1 and "No model at" in text and "Train one first" in text


def test_evaluate(run):
    code, text = run("evaluate")
    assert code == 0
    assert "keyword" in text and "machine learning" in text
    assert "real / guess" in text and "precision" in text and "recall" in text


def test_analyse_with_reports(run, tmp_path):
    run("train")
    code, text = run("analyse", "--top", "3", "--html", str(tmp_path / "r.html"),
                     "--csv", str(tmp_path / "r.csv"))
    assert code == 0
    assert "Complaints analysed: 300" in text
    assert text.count("\n  [") == 3
    assert "includes the complaints it was trained on" in text
    assert (tmp_path / "r.html").exists() and (tmp_path / "r.csv").exists()


def test_analyse_with_a_limit(run):
    run("train")
    code, text = run("--limit", "20", "analyse", "--top", "0")
    assert "Complaints analysed: 20" in text and "\n  [" not in text


def test_words(run):
    code, text = run("words", "--top", "3")
    assert code == 0 and "Complaints by product:" in text and "Most common words:" in text


def test_words_for_a_product(run):
    code, text = run("words", "--product", "  money TRANSFER ")
    assert code == 0 and "Clue words for Money transfer" in text and "transfer" in text


def test_words_for_an_unknown_product(run):
    code, text = run("words", "--product", "Pets")
    assert code == 1 and "No product called 'Pets'" in text and "Mortgage" in text


def test_download(run, tmp_path, monkeypatch):
    fake = FakeFetch([ITEM])
    monkeypatch.setattr(cli, "CfpbDownloader", lambda: CfpbDownloader(fake))
    out = tmp_path / "cfpb.csv"
    code, text = run("download", "--count", "5", "--out", str(out))
    assert code == 0 and "Saved 1 complaints" in text and out.exists()
    assert "size=5" in fake.urls[0]


def test_download_bad_count(run):
    code, text = run("download", "--count", "0")
    assert code == 1 and "count must be between" in text


def test_missing_data_file(tmp_path):
    out = io.StringIO()
    assert cli.main(["--data", str(tmp_path / "nope.csv"), "words"], out=out) == 1
    assert "No file at" in out.getvalue()


def test_analyse_new_complaints_with_no_known_product(run, tmp_path):
    # Fresh complaints that nobody has labelled yet, so there's no accuracy to show.
    path = tmp_path / "new.csv"
    path.write_text("Date received,Product,Issue,Consumer complaint narrative,Company,State,"
                    "Complaint ID\n2026-11-10,Unknown,,My card was hacked,Bank,TX,1\n",
                    encoding="utf-8")
    run("train")
    code, text = run("--data", str(path), "analyse")
    assert code == 0
    assert "Product guessed right" not in text and "includes the complaints" not in text


def test_file_with_no_usable_rows(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Date received,Product,Issue,Consumer complaint narrative,Company,State,"
                    "Complaint ID\n2026-11-01,Mortgage,Late,,Bank,TX,1\n", encoding="utf-8")
    out = io.StringIO()
    assert cli.main(["--data", str(path), "words"], out=out) == 1
    assert "Skipped 1 rows" in out.getvalue() and "No complaints with text" in out.getvalue()


def test_runs_as_a_module(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["complaints", "--data", str(SAMPLE), "words", "--top", "1"])
    with pytest.raises(SystemExit) as finished:
        runpy.run_module("complaints", run_name="__main__")
    assert finished.value.code == 0 and "Complaints by product" in capsys.readouterr().out


def test_main_writes_to_stdout_by_default(capsys):
    assert cli.main(["--data", str(SAMPLE), "words", "--top", "1"]) == 0
    assert "Most common words" in capsys.readouterr().out


def test_defaults_point_at_the_data_folder():
    args = cli.build_parser().parse_args(["words"])
    assert args.data == Path("data/sample_complaints.csv")
    assert args.model_file == Path("data/model.pkl")

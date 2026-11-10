"""The command line interface.

    python -m complaints demo                     try it straight away
    python -m complaints train                    learn from the data, save the model
    python -m complaints evaluate                 keyword vs machine learning
    python -m complaints classify "My card was charged twice"
    python -m complaints analyse --html data/report.html
    python -m complaints words --product Mortgage
    python -m complaints download --count 2000    real complaints from the CFPB

Every command reads data/sample_complaints.csv unless --data says otherwise.
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, TextIO

from .analyzer import ComplaintAnalyzer
from .classifier import Classifier, KeywordClassifier
from .download import MAX_COUNT, CfpbDownloader
from .evaluation import compare, confusion_matrix, evaluate, train_test_split
from .exceptions import ComplaintError
from .ml import MLClassifier
from .models import Complaint
from .reports import CsvExport, HtmlReport, Summary, TextReport
from .sources import CsvComplaintSource
from .stats import WordStats, counts_by
from .storage import load_model, save_model

DEFAULT_DATA = Path("data/sample_complaints.csv")
DEFAULT_MODEL = Path("data/model.pkl")
KINDS = {"ml": MLClassifier, "keyword": KeywordClassifier}
KIND_NAMES = {"ml": "machine learning", "keyword": "keyword"}

DEMO_TEXTS = [
    "Someone hacked my account and sent a wire transfer of {$900.00} to a stranger. Unacceptable!",
    "The debt collector calls me late at night and threatens to tell my employer.",
    "My escrow payment went up with no explanation.",
    "I was charged an annual fee that nobody told me about.",
]


class Context:
    """What a command gets to work with while it runs."""

    def __init__(self, out: TextIO, args: argparse.Namespace) -> None:
        self.out = out
        self.args = args

    def say(self, text: str = "") -> None:
        print(text, file=self.out)

    def complaints(self) -> list[Complaint]:
        source = CsvComplaintSource(self.args.data)
        complaints = source.load(self.args.limit)
        skipped = source.report.skipped_no_text + source.report.skipped_bad
        if skipped:
            self.say(f"(Skipped {skipped} rows with no text or a bad date.)")
        if not complaints:
            raise ComplaintError(f"No complaints with text found in {self.args.data}.")
        return complaints

    def model(self) -> Classifier:
        return load_model(self.args.model_file)


class Command(ABC):
    """One thing the user can ask for. Each command is its own small class."""

    name = ""
    help = ""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this command's own options. Commands with none can skip it."""

    @abstractmethod
    def run(self, ctx: Context) -> None:
        """Do the job. Raise a ComplaintError if it can't be done."""


class TrainCommand(Command):
    name = "train"
    help = "learn from the data, mark it on unseen complaints, and save it"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--kind", choices=sorted(KINDS), default="ml")

    def run(self, ctx: Context) -> None:
        complaints = ctx.complaints()
        train, test = train_test_split(complaints)
        kind = KINDS[ctx.args.kind]
        result = evaluate(kind().fit(train), test)
        ctx.say(f"Trained a {KIND_NAMES[ctx.args.kind]} classifier on {len(train)} complaints.")
        ctx.say(f"Marked on {len(test)} it had never seen: {result.accuracy:.1%} right.")
        # The test was only for marking. The saved one learns from everything.
        final = kind().fit(complaints)
        path = save_model(final, ctx.args.model_file)
        ctx.say(f"Retrained on all {len(complaints)} and saved to {path}.")


class EvaluateCommand(Command):
    name = "evaluate"
    help = "compare the keyword and machine learning classifiers fairly"

    def run(self, ctx: Context) -> None:
        train, test = train_test_split(ctx.complaints())
        ml = MLClassifier()
        scores = compare({"keyword": KeywordClassifier(), "machine learning": ml}, train, test)
        ctx.say(f"Trained on {len(train)}, tested on {len(test)} unseen complaints.\n")
        for name, result in scores.items():
            ctx.say(f"  {name:<18}{result.accuracy:>7.1%}")
        matrix = confusion_matrix(ml, test)
        ctx.say("\nMachine learning, real product down the side, guess across the top:\n")
        ctx.say(matrix.table())
        ctx.say("")
        for label in matrix.labels:
            ctx.say(f"  {label:<32} precision {matrix.precision(label):.0%}"
                    f"   recall {matrix.recall(label):.0%}")


class ClassifyCommand(Command):
    name = "classify"
    help = "what is one complaint about, and how urgent is it?"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("text", nargs="+", help="the complaint, in quotes or not")

    def run(self, ctx: Context) -> None:
        result = ComplaintAnalyzer(ctx.model()).analyse_text(" ".join(ctx.args.text))
        show_result(ctx, result)


class AnalyseCommand(Command):
    name = "analyse"
    help = "analyse every complaint in the data file and report on them"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--top", type=int, default=10, help="how many urgent ones to list")
        parser.add_argument("--html", type=Path, help="also save a web page report here")
        parser.add_argument("--csv", type=Path, help="also save every result here, for Excel")

    def run(self, ctx: Context) -> None:
        results = ComplaintAnalyzer(ctx.model()).analyse_all(ctx.complaints())
        summary = Summary.of(results, urgent=max(ctx.args.top, 0))
        ctx.say(TextReport().render(summary).rstrip())
        if summary.accuracy is not None:
            ctx.say("\n(Accuracy here includes the complaints it was trained on, so it "
                    "looks better than it is. Use evaluate for a fair score.)")
        if ctx.args.html:
            ctx.say(f"\nSaved {HtmlReport().save(summary, ctx.args.html)}. Open it in a browser.")
        if ctx.args.csv:
            ctx.say(f"Saved {CsvExport().save(results, ctx.args.csv)}.")


class WordsCommand(Command):
    name = "words"
    help = "the most common words and phrases, and each product's clue words"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--product", help="show the clue words for one product")
        parser.add_argument("--top", type=int, default=10)

    def run(self, ctx: Context) -> None:
        complaints = ctx.complaints()
        stats, top = WordStats(), ctx.args.top
        if ctx.args.product:
            products = {name.lower(): name for name, _ in counts_by(complaints)}
            product = products.get(ctx.args.product.strip().lower())
            if product is None:
                raise ComplaintError(f"No product called {ctx.args.product!r}. "
                                     f"Try one of: {', '.join(products.values())}.")
            ctx.say(f"Clue words for {product} (how many times more common than elsewhere):")
            for word, score in stats.distinctive_words(complaints, product, top):
                ctx.say(f"  {word:<20}{score:>8.1f}")
            return
        ctx.say("Complaints by product:")
        for name, count in counts_by(complaints):
            ctx.say(f"  {name:<32}{count:>6}")
        ctx.say("\nMost common words:")
        ctx.say("  " + ", ".join(f"{w} ({n})" for w, n in stats.top_words(complaints, top)))
        ctx.say("\nMost common two word phrases:")
        ctx.say("  " + ", ".join(f"{w} ({n})" for w, n in stats.top_phrases(complaints, 2, top)))


class DownloadCommand(Command):
    name = "download"
    help = "download real complaints from the CFPB (needs the internet)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--count", type=int, default=1000, help=f"1 to {MAX_COUNT}")
        parser.add_argument("--product", help='only one product, e.g. "Mortgage"')
        parser.add_argument("--out", type=Path, default=Path("data/cfpb_complaints.csv"))

    def run(self, ctx: Context) -> None:
        downloader = CfpbDownloader()
        ctx.say("Asking the CFPB for complaints. This can take a minute...")
        try:
            rows = downloader.download(ctx.args.count, ctx.args.product)
        except ValueError as error:
            raise ComplaintError(str(error)) from None
        path = downloader.save(rows, ctx.args.out)
        ctx.say(f"Saved {len(rows)} complaints to {path}.")
        ctx.say(f"Use them with --data {path}, e.g. python -m complaints --data {path} train")


class DemoCommand(Command):
    name = "demo"
    help = "train on the sample in memory and try some example complaints"

    def run(self, ctx: Context) -> None:
        classifier = MLClassifier().fit(ctx.complaints())
        analyzer = ComplaintAnalyzer(classifier)
        for text in DEMO_TEXTS:
            ctx.say(f'\n"{text}"')
            show_result(ctx, analyzer.analyse_text(text))


def show_result(ctx: Context, result) -> None:
    ctx.say(f"  Product:  {result.prediction.label} ({result.prediction.confidence:.0%} sure)")
    ctx.say(f"  Priority: {result.priority} of 5")
    ctx.say(f"  Mood:     {result.mood.value} ({result.urgency.sentiment.score:+.2f})")
    for reason in result.urgency.reasons:
        ctx.say(f"  - {reason}")


COMMANDS: tuple[Command, ...] = (
    DemoCommand(), TrainCommand(), EvaluateCommand(), ClassifyCommand(), AnalyseCommand(),
    WordsCommand(), DownloadCommand(),
)


def build_parser(commands: Sequence[Command] = COMMANDS) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m complaints",
                                     description="Sort bank complaints by product and urgency.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="the complaints CSV file")
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL,
                        help="where the trained model is saved")
    parser.add_argument("--limit", type=int, help="only read this many complaints")
    sub = parser.add_subparsers(dest="command", metavar="command")
    for command in commands:
        child = sub.add_parser(command.name, help=command.help, description=command.help)
        command.add_arguments(child)
        child.set_defaults(handler=command)
    return parser


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run one command. Returns 0 if it worked, 1 if it didn't."""
    out = out or sys.stdout
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(out)
        return 0
    try:
        args.handler.run(Context(out, args))
    except ComplaintError as error:
        print(f"Error: {error}", file=out)
        return 1
    return 0

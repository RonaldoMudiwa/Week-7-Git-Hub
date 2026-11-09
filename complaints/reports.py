"""Reports: the analysis turned into something a manager can read.

Summary works out the numbers once. Each ReportBuilder then shows them in
its own way: TextReport for the terminal, HtmlReport as a web page with
charts. CsvExport saves one row per complaint for Excel.
"""

from __future__ import annotations

import base64
import csv
import io
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Sequence

from .analyzer import AnalysisResult
from .sentiment import Mood


@dataclass(frozen=True)
class Summary:
    total: int
    by_product: list[tuple[str, int]]        # biggest first
    by_priority: dict[int, int]              # 5 down to 1, every level shown
    by_mood: dict[Mood, int]
    most_urgent: list[AnalysisResult]
    accuracy: float | None                   # None when the real products aren't known

    @classmethod
    def of(cls, results: Sequence[AnalysisResult], urgent: int = 10) -> Summary:
        products = Counter(r.prediction.label for r in results)
        priorities = Counter(r.priority for r in results)
        moods = Counter(r.mood for r in results)
        known = [r.guessed_right for r in results if r.guessed_right is not None]
        return cls(
            total=len(results),
            by_product=sorted(products.items(), key=lambda item: (-item[1], item[0])),
            by_priority={level: priorities.get(level, 0) for level in range(5, 0, -1)},
            by_mood={mood: moods.get(mood, 0) for mood in Mood},
            most_urgent=sorted(results, key=lambda r: -r.priority)[:urgent],
            accuracy=round(sum(known) / len(known), 3) if known else None,
        )


def bar(count: int, biggest: int, width: int) -> str:
    """A bar of #s, as long as count is compared with the biggest. Anything
    above zero gets at least one #, so small groups don't vanish."""
    length = round(count / biggest * width)
    return "#" * (max(length, 1) if count else 0)


def percent(part: int, whole: int) -> str:
    return f"{part / whole:.0%}" if whole else "0%"


class ReportBuilder(ABC):
    @abstractmethod
    def render(self, summary: Summary) -> str:
        """The whole report as text."""

    def save(self, summary: Summary, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(summary), encoding="utf-8")
        return path


class TextReport(ReportBuilder):
    def render(self, summary: Summary) -> str:
        lines = [f"Complaints analysed: {summary.total}"]
        if summary.accuracy is not None:
            lines.append(f"Product guessed right: {summary.accuracy:.1%}")
        lines += ["", "By product"]
        lines += [f"  {name:<32}{count:>5}  {percent(count, summary.total):>4}"
                  for name, count in summary.by_product]
        lines += ["", "By priority (5 = deal with it today)"]
        biggest = max(summary.by_priority.values(), default=0) or 1
        lines += [f"  {level}  {bar(count, biggest, 40):<40} {count}"
                  for level, count in summary.by_priority.items()]
        lines += ["", "Mood"]
        lines += [f"  {mood.value:<10}{count:>5}" for mood, count in summary.by_mood.items()]
        lines += ["", "Most urgent"]
        for r in summary.most_urgent:
            lines.append(f"  [{r.priority}] {r.complaint.complaint_id} "
                         f"{r.prediction.label}: {r.complaint.short(60)}")
            lines += [f"        - {reason}" for reason in r.urgency.reasons]
        return "\n".join(lines) + "\n"


class HtmlReport(ReportBuilder):
    """One web page with the charts inside it, so it opens offline and can be emailed."""

    def render(self, summary: Summary) -> str:
        product_chart = bar_chart([n for n, _ in summary.by_product],
                                  [c for _, c in summary.by_product], "Complaints by product")
        priority_chart = bar_chart([f"Priority {p}" for p in summary.by_priority],
                                   list(summary.by_priority.values()), "Complaints by priority")
        accuracy = ("" if summary.accuracy is None else
                    f"<p>Product guessed right: <strong>{summary.accuracy:.1%}</strong></p>")
        rows = "".join(
            f"<tr><td>{r.priority}</td><td>{escape(r.complaint.complaint_id)}</td>"
            f"<td>{escape(r.prediction.label)}</td><td>{escape(r.complaint.short(90))}</td>"
            f"<td>{escape('; '.join(r.urgency.reasons))}</td></tr>"
            for r in summary.most_urgent)
        moods = ", ".join(f"{m.value} {c}" for m, c in summary.by_mood.items())
        return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Complaint report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; max-width: 960px; margin: 24px auto;
       padding: 0 16px; color: #1d2733; }}
img {{ max-width: 100%; }}
table {{ border-collapse: collapse; width: 100%; }}
td, th {{ border-bottom: 1px solid #dde3ea; padding: 6px; text-align: left; vertical-align: top; }}
</style></head><body>
<h1>Complaint report</h1>
<p>Complaints analysed: <strong>{summary.total}</strong>. Mood: {escape(moods)}.</p>
{accuracy}
<img alt="Complaints by product" src="data:image/png;base64,{product_chart}">
<img alt="Complaints by priority" src="data:image/png;base64,{priority_chart}">
<h2>Most urgent</h2>
<table><tr><th>Priority</th><th>Id</th><th>Product</th><th>Complaint</th><th>Why</th></tr>
{rows}</table>
</body></html>
"""


def bar_chart(labels: list[str], values: list[int], title: str) -> str:
    """A sideways bar chart as a PNG picture, written as text (base64) so it
    can sit inside the web page."""
    import matplotlib
    matplotlib.use("Agg")       # draw to a file, never try to open a window
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(8, 0.45 * max(len(labels), 3) + 1))
    axes.barh(labels[::-1], values[::-1], color="#3b7dd8")
    axes.set_title(title)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class CsvExport:
    """One row per complaint, most urgent first, ready to open in Excel."""

    COLUMNS = ["complaint_id", "received", "priority", "mood", "sentiment", "guessed_product",
               "confidence", "real_product", "reasons", "narrative"]

    def save(self, results: Sequence[AnalysisResult], path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # utf-8-sig so Excel shows £ and other symbols properly.
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(self.COLUMNS)
            for r in results:
                writer.writerow([r.complaint.complaint_id, r.complaint.received.isoformat(),
                                 r.priority, r.mood.value, r.urgency.sentiment.score,
                                 r.prediction.label, r.prediction.confidence,
                                 r.complaint.product, "; ".join(r.urgency.reasons),
                                 r.complaint.narrative])
        return path

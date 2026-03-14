"""Report generation: JSON, CSV, and markdown summaries."""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path
from typing import Any

from continual_benchmark.core.io import write_json
from continual_benchmark.core.schemas import MetricReport, QAReport, StageEvalResult


def write_stage_results_csv(results: list[StageEvalResult], path: Path) -> None:
    """Write per-stage evaluation results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "family", "spec_id", "split", "accuracy", "total", "correct"])
        for r in results:
            writer.writerow([
                r.stage, r.family, r.spec_id, r.split,
                f"{r.accuracy:.4f}", r.total, r.correct_count,
            ])


def write_metrics_csv(report: MetricReport, path: Path) -> None:
    """Write metrics to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Average accuracy per stage
        writer.writerow(["Metric", "Value"])
        writer.writerow(["average_forgetting", f"{report.average_forgetting:.4f}"])
        writer.writerow(["average_backward_transfer", f"{report.average_backward_transfer:.4f}"])
        writer.writerow(["average_forward_transfer", f"{report.average_forward_transfer:.4f}"])
        writer.writerow([])

        # Per-stage average accuracy
        writer.writerow(["Training Stage", "Average Accuracy"])
        for i, acc in enumerate(report.average_accuracy):
            writer.writerow([i + 1, f"{acc:.4f}"])
        writer.writerow([])

        # Per-task forgetting
        writer.writerow(["Task Stage", "Forgetting"])
        for stage, fgt in sorted(report.forgetting.items()):
            writer.writerow([stage, f"{fgt:.4f}"])


def generate_markdown_summary(
    report: MetricReport,
    stage_results: list[StageEvalResult] | None = None,
) -> str:
    """Generate a markdown summary of metrics and results."""
    lines = ["# Continual Learning Benchmark Report", ""]

    # Overall metrics
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"- **Average Forgetting:** {report.average_forgetting:.4f}")
    lines.append(f"- **Average Backward Transfer:** {report.average_backward_transfer:.4f}")
    lines.append(f"- **Average Forward Transfer:** {report.average_forward_transfer:.4f}")
    lines.append("")

    # Average accuracy progression
    if report.average_accuracy:
        lines.append("## Average Accuracy by Training Stage")
        lines.append("")
        lines.append("| Training Stage | Average Accuracy |")
        lines.append("|---|---|")
        for i, acc in enumerate(report.average_accuracy):
            lines.append(f"| {i + 1} | {acc:.4f} |")
        lines.append("")

    # Family breakdown
    if report.family_breakdown:
        lines.append("## Family Breakdown")
        lines.append("")
        lines.append("| Family | Avg Accuracy | Avg Forgetting |")
        lines.append("|---|---|---|")
        for family, metrics in sorted(report.family_breakdown.items()):
            acc = metrics.get("average_accuracy", 0.0)
            fgt = metrics.get("average_forgetting", 0.0)
            lines.append(f"| {family} | {acc:.4f} | {fgt:.4f} |")
        lines.append("")

    # Per-stage results
    if stage_results:
        lines.append("## Per-Stage Results")
        lines.append("")
        lines.append("| Stage | Family | Accuracy | Correct/Total |")
        lines.append("|---|---|---|---|")
        for r in stage_results:
            lines.append(
                f"| {r.stage} | {r.family} | {r.accuracy:.4f} | "
                f"{r.correct_count}/{r.total} |"
            )
        lines.append("")

    # Performance matrix
    if report.performance_matrix:
        pm = report.performance_matrix
        lines.append("## Performance Matrix")
        lines.append("")
        header = "| Train \\ Eval | " + " | ".join(str(s) for s in pm.eval_stages) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(pm.eval_stages) + 1))
        for i, train_stage in enumerate(pm.training_stages):
            row_vals = []
            for j in range(len(pm.eval_stages)):
                val = pm.matrix[i][j]
                row_vals.append(f"{val:.2f}" if val is not None else "-")
            lines.append(f"| {train_stage} | " + " | ".join(row_vals) + " |")
        lines.append("")

    return "\n".join(lines)


def write_full_report(
    report: MetricReport,
    stage_results: list[StageEvalResult],
    output_dir: Path,
) -> None:
    """Write all report formats to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    write_json(report, output_dir / "metrics.json")
    write_json(
        [r.model_dump() for r in stage_results],
        output_dir / "stage_results.json",
    )

    # CSV
    write_stage_results_csv(stage_results, output_dir / "stage_results.csv")
    write_metrics_csv(report, output_dir / "metrics.csv")

    # Markdown
    md = generate_markdown_summary(report, stage_results)
    with open(output_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(md)

    # Performance matrix as standalone JSON
    if report.performance_matrix:
        write_json(report.performance_matrix, output_dir / "performance_matrix.json")

"""Command-line interface for the continual benchmark framework."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

import continual_benchmark
from continual_benchmark.core.constants import DEFAULT_GLOBAL_SEED

app = typer.Typer(
    name="continual-benchmark",
    help="A research-grade benchmark framework for continual learning with LLMs.",
    add_completion=False,
)
console = Console()


@app.callback()
def main():
    """Continual Benchmark CLI."""


@app.command()
def version():
    """Print the version."""
    console.print(f"continual-benchmark {continual_benchmark.__version__}")


@app.command("list-streams")
def list_streams(
    search_dir: Optional[Path] = typer.Option(
        None, "--dir", help="Directory to search for stream definitions",
    ),
):
    """List available stream definitions."""
    from continual_benchmark.streams.loader import list_available_streams

    streams = list_available_streams(search_dir)
    if not streams:
        console.print("[yellow]No stream definitions found.[/yellow]")
        return

    table = Table(title="Available Streams")
    table.add_column("Stream ID", style="cyan")
    for s in streams:
        table.add_row(s)
    console.print(table)


@app.command("build-suite")
def build_suite(
    suite: str = typer.Option("v1", help="Suite version"),
    stream: str = typer.Option(..., help="Stream definition ID"),
    out: Path = typer.Option(..., help="Output directory"),
    seed: int = typer.Option(DEFAULT_GLOBAL_SEED, help="Global seed"),
    stream_dir: Optional[Path] = typer.Option(
        None, "--stream-dir", help="Custom directory for stream definitions",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Build a benchmark suite from a stream definition."""
    from continual_benchmark.build.build_suite import build

    if verbose:
        from continual_benchmark.utils.logging import setup_logging
        setup_logging(verbose=True)

    console.print(f"Building suite [cyan]{suite}[/cyan] with stream [cyan]{stream}[/cyan]...")
    build(
        suite=suite,
        stream_id=stream,
        output_dir=out,
        global_seed=seed,
        stream_dir=stream_dir,
    )
    console.print(f"[green]Suite built successfully at {out}[/green]")


@app.command("qa-suite")
def qa_suite(
    path: Path = typer.Option(..., help="Path to built suite"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run QA checks on a built suite."""
    from continual_benchmark.build.qa_suite import run_qa

    report = run_qa(path, verbose=verbose)
    console.print(f"\nQA Report: [cyan]{report.passed}[/cyan]/{report.total_checks} passed")
    if report.failed > 0:
        console.print(f"[red]{report.failed} checks FAILED[/red]")
        for check in report.checks:
            if not check.passed:
                console.print(f"  [red]FAIL[/red] {check.name}: {check.message}")
    if report.warnings > 0:
        console.print(f"[yellow]{report.warnings} warnings[/yellow]")
    if report.failed == 0:
        console.print("[green]All checks passed![/green]")


@app.command("inspect-stage")
def inspect_stage(
    path: Path = typer.Option(..., help="Path to built suite"),
    stage: int = typer.Option(..., help="Stage number to inspect"),
    split: str = typer.Option("train", help="Split to inspect"),
    n: int = typer.Option(3, help="Number of examples to show"),
):
    """Inspect examples from a specific stage."""
    import json

    from continual_benchmark.core.io import read_instances
    from continual_benchmark.utils.paths import split_path

    jsonl_path = split_path(path, stage, split)
    if not jsonl_path.exists():
        console.print(f"[red]File not found: {jsonl_path}[/red]")
        raise typer.Exit(1)

    instances = read_instances(jsonl_path)
    console.print(f"\n[cyan]Stage {stage}, split '{split}': {len(instances)} instances[/cyan]\n")

    for inst in instances[:n]:
        console.print(f"[bold]UID:[/bold] {inst.uid}")
        console.print(f"[bold]Family:[/bold] {inst.family}")
        console.print(f"[bold]Prompt:[/bold]\n{inst.prompt[:500]}...")
        console.print(f"[bold]Target:[/bold] {inst.target[:200]}")
        console.print(f"[bold]Metadata:[/bold] {json.dumps(inst.metadata, indent=2)}")
        console.print("---")


@app.command("score")
def score(
    gold: Path = typer.Option(..., help="Path to built suite (gold data)"),
    pred: Path = typer.Option(..., help="Path to predictions JSONL"),
    out: Path = typer.Option("reports/", help="Output directory for reports"),
    split: str = typer.Option("public_test", help="Split to score"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Score predictions against gold data."""
    from continual_benchmark.eval.score import score_predictions

    results = score_predictions(gold, pred, split=split, verbose=verbose)

    from continual_benchmark.core.io import write_json
    out.mkdir(parents=True, exist_ok=True)
    write_json([r.model_dump() for r in results], out / "stage_results.json")
    console.print(f"[green]Scored {len(results)} stages. Results at {out}[/green]")

    for r in results:
        console.print(
            f"  Stage {r.stage} ({r.family}): "
            f"{r.accuracy:.1%} ({r.correct_count}/{r.total})"
        )


@app.command("compute-metrics")
def compute_metrics(
    matrix: Path = typer.Option(..., help="Path to performance_matrix.json"),
    out: Path = typer.Option("reports/", help="Output directory"),
):
    """Compute continual learning metrics from a performance matrix."""
    from continual_benchmark.core.io import read_json, write_json
    from continual_benchmark.core.schemas import PerformanceMatrix
    from continual_benchmark.eval.metrics import compute_cl_metrics

    data = read_json(matrix)
    perf_matrix = PerformanceMatrix.model_validate(data)
    report = compute_cl_metrics(perf_matrix)

    out.mkdir(parents=True, exist_ok=True)
    write_json(report, out / "metrics.json")
    console.print(f"[green]Metrics written to {out / 'metrics.json'}[/green]")
    console.print(f"  Average forgetting: {report.average_forgetting:.4f}")
    console.print(f"  Average backward transfer: {report.average_backward_transfer:.4f}")


if __name__ == "__main__":
    app()

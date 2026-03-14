# Continual Benchmark

A research-grade benchmark framework for continual learning with language models.

## Why This Benchmark Exists

Existing continual learning benchmarks for LLMs are often too easy for frontier models, vulnerable to contamination, and rely on fuzzy evaluation. This benchmark addresses these issues with:

- **Generator-based data creation** — no static datasets, deterministic generation from seeds
- **Formal verification** — every task has an automatic, deterministic verifier
- **Concept drift** — operators and rules change across stages to create real forgetting
- **Controlled interference** — configurable task similarity and stream ordering
- **Rigorous CL metrics** — forgetting, backward/forward transfer, performance matrices

## Task Families

| Family | What it tests | Verification |
|---|---|---|
| **DSL Execution** | Execute programs in a custom DSL with changing operator semantics | Interpreter-based exact match |
| **Structured Transform** | Apply compositional JSON transformation rules | Canonicalized JSON comparison |
| **SQL Reasoning** | Answer questions over synthetic databases | SQLite query execution |
| **API Code Generation** | Write code using a stage-specific toy API | Sandboxed test execution |

## Installation

Requires [uv](https://docs.astral.sh/uv/). That's it — uv handles git, Python, virtualenv, and dependencies.

```bash
git clone <repo-url>
cd cl_benchmark
uv sync --all-extras    # creates venv + installs everything
```

## Quickstart

```bash
# List available streams
uv run continual-benchmark list-streams

# Build a benchmark suite
uv run continual-benchmark build-suite --suite v1 --stream v1_clustered --out artifacts/v1_clustered

# Run QA checks
uv run continual-benchmark qa-suite --path artifacts/v1_clustered

# Inspect examples from a stage
uv run continual-benchmark inspect-stage --path artifacts/v1_clustered --stage 3

# Score predictions
uv run continual-benchmark score --gold artifacts/v1_clustered --pred predictions.jsonl --out reports/run_001

# Compute CL metrics
uv run continual-benchmark compute-metrics --matrix reports/run_001/performance_matrix.json --out reports/run_001
```

## Streams

Three built-in stream orderings:

- **v1_clustered** (12 stages) — Tasks grouped by family. Strong within-family adaptation, cross-family forgetting.
- **v1_interleaved** (12 stages) — Round-robin across families. Continuous cross-family interference.
- **v1_drift** (16 stages) — Revisits families with increasing semantic drift. Maximum retention stress.

## Evaluation Protocol

At each stage, a model trains on the current stage's data, then is evaluated on **all** tasks seen so far.

**Track A (No-Data Replay):** Original raw data from past stages is unavailable.
**Track B (Limited-Memory Replay):** A bounded replay buffer stores examples from past stages.

## Metrics

- **Performance matrix** — Score on task `j` after training through stage `i`
- **Average accuracy** — Mean performance over seen tasks after each stage
- **Forgetting** — Max previous performance minus current performance per task
- **Backward transfer** — Performance change on old tasks after learning new ones
- **Forward transfer** — Zero-shot performance on unseen tasks

## Data Format

Instances are JSONL with fields: `uid`, `suite`, `stream_id`, `family`, `spec_id`, `stage`, `split`, `prompt`, `target`, `seed`, `metadata`.

Predictions are JSONL with fields: `uid`, `prediction` (plus optional `model_name`, `stage_trained_through`).

## Repository Structure

```
continual_benchmark/
  cli.py              # Typer CLI
  core/               # Schemas, registry, RNG, hashing, IO
  tasks/              # Task family implementations
    dsl_exec/         # DSL execution
    structured_transform/
    sql_reasoning/
    api_code/
  streams/            # Stream definitions and loading
  build/              # Suite builder and QA
  eval/               # Scoring, metrics, reporting
  baselines/          # Sequential FT, replay buffer, synthetic rehearsal
  utils/              # Logging, paths, text utilities
tests/                # Unit, integration, reproducibility tests
docs/                 # Documentation
```

## Adding New Task Families

See [docs/adding_new_task_family.md](docs/adding_new_task_family.md).

## Running Tests

```bash
uv run pytest tests/ -q
```

## License

MIT

# Continual Learning Benchmark for LLMs

A generator-based, verifier-based benchmark framework for studying catastrophic forgetting in language models.

## Why This Exists

Existing CL benchmarks for LLMs are too easy for frontier models, contamination-prone, and rely on fuzzy evaluation. This benchmark uses **formal specifications + reference verifiers** so that every answer is provably correct or incorrect — no ROUGE, no BERTScore, no human eval.

## Core Ideas

- **Generator-based**: Every instance is created programmatically from a seed. No static datasets to memorize.
- **Verifier-based**: Interpreters, SQL execution, and sandboxed test runners — not fuzzy metrics.
- **Concept drift**: Operator semantics, schemas, and API behaviors change across stages.
- **Rigorous CL metrics**: Full performance matrix, forgetting, backward/forward transfer, family breakdowns.

## Task Families

| Family | What the model does | How we verify | Drift mechanism |
|---|---|---|---|
| **DSL Execution** | Execute programs in a mini DSL | Interpreter exact match | Operator semantics change (e.g., `ADD(x,y)` → `x+y+1`) |
| **Structured Transform** | Apply compositional rule chains to JSON | Canonicalized JSON comparison | Rule precedence/schema changes |
| **SQL Reasoning** | Answer questions about synthetic databases | SQLite reference query execution | Schema drift (renamed columns, new tables) |
| **API Code Generation** | Write functions using a toy API | Sandboxed test execution | API behavior changes across versions |

## Evaluation Protocol

1. Training proceeds through **stages** (1, 2, ..., T)
2. After each stage, evaluate on **all seen tasks** — producing a performance matrix `M[i][j]`
3. Two tracks: **No-Data Replay** (no access to past data) and **Limited-Memory Replay** (bounded buffer)

## Metrics

- **Average accuracy**: Mean score across seen tasks after each stage
- **Forgetting**: `max(M[1..T][j]) - M[T][j]` — how much each task degraded
- **Backward transfer**: `M[T][j] - M[j][j]` — change on old tasks after learning new ones
- **Forward transfer**: `M[j-1][j]` — zero-shot performance before learning a task
- All metrics broken down by task family

## Default Streams

- `v1_clustered` (12 stages): Families grouped — DSL, Transform, SQL, API in blocks
- `v1_interleaved` (12 stages): Round-robin across families, 3 rounds
- `v1_drift` (16 stages): 4 rounds of all families with escalating drift intensity


## Baselines Included

- **Sequential fine-tuning**: Train stage-by-stage, no replay (pluggable trainer interface)
- **Replay buffer**: FIFO, reservoir sampling, or token-budget variants
- **Synthetic rehearsal**: Generate pseudo-examples from past task specs

## Key Design Decisions

- Deterministic generation: same seed → identical output, always
- Subprocess sandbox for code execution (not RestrictedPython — known vulnerabilities)
- 12–16 stage defaults (30 is too expensive for practical research; custom streams support any length)
- Compositional transform rules to resist saturation by frontier models
- Specs colocated with task family code for self-contained modularity

## Repository Structure

```
continual_benchmark/
  core/          # Schemas, registry, RNG, hashing, IO
  tasks/         # 4 task families (generator + verifier + canonicalizer each)
  streams/       # YAML stream definitions + loader/builder
  build/         # Suite builder + QA pipeline
  eval/          # Scoring, metrics, performance matrix, reporting
  baselines/     # Sequential FT, replay buffer, synthetic rehearsal
  cli.py         # Typer CLI (build-suite, qa-suite, score, compute-metrics, ...)
tests/           # 60 tests — unit, integration, reproducibility
docs/            # Protocol, task families, metrics, extension guide
```

## Status

- All 4 task families implemented and verified
- Full build → QA → score → metrics pipeline working end-to-end
- 60 tests passing
- 3 default stream definitions shipped
- Baseline scaffolds ready for real trainer integration

# Benchmark Overview

## What Is This?

A generator-based continual learning benchmark for language models. Unlike static dataset collections, this benchmark:

1. **Generates tasks from formal specifications** — every instance is created programmatically with a seed
2. **Verifies outputs automatically** — interpreters, SQL execution, and test runners, not fuzzy metrics
3. **Introduces true novelty at each stage** — custom DSLs, stage-specific rules, and evolving APIs
4. **Supports concept drift** — operator semantics, schema structures, and API behaviors change across stages
5. **Measures catastrophic forgetting rigorously** — through a full performance matrix and CL metrics

## Design Goals

- Remain challenging for frontier models (tasks require learning stage-specific behavior)
- Make forgetting observable and measurable
- Support controlled task similarity and interference
- Enable both no-replay and limited-memory replay evaluation tracks
- Be fully reproducible, extensible, and open

## How It Works

1. **Define a stream** — an ordered sequence of stages, each specifying a task family, difficulty, and drift config
2. **Build a suite** — the build pipeline generates train/dev/test splits for each stage deterministically
3. **Train continually** — a model trains stage by stage, evaluated on all seen tasks after each stage
4. **Compute metrics** — forgetting, backward transfer, forward transfer, family breakdowns

## Task Families

See [task_families.md](task_families.md) for detailed descriptions of each family.

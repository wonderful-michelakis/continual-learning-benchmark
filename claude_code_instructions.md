# Prompt for Claude Code

You are helping me build a **research-grade benchmark framework for continual learning with language models**.

This is not a toy project and not just a dataset dump. I want you to build a **full benchmark system** that can support a paper submission and ideally become a strong community benchmark.

Your job is to produce the **entire codebase skeleton and working implementation** for the benchmark, including dataset generation, task stream construction, evaluation, metrics, baselines, documentation, and developer tooling.

I want you to think like a **research engineer + benchmark designer**, not just like an app developer.

You should prioritize:
- correctness
- reproducibility
- clarity
- modularity
- scientific usefulness
- extensibility
- strong defaults
- excellent documentation
- clean code
- deterministic generation
- benchmark integrity

Do not build a hacky prototype. Build a solid benchmark framework that could reasonably back a paper and open-source release.

---

# 1. High-level context: what we are building

We are building a **new benchmark for continual learning (CL) for language models / instruction-tuned models**.

The motivation is that many existing continual learning benchmarks used in LLM work are either:

- too easy for modern frontier models  
- based on old static datasets  
- vulnerable to contamination  
- weakly diagnostic  
- hard to verify rigorously  
- mostly natural-language tasks with fuzzy evaluation metrics  

A common example is CL work built on top of SuperNI-style streams. Those are useful historically, but they are not enough for a next-generation benchmark because modern models often already do very well on many such tasks without much adaptation. That reduces headroom and makes it difficult to study real catastrophic forgetting.

So instead of another static collection of old NLP tasks, we want to build a **generator-based, verifier-based continual learning benchmark**.

The benchmark should:

1. remain challenging even for strong modern models  
2. introduce **true novelty** at each stage  
3. support **continual learning protocols**  
4. have **automatic and deterministic verification**  
5. enable **controlled task similarity**  
6. enable **concept drift**  
7. support **no-data replay** and **limited-memory replay** regimes  
8. be reproducible and extensible  
9. ideally support public and hidden/private test splits later  

We are not building a single dataset file.  
We are building a **benchmark framework**.

---

# 2. Scientific design goals

The benchmark must explicitly address the following scientific issues.

## 2.1 Avoid “too easy for current models”

The benchmark should not mainly test broad internet-exposed skills like generic summarization or sentiment analysis.

Instead, the benchmark must focus on:

- newly introduced rules  
- synthetic or authored specifications  
- structured reasoning  
- verifiable outputs  
- distribution shift  
- compositionality  
- concept drift  

The core idea is that a model should need to **learn stage-specific behavior** rather than relying purely on pretrained knowledge.

---

## 2.2 Make forgetting observable

The benchmark should be designed so that catastrophic forgetting is actually measurable and meaningful.

That means:

- tasks should not all be independent and trivial  
- there should be related tasks and conflicting tasks  
- there should be long task streams  
- there should be drift  
- there should be a reason for replay / regularization methods to matter  

---

## 2.3 Make evaluation rigorous

Evaluation should be:

- automatic  
- deterministic  
- reproducible  
- explainable  
- decomposable by task family / stream / drift type  

We should avoid fuzzy metrics whenever possible.

Prefer:

- exact match after canonicalization  
- program execution  
- interpreter-based checking  
- SQL execution  
- unit tests  
- structural equality  

---

## 2.4 Make the benchmark configurable and long-lived

We want:

- generator-based dataset creation  
- fixed-seed public splits  
- future private-test support  
- versioned suites  
- easy addition of new task families  
- standard metrics and reporting  

---

# 3. Core benchmark concept

The benchmark will contain several **task families**, each built around a **formal specification** and a **reference verifier**.

Each stage in the continual stream introduces one or more tasks derived from a spec.

Examples of task families include:

1. **DSL execution**
2. **Structured transformation**
3. **Database query reasoning**
4. **API code generation**

All of these are designed to be:

- novel  
- stage-specific  
- automatically verifiable  

The benchmark should support:

- train/dev/public_test/private_test splits  
- deterministic generation using seeds  
- stream definitions that specify stage order and drift patterns  
- continual evaluation after each stage  

---

# 4. What you must build

Build a Python codebase that implements the benchmark end to end.

Use modern, clean Python. Prefer Python 3.11+.

Use:

- `pydantic` for schemas/config objects if helpful  
- `typer` or `click` for CLI  
- `pytest` for tests  
- `sqlite3` for DB tasks  
- standard library as much as possible  
- `hypothesis` optionally for property-based tests in code tasks  
- `pyproject.toml`  
- modular package layout  

Do not overengineer with unnecessary frameworks.

---

# 5. Required repository structure

Create the repository with approximately this structure:

```text
continual_benchmark/
  README.md
  pyproject.toml
  .gitignore
  LICENSE
  docs/
    benchmark_overview.md
    protocol.md
    task_families.md
    metrics.md
    adding_new_task_family.md
  continual_benchmark/
    __init__.py
    cli.py
    core/
      __init__.py
      schemas.py
      registry.py
      rng.py
      hashing.py
      io.py
      constants.py
      manifests.py
    tasks/
      __init__.py
      families.py
      dsl_exec/
        __init__.py
        generator.py
        verifier.py
        interpreter.py
        canonicalize.py
        specs/          # YAML spec configs colocated with family code
      structured_transform/
        __init__.py
        generator.py
        verifier.py
        canonicalize.py
        specs/
      sql_reasoning/
        __init__.py
        generator.py
        verifier.py
        db_builder.py
        canonicalize.py
        specs/
      api_code/
        __init__.py
        generator.py
        verifier.py
        sandbox.py
        test_builder.py
        specs/
    streams/
      __init__.py
      definitions/
        v1_clustered.yaml
        v1_interleaved.yaml
        v1_drift.yaml
      loader.py
      builder.py
      similarity.py
    build/
      __init__.py
      build_suite.py
      qa_suite.py
      difficulty_checks.py
    eval/
      __init__.py
      score.py
      metrics.py
      matrix.py
      reporting.py
    baselines/
      __init__.py
      sequential_ft/
      replay_buffer/
      synthetic_rehearsal/
    utils/
      __init__.py
      logging.py
      paths.py
      text.py
  examples/
    build_v1.sh
    evaluate_dummy_model.sh
    sample_outputs/
  tests/
    test_schemas.py
    test_rng.py
    test_streams.py
    test_build.py
    test_metrics.py
    test_dsl_exec.py
    test_transform.py
    test_sql_reasoning.py
    test_api_code.py

    # 6. Required benchmark protocol

Implement the benchmark around **continual stages**.

At each stage:

- a model trains on current stage train data  
- then it is evaluated on the test/dev sets of all tasks seen so far  

We want support for two official tracks:

## Track A: No-Data Replay (ND)

After stage `t`, the original raw data of stages `< t` is no longer available.  
Methods may store:

- model weights
- fixed metadata if explicitly allowed
- synthetic samples if method chooses to generate them

But raw original examples from past stages may not be reused.

## Track B: Limited-Memory Replay (LM)

A method may store a bounded replay buffer from previous tasks:

- either bounded by example count
- or bounded by total token count

Implement the code so both tracks are representable, even if the full training loop is only lightly scaffolded.

---

# 7. Metrics to implement

Implement proper continual learning metrics.

At minimum:

- average performance after each stage
- forgetting
- backward transfer
- forward transfer
- task-wise retention

You should implement them in a standard and transparent way.

The code should compute:

## 7.1 Performance matrix

- rows = training stage
- columns = evaluation task/stage
- cell = score on task `j` after training through stage `i`

## 7.2 Average accuracy/performance

- average over seen tasks after each stage

## 7.3 Forgetting

- for task `j`, max previous performance on `j` minus current performance on `j`
- aggregate forgetting across tasks

## 7.4 Backward transfer

- improvement/degradation on old tasks after learning new tasks

## 7.5 Forward transfer

- optional, if evaluable from protocol assumptions

Also implement:

- family-wise breakdowns
- stream-wise breakdowns
- drift vs non-drift breakdowns

Produce nice JSON and CSV reports.

---

# 8. Benchmark data model

Create strong typed schemas for:

- spec
- instance
- manifest
- stream
- evaluation result
- metric report

Use Pydantic models or dataclasses.

## 8.1 Spec

A spec represents a task distribution for a stage.

Fields should include ideas like:

- suite name/version
- spec id
- task family
- family version
- stage id
- description / instruction / rules
- generator config
- verifier path
- metadata such as cluster id, drift type, difficulty tags

## 8.2 Instance

An instance should include:

- unique stable id
- spec id
- stage
- split
- prompt
- target
- metadata
- seed
- optional canonical structured representation

## 8.3 Manifest

Every built artifact should include:

- suite version
- generation parameters
- seed ranges
- hashes
- counts
- timestamps
- git commit if available

The goal is reproducibility.

---

# 9. Deterministic generation requirements

This is critical.

All task generation must be deterministic given:

- suite version
- split
- stage
- spec config
- seed

Implement a clear RNG utility so generation is reproducible.

For example:

- global suite seed
- split-specific seed offset
- stage-specific derived seeds
- instance-level seeds

Provide utilities to:

- derive child seeds
- hash configs consistently
- ensure exact repeatability

When the same suite is built twice with the same config, outputs should be identical.

---

# 10. Required task families

Implement at least the following four task families.

---

## 10.1 Task family 1: DSL Execution

### Purpose

This family introduces a small domain-specific language.  
The model receives:

- a task specification
- a program
- an input state

and must output the final result.

### Why this family matters

It creates:

- real novelty
- formal semantics
- easy verification
- strong opportunities for drift and interference

### Required implementation

Build:

- a simple parser or structured program representation
- an interpreter
- a generator that samples valid programs and inputs
- a verifier that compares predicted output with interpreter output
- canonicalization utilities

### Include concept drift

At later stages, some operators should change semantics.

Example idea:

Stage 1:

- `ADD(x,y) = x + y`
- `MUL(x,y) = x * y`

Later stage:

- `ADD(x,y) = x + y + 1`

Or add new operators:

- `SWAP`
- `FOLD`
- `CLIP`
- `MAPINC`

You do not need to use exactly those names.

### Prompt format

The prompt should contain:

- task instruction
- current DSL rules
- program
- input
- output placeholder

### Verification

Use exact comparison after canonicalization.

### Metadata

Store:

- AST-like structure
- program length
- operator counts
- whether drifted semantics are involved

### Difficulty control

Support configurable:

- program length
- nesting depth
- number of operators
- amount of drift

---

## 10.2 Task family 2: Structured Transformation

### Purpose

The model transforms a structured input into another structured output using stage-specific rules.

Examples:

- JSON transformation
- schema mapping
- field renaming
- nested structure conversion
- filter/project/aggregate operations
- value normalization

### Why this matters

It tests:

- rule following
- compositionality
- structured output correctness
- formatting retention

### Required implementation

Build:

- generator for random structured input objects
- **compositional rule chains** sampled from stage config (rules apply sequentially; later rules may depend on earlier results)
- reference solver that applies rules
- verifier that parses and canonicalizes JSON output

**Rule types implemented:**
1. **RenameFieldRule** — rename a field key
2. **RemoveFieldRule** — drop a field
3. **AddComputedFieldRule** — derive a new field from existing values (e.g., sum, product)
4. **FilterListRule** — filter list elements by a predicate
5. **FlattenNestedRule** — flatten nested objects to top-level with dotted keys
6. **ConditionalTransformRule** — apply a transformation only if a condition holds
7. **AggregateValuesRule** — compute aggregates (sum, mean, count) over numeric fields
8. **TypeCastRule** — convert field types (str→int, int→str, etc.)

Rules are composed into chains of increasing complexity at higher difficulty levels. This ensures the family tests genuine rule-following ability, not just trivial JSON manipulation.

### Prompt format

Prompt should include:

- transformation specification
- input object
- required output schema format

### Verification

- parse JSON
- canonicalize ordering
- compare exact structure and values

### Drift options

Allow later stages to:

- change transformation order
- change precedence rules
- add/remove fields
- introduce schema variants

### Metadata

Store:

- schema depth
- rule count
- transformation categories
- difficulty tags

---

## 10.3 Task family 3: SQL / Database Reasoning

### Purpose

The model reasons over synthetic relational databases.

### Why this matters

This is a nice middle ground between symbolic tasks and language tasks.  
It introduces:

- compositional reasoning
- data dependence
- exact verification
- stage-local schemas

### Required implementation

Build:

- synthetic schema generator
- deterministic database instance builder using sqlite
- question generator
- reference query execution
- verifier

### Output format

I prefer the model to output the **query result**, not SQL text, because scoring SQL robustly is harder.

Return results as canonical JSON tables or structured arrays.

### Prompt should include

- schema description
- table contents or relevant rows
- natural language query

Or optionally:

- schema plus compact table dumps

### Verification

- compute true result with sqlite
- compare canonicalized prediction

### Drift options

Allow:

- schema drift
- renamed columns
- new join paths
- new aggregation rules

### Metadata

Store:

- number of tables
- join count
- query type
- aggregation type
- schema family

---

## 10.4 Task family 4: API Code Generation

### Purpose

The model is introduced to a toy API or library with stage-specific docs and must generate code using that API.

### Why this matters

This family should be relatively strong against saturation because it requires learning a novel API surface.

### Required implementation

Build:

- toy API definitions or stage-specific library docs
- prompt templates asking for functions to be implemented
- reference tests
- code verifier using execution in a constrained sandbox

### Verification

- run tests
- exact pass/fail or partial score by tests passed
- use safe constrained execution

### Safety

Do not make the sandbox insecure.
Keep it minimal and controlled.
Use a restricted execution environment suitable for benchmark code checking.

**Implementation approach:** Use `subprocess.run()` with timeout for each test execution, `resource.setrlimit()` (macOS/Linux) to cap CPU time, memory, and file descriptors, and an **AST pre-scan** as a lightweight first pass to reject dangerous patterns (`__import__`, `os`, `sys`, `subprocess`, `open`, `exec`, `eval`). A restricted builtins dict is passed to the exec namespace — no filesystem/network access. This is the approach used by HumanEval and similar benchmarks. RestrictedPython is avoided due to known RCE vulnerabilities.

### Optional

Use `hypothesis` to generate extra edge-case tests.

### Drift options

Allow:

- API behavior changes
- renamed functions
- changed argument expectations
- new constraints

### Metadata

Store:

- required functions
- test count
- API version
- drift tags

---

# 11. Streams and task ordering

Implement stream definitions using YAML or JSON.

A stream specifies:

- stream id
- suite version
- ordered stages
- which specs appear at each stage
- metadata about similarity/drift

**Default stream sizes:** 12–16 stages (not 30). 30 stages is expensive to evaluate; researchers can define custom longer streams. The framework supports arbitrary lengths.

Implement at least these example streams:

## 11.1 Clustered stream (`v1_clustered` — 12 stages)

Tasks from similar families are grouped together: DSL(1-3), Transform(4-6), SQL(7-9), API(10-12), with progressive drift.

This can create strong within-family adaptation and later cross-family forgetting.

## 11.2 Interleaved stream (`v1_interleaved` — 12 stages)

Round-robin families × 3 rounds with increasing drift.

This creates a different interference profile.

## 11.3 Drift-heavy stream (`v1_drift` — 16 stages)

4 rounds of all families with escalating drift (0.3→0.7→0.9 intensity).

This should strongly stress continual retention.

Implement stream loading, validation, and materialization.

---

# 12. Similarity metadata

The benchmark should support controlled task similarity.

Implement metadata fields like:

- cluster id
- family id
- drift parent
- drift type
- expected similarity level
- expected interference level

Also implement utilities to summarize similarity structure, even if the similarity is not fully learned from embeddings.

A lightweight hand-authored similarity matrix is fine.

---

# 13. Build pipeline

Implement CLI commands to build suites and artifacts.

Examples of desired commands:

```bash
python -m continual_benchmark.cli build-suite --suite v1 --stream v1_clustered --out data/v1_clustered
python -m continual_benchmark.cli qa-suite --path data/v1_clustered
python -m continual_benchmark.cli inspect-stage --path data/v1_clustered --stage 3

The build pipeline should:

*load stream definition
*instantiate specs
*generate train/dev/public_test/private_test splits
*write JSONL or Parquet artifacts
*write manifests
*run QA check

# 14. QA / validation suite

This is very important. Build a strong QA system.

The QA system should check:

- schema validity
- deterministic regeneration
- no duplicate IDs
- no duplicate prompts across splits where disallowed
- reference verifier correctness
- split sizes
- prompt/target not empty
- task-family-specific invariants

Also implement checks inspired by dynamic benchmark quality criteria.

## 14.1 Correctness

Reference solutions must always verify.

For every generated instance:

- run the reference solver
- run the verifier on the reference output
- assert a perfect score or successful verification

If this fails, the generator or verifier is broken.

## 14.2 Collision

Avoid duplicates or trivial repeats.

Check for:

- duplicate instance IDs
- duplicate prompts within a split
- duplicate prompt-target pairs across splits
- overly similar instances if they violate diversity constraints

Use canonicalized prompt hashes and structured metadata hashes where useful.

## 14.3 Stability

Rebuilding from the same config should reproduce the same data.

Implement checks that:

- rebuild a small sample of the suite using the same seeds
- compare hashes, manifests, and serialized instances
- confirm byte-level or canonical equality

## 14.4 Diversity

Within a stage, examples should vary in content and structure.

Examples of diversity checks:

- operator distribution for DSL tasks
- rule diversity for structured transforms
- schema/query diversity for SQL tasks
- API/test diversity for code tasks

A stage should not collapse to near-identical examples.

## 14.5 Complexity

Tasks should meet configurable difficulty thresholds.

Examples:

- DSL programs must exceed a minimum length/depth
- transform tasks must use at least `k` rules
- SQL tasks must exceed a minimum query complexity
- code tasks must include enough meaningful tests

Provide a QA report summarizing all checks, warnings, and failures.

---

# 15. Difficulty control

Implement difficulty parameters for each family.

Examples:

## DSL

- program length
- nesting depth
- operator variety
- number of intermediate variables
- drift intensity

## Structured transform

- schema depth
- number of rules
- number of nested fields
- list operations
- aggregation count

## SQL

- number of tables
- join depth
- aggregation complexity
- filter complexity
- schema size

## API code

- function count
- edge-case count
- API surface size
- control-flow complexity in the expected solution
- number of tests/property tests

Support:

- per-stage difficulty configs
- suite-wide difficulty scaling
- family-specific default difficulty schedules

This is important so the benchmark can evolve over time and remain useful.

---

# 16. Evaluation framework

Implement a robust evaluation system.

Given predictions for one or more stages, the evaluation code should:

- load gold data
- route each example to the correct verifier
- compute per-example scores
- aggregate by task, stage, family, and stream
- build the continual learning performance matrix
- compute metrics
- write reports

Desired commands:

```bash
python -m continual_benchmark.cli score \
  --gold data/v1_clustered \
  --pred predictions.jsonl \
  --out reports/run_001

python -m continual_benchmark.cli compute-metrics \
  --matrix reports/run_001/performance_matrix.json \
  --out reports/run_001/metrics.json
  
Reports should include:

- JSON
- CSV
- optional markdown summary

The evaluation framework should also support:

* partial submissions
* stage-specific evaluation
* family-level summaries
* error analysis outputs for failed predictions

# 17. Baselines

Implement at least lightweight scaffold baselines.

These do not need to fully train giant models, but the codebase should include realistic benchmark baseline logic.

## 17.1 Sequential fine-tuning baseline

A minimal baseline that sequentially trains on stage data only.

You can stub the actual model training if necessary, but structure it clearly so a real model can later be plugged in.

The scaffold should include:

- stage loop
- train hook
- evaluate hook
- checkpoint hook
- report generation

Design this baseline so that someone can later plug in:

- a Hugging Face trainer
- LoRA / PEFT fine-tuning
- full fine-tuning
- custom training code

The important thing is that the benchmark provides a canonical sequential-training skeleton.

## 17.2 Replay buffer baseline

A limited-memory baseline with:

- reservoir sampling or FIFO
- configurable buffer size
- example-count and token-budget variants

This baseline should expose exactly how replay is managed.

It should make it easy to compare:

- no replay
- small replay
- larger replay
- different sampling strategies

## 17.3 Synthetic rehearsal baseline

A simple framework for pseudo-rehearsal:

- generate synthetic examples from task specs or prior summaries
- combine them with current stage training
- track synthetic data provenance separately from real data

This does not need to reproduce any paper exactly, but it should be a credible placeholder for self-synthesized rehearsal style methods.

The key is that the benchmark framework should make such methods easy to plug in.

### Important design note

Even if the baselines are lightweight, they should still reflect the benchmark protocol correctly.

That means:

- stage-by-stage training
- evaluation on all seen tasks after each stage
- storage accounting for memory-based methods
- explicit separation between real data, replay data, and synthetic data

---

# 18. Example data format

Please standardize a JSONL format for instances.

A single instance might look like:

```json
{
  "uid": "v1_dsl_stage03_train_000123_abcd1234",
  "suite": "v1",
  "stream_id": "v1_clustered",
  "family": "dsl_exec",
  "spec_id": "dsl_exec:v1:stage03",
  "stage": 3,
  "split": "train",
  "prompt": "You are given the following DSL rules...\nProgram: ...\nInput: ...\nOutput:",
  "target": "{\"result\": 17}",
  "metadata": {
    "difficulty": 3,
    "program_length": 5,
    "operators": ["ADD", "MUL", "CLIP"],
    "drift_type": null
  },
  "seed": 18723123
}

The exact schema can differ, but it should be clear and stable.

Also define related artifact formats for:

- manifests
- stage summaries
- QA reports
- evaluation outputs
- metrics reports
## 18.1 Suggested artifact files

A built suite should ideally contain files like:

    artifacts/
      v1_clustered/
        manifest.json
        stream.json
        stage_001/
          train.jsonl
          dev.jsonl
          public_test.jsonl
          private_test_manifest.json
          summary.json
        stage_002/
          train.jsonl
          dev.jsonl
          public_test.jsonl
          private_test_manifest.json
          summary.json
        qa/
          qa_report.json
          qa_report.md

You may refine this structure, but keep it organized and reproducible.

### Artifact file descriptions

#### `manifest.json`

Contains the high-level metadata describing the suite build.

Example fields:

    {
      "suite": "v1",
      "stream_id": "v1_clustered",
      "created_at": "2026-03-14T12:00:00Z",
      "generator_version": "0.1.0",
      "global_seed": 123456,
      "stage_count": 12,
      "families": ["dsl_exec", "structured_transform", "sql_reasoning", "api_code"],
      "git_commit": "optional_commit_hash"
    }

This file is critical for reproducibility.

#### `stream.json`

Materialized version of the stream definition used to generate the benchmark.

Example:

    {
      "stream_id": "v1_clustered",
      "stages": [
        {
          "stage": 1,
          "spec_ids": ["dsl_exec:v1:stage01"]
        },
        {
          "stage": 2,
          "spec_ids": ["dsl_exec:v1:stage02"]
        }
      ]
    }

This ensures that anyone can reconstruct exactly which stages and specs were used.

#### `summary.json`

Per-stage metadata summary.

Suggested contents:

- number of train/dev/test examples
- family name
- spec id
- difficulty settings
- drift metadata
- hashes of split files

This file helps with quick inspection and debugging.

#### `private_test_manifest.json`

This file should not contain hidden targets, but should contain enough metadata to support future hidden/private evaluation workflows.

Possible fields:

- instance IDs
- seeds or seed references if appropriate
- verifier identifiers
- schema/version information
- any hidden-eval bookkeeping fields

---

## 18.2 Prediction file format

Please also define a clean prediction format.

A prediction JSONL file should ideally contain one prediction per instance:

    {
      "uid": "v1_dsl_stage03_public_test_000123_abcd1234",
      "prediction": "{\"result\": 17}"
    }

Optional extra fields:

    {
      "uid": "v1_dsl_stage03_public_test_000123_abcd1234",
      "prediction": "{\"result\": 17}",
      "model_name": "example-model",
      "stage_trained_through": 3,
      "metadata": {
        "generation_mode": "greedy"
      }
    }

The scoring pipeline should only require `uid` and `prediction`, but should gracefully preserve optional metadata.

---

# 19. Prompt design philosophy for generated tasks

Prompts should be:

- clear
- self-contained when possible
- deterministic in formatting
- realistic enough for instruction-tuned LMs
- easy to parse/debug

Do not create messy prompts.

For each family, define prompt templates and keep them readable.

Also include developer docs explaining:

- what each prompt format is testing
- why it exists
- how it should be interpreted

Prompts should avoid unnecessary verbosity and should clearly separate:

- task description
- rules/specification
- input/context
- required output format

## 19.1 Output format discipline

Each prompt should explicitly require one output format.

Examples:

- JSON object
- JSON list
- scalar wrapped in JSON
- code only

Avoid ambiguous output instructions.

The benchmark should minimize false negatives caused by prompt ambiguity.

## 19.2 Prompt-template requirements by family

### DSL execution prompts

Each DSL prompt should clearly include:

- a short task description
- the active DSL rules
- the program
- the input bindings
- the exact expected output format

Avoid overly narrative wording.

### Structured transformation prompts

Each transform prompt should clearly include:

- the transformation rules
- the input object
- any schema constraints
- the required output format

The model should not have to infer hidden conventions.

### SQL reasoning prompts

Each SQL prompt should clearly include:

- schema definition
- relevant table contents
- the question
- the return format

If ordering matters, say so explicitly. If it does not matter, the verifier can canonicalize order.

### API code prompts

Each code-generation prompt should clearly include:

- the toy API documentation
- the required function signature
- the required behavior
- output constraints such as “return code only”

If test-relevant assumptions exist, they should be documented in the prompt or spec.

---

# 20. Canonicalization

This is important.

For verification, implement canonicalization utilities.

Examples:

- JSON sorted keys
- normalized whitespace
- numeric normalization
- row-order-insensitive table comparison when appropriate
- exact structure normalization

Make verifiers robust to harmless formatting variation but strict on semantics.

Examples of what should be canonicalized:

- `{"a":1,"b":2}` vs `{"b":2,"a":1}`
- floats like `1`, `1.0`, `1.000`
- extra spaces or trailing newlines
- table rows where ordering is irrelevant

Do not canonicalize away meaning-changing differences.

## 20.1 Family-specific canonicalization

Implement family-specific helpers when useful.

### DSL

- parse JSON result
- normalize numbers
- ensure required keys exist

### Structured transform

- parse JSON
- recursively sort object keys
- preserve list order unless rule semantics say order is irrelevant

### SQL

- canonicalize result rows
- optionally sort rows when the query semantics do not imply ordering
- normalize scalar/table representations

### API code

- normalize code fences when necessary before extraction
- isolate executable code body safely
- do not accept malformed code as correct

## 20.2 Canonicalization API design

Create a clean canonicalization interface, for example:

- `canonicalize_json_output`
- `canonicalize_table_output`
- `canonicalize_scalar_output`
- `extract_code_block`
- `normalize_numeric_value`

Make it easy for new task families to reuse these utilities.

---

# 21. Code quality expectations

I want strong engineering quality.

Please:

- write clear docstrings
- use type hints throughout
- keep functions reasonably small
- avoid global state
- use tests
- separate generation from verification cleanly
- separate config from logic
- avoid magic constants
- comment the scientifically important parts

Also:

- prefer explicit naming over cleverness
- make modules easy to extend
- include helpful error messages
- keep public interfaces stable and documented

Where appropriate, use:

- dataclasses or Pydantic models for structured data
- small pure functions for canonicalization and validation
- clear dependency boundaries across modules

## 21.1 Error handling expectations

Error handling should be practical and informative.

Examples:

- invalid generated instance should raise a meaningful generation error
- malformed prediction should return a structured verification failure
- missing stream config should produce a clear CLI error
- hash/manifest mismatches should produce actionable QA diagnostics

---

# 22. Testing requirements

Write tests for:

- schema validation
- reproducibility
- each task family generator/verifier pair
- stream parsing
- metrics correctness
- manifest generation
- canonicalization
- build pipeline sanity

At minimum include:

- unit tests
- smoke tests
- a few end-to-end integration tests

Examples:

- generate a mini suite with 2 stages
- verify reference outputs pass
- build a performance matrix from synthetic predictions
- compute forgetting correctly
- rebuild the same suite twice and compare hashes

Tests should be easy to run with:

    pytest -q

## 22.1 Suggested test strategy

Please organize tests into three levels:

### Unit tests

Small, focused checks for:

- parsers
- interpreters
- verifiers
- canonicalizers
- metric functions
- seed derivation

### Integration tests

Checks that:

- build a mini suite
- run QA
- score mock predictions
- compute CL metrics

### Regression / reproducibility tests

Checks that:

- manifests remain stable
- regenerated tiny fixtures match expected hashes
- stream definitions parse and materialize identically

## 22.2 Minimum smoke-test path

At a minimum, the repo should include one smoke-test flow that proves the full system works:

1. load a tiny stream
2. generate two small stages
3. verify all reference targets
4. create dummy predictions
5. score predictions
6. compute CL metrics
7. emit reports

This is critical for maintainability.

---

# 23. Documentation requirements

Please create comprehensive docs.

At minimum:

- `README.md`
- benchmark overview
- protocol explanation
- task family descriptions
- metrics explanation
- developer guide for adding new task families
- CLI usage examples

The README should explain:

- what the benchmark is
- why it exists
- how to install
- how to build a suite
- how to inspect examples
- how to score predictions
- how to extend it

The docs should be written for both:

- benchmark users
- benchmark contributors

## 23.1 Documentation quality

Please make the docs:

- technically precise
- easy to navigate
- consistent with actual code
- useful for a research audience

Include example commands and example outputs.

Also document:

- how concept drift is represented
- how similarity metadata is encoded
- how to add a new stream
- how to add a new verifier-backed family

## 23.2 README structure suggestion

A good README should include sections like:

- Overview
- Why this benchmark exists
- Installation
- Quickstart
- Building a suite
- Running QA
- Scoring predictions
- Computing metrics
- Repository structure
- Adding new task families
- License

---

# 24. Non-goals / what not to do

Do not:

- build a web app
- build an overcomplicated database backend
- depend on heavyweight orchestration frameworks
- write placeholder code everywhere without working implementations
- focus only on docs and ignore functionality
- create only one trivial toy task and stop

We need a genuinely useful, modular, working benchmark framework.

Also do not:

- hard-code all examples as static files
- couple all logic tightly into one script
- make reproducibility an afterthought
- use unsafe code execution for the code-task verifier
- rely on brittle prompt parsing logic where structured metadata should be used

## 24.1 General engineering anti-patterns to avoid

Please avoid:

- giant monolithic files
- circular imports
- hidden global registries without documentation (the task family registry uses an explicit `@register_family` decorator pattern in `core/registry.py`, with `TASK_FAMILY_REGISTRY` dict populated via imports in `tasks/__init__.py` — no hidden auto-discovery)
- non-deterministic generation behavior
- unverifiable task designs
- ad hoc CLI behavior that bypasses core abstractions

---

# 25. Expected implementation scope

I want a real implementation, but it is okay to phase it intelligently.

## Preferred order of implementation

Please build in this order.

### Phase 1: Core infrastructure

- schemas
- registry
- RNG
- hashing
- manifests
- CLI
- stream loading

### Phase 2: Task families

- DSL execution
- Structured transformation
- SQL reasoning

### Phase 3: Build and evaluation

- suite builder
- QA
- scoring
- continual-learning metrics
- reports

### Phase 4: API code family

- safe test execution
- prompt generation
- test-based verification

### Phase 5: Baseline scaffolds + extra docs + polish

If needed, clearly separate “fully implemented” from “scaffolded but extendable”.

## 25.1 Deliver incrementally but coherently

If you need to implement in stages, make each stage runnable.

For example:

- after Phase 1, stream definitions and schemas should already work
- after Phase 2, a mini suite should already build
- after Phase 3, metrics should already compute end-to-end

Do not leave the project in a state where nothing runs until the very end.

## 25.2 Prioritization principle

If tradeoffs arise, prioritize in this order:

1. correctness
2. reproducibility
3. modularity
4. benchmark usefulness
5. polish

A smaller working system is better than a large but fragile one.

---

# 26. Concrete examples for each task family

Use these as guidance.

## 26.1 DSL example

Prompt example:

    Task: Execute the following program according to the DSL rules.

    Rules:
    - ADD(x, y) returns x + y
    - MUL(x, y) returns x * y
    - CLIP(x, lo, hi) returns x clipped to [lo, hi]

    Program:
    z1 = ADD(a, b)
    z2 = MUL(z1, c)
    out = CLIP(z2, 0, 20)

    Input:
    a = 2
    b = 3
    c = 5

    Return the final value of out as JSON with key "result".

Target:

    {"result": 20}

Later drift example:

    Rules:
    - ADD(x, y) returns x + y + 1
    ...

This should be reflected by the generator.

## 26.2 Structured transformation example

Prompt example:

    Task: Transform the input JSON according to the rules.

    Rules:
    1. Rename field "fname" to "first_name"
    2. Rename field "lname" to "last_name"
    3. Create a new field "full_name" by concatenating first_name and last_name with a space
    4. Remove the field "temp"
    5. Under "scores", keep only values >= 50

    Input:
    {
      "fname": "Ada",
      "lname": "Lovelace",
      "temp": "remove_me",
      "scores": [40, 50, 88]
    }

    Return valid JSON only.

Target:

    {
      "first_name": "Ada",
      "last_name": "Lovelace",
      "full_name": "Ada Lovelace",
      "scores": [50, 88]
    }

## 26.3 SQL reasoning example

Prompt example:

    Database schema:

    employees(id, name, dept_id, salary)
    departments(dept_id, dept_name)

    employees:
    (1, "Alice", 10, 120)
    (2, "Bob", 10, 100)
    (3, "Cara", 20, 150)

    departments:
    (10, "Research")
    (20, "Sales")

    Question:
    What are the names of employees in Research with salary >= 110?

    Return the answer as a JSON list of strings.

Target:

    ["Alice"]

## 26.4 API code example

Prompt example:

    You are given a library with the following API:

    normalize(xs): returns a list scaled so the sum is 1.0. If the sum is 0, returns a zero vector of same length.
    top_k(xs, k): returns the k largest values in descending order.
    pairwise_sum(xs): returns [xs[0]+xs[1], xs[2]+xs[3], ...]. If odd length, last element is kept.

    Write a Python function:

    def summarize(xs, k):
        ...
        
    It should:
    1. normalize xs
    2. compute pairwise_sum on the normalized list
    3. return top_k of the result with parameter k

The verifier should run tests against the submitted code.

---

# 27. Deliverables I expect from you

I want you to actually generate:

1. the repository structure
2. the core implementation files
3. working code for the main benchmark flow
4. tests
5. docs
6. sample stream definitions
7. example commands

Do not just describe the architecture.  
Actually write the code.

If you need to phase the implementation across multiple files or steps, do so systematically.

## 27.1 Deliverable quality bar

The generated repository should be:

- runnable
- coherent
- documented
- testable
- reasonably polished
- easy for a researcher to extend

---

# 28. Important implementation choices

Please make the following design choices unless there is a strong reason not to:

- use JSONL for dataset artifacts
- use YAML for stream/spec configs where appropriate
- use deterministic seed-based generation
- use exact/canonicalized verification
- use Typer for CLI
- use Pytest for tests
- use SQLite for SQL tasks
- use safe restricted execution for code tasks (subprocess + resource limits + AST pre-scan; NOT RestrictedPython)
- use Pydantic for structured schemas/configs

If you deviate, explain why.

## 28.1 Dependency philosophy

Prefer a small, robust dependency set.

Only introduce dependencies when they materially improve:

- correctness
- safety
- ergonomics
- maintainability

Avoid dependency sprawl.

---

# 29. Developer ergonomics

Make the code pleasant to use.

I want commands like:

    python -m continual_benchmark.cli list-streams
    python -m continual_benchmark.cli build-suite --suite v1 --stream v1_clustered
    python -m continual_benchmark.cli inspect-stage --path ./artifacts/v1_clustered --stage 2
    python -m continual_benchmark.cli score --gold ./artifacts/v1_clustered --pred ./predictions.jsonl
    python -m continual_benchmark.cli qa-suite --path ./artifacts/v1_clustered

Also include:

- sample outputs
- a tiny smoke-test suite
- examples in the README

## 29.1 CLI expectations

The CLI should be:

- discoverable
- consistent
- script-friendly
- well-documented

Subcommands should have clear help text and sensible defaults.

---

# 30. How I want you to proceed

Proceed in a structured way:

1. first outline the implementation plan briefly
2. then create the repository structure
3. then implement core infrastructure
4. then implement task families
5. then implement build/eval/metrics
6. then implement tests/docs
7. then summarize what is complete and what remains extendable

Do not stall in abstract planning.  
Move into code generation quickly.

Where something is ambiguous, make a strong reasonable decision and document it.

## 30.1 Execution style

When implementing:

- make reasonable assumptions instead of blocking on ambiguity
- keep the system internally consistent
- document the key design decisions
- ensure each phase leaves the repository in a usable state

---

# 31. Final reminder of the research intent

This benchmark exists because we want a **continual learning benchmark for modern language models that is not trivially saturated by pretrained knowledge**.

The benchmark should emphasize:

- new rules
- structured reasoning
- verifiable outputs
- continual adaptation
- concept drift
- controlled interference
- replay-relevant evaluation

In short:

We are building a **generator-based continual learning benchmark framework for LLMs**, with multiple task families, deterministic data generation, rigorous verification, continual-learning metrics, and clean research-grade infrastructure.

Please build it accordingly.

---

# 32. Implementation Deviation Notes

The following deviations from the original spec were made during implementation, with justifications:

### 1. Sandbox: subprocess + resource limits + AST pre-scan (not RestrictedPython)
**Why:** RestrictedPython has known RCE vulnerabilities (March 2025 disclosure) and is not truly sandboxed. The subprocess approach (used by HumanEval and similar benchmarks) provides genuine process isolation with configurable resource limits.

### 2. Default stream sizes: 12–16 stages (not 30)
**Why:** 30 stages is expensive to evaluate for practical research use. Default streams: `v1_clustered` (12 stages), `v1_interleaved` (12 stages), `v1_drift` (16 stages). The framework supports arbitrary-length custom streams.

### 3. Spec configs colocated with task family code (not top-level `specs/` directory)
**Why:** Each task family is self-contained with its own `specs/` subdirectory. This makes adding new families easier and avoids scattering related files across the tree. Stream definitions (which reference specs across families) remain in `streams/definitions/`.

### 4. Structured transform: compositional rule chaining
**Why:** Simple field renames are trivially solvable by frontier models. The implementation uses 8 composable rule types (rename, remove, computed fields, list filtering, nested flattening, conditional transforms, aggregation, type casting) applied in chains, ensuring the family tests genuine rule-following ability.

### 5. Registry: explicit decorator-based pattern
**Why:** `TASK_FAMILY_REGISTRY` dict is populated via `@register_family` decorators, with all imports explicit in `tasks/__init__.py`. No hidden auto-discovery or metaclass magic. Documented in `core/registry.py`.

### 6. `tasks/families.py` added
**Why:** Concrete `TaskFamily` subclass implementations (DSLExecFamily, StructuredTransformFamily, SQLReasoningFamily, APICodeFamily) are registered in a single `tasks/families.py` file rather than scattered across family subdirectories, keeping the registry centralized and import-explicit.
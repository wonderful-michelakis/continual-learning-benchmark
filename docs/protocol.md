# Evaluation Protocol

## Continual Learning Setup

The benchmark follows a standard continual learning protocol:

1. Training proceeds through **stages** (1, 2, ..., T)
2. At each stage `t`, the model receives training data for stage `t`
3. After training on stage `t`, the model is evaluated on the test sets of **all stages 1..t**
4. This produces a **performance matrix** where entry `M[t][j]` = score on task `j` after training through stage `t`

## Tracks

### Track A: No-Data Replay (ND)

After stage `t`, the original raw data of stages `< t` is **not available**. Methods may store:
- Model weights / checkpoints
- Fixed metadata (if explicitly allowed by the method)
- Synthetic samples (if the method generates them)

But raw original examples from past stages may not be reused.

### Track B: Limited-Memory Replay (LM)

Methods may store a **bounded replay buffer** from previous tasks:
- Bounded by example count (e.g., 500 examples total)
- Or bounded by total token count (e.g., 100K tokens)

The buffer management strategy (FIFO, reservoir sampling, etc.) is part of the method.

## Splits

Each stage produces four splits:
- `train` — used for training (200 examples by default)
- `dev` — for method development and hyperparameter tuning
- `public_test` — for public evaluation and leaderboard
- `private_test` — held out for future blind evaluation

## Scoring

Predictions are JSONL files with `uid` and `prediction` fields. The scoring pipeline:
1. Matches predictions to gold instances by UID
2. Routes each example to the correct family verifier
3. Computes per-example binary scores (correct/incorrect)
4. Aggregates by stage, family, and stream

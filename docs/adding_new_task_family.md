# Adding a New Task Family

## Steps

### 1. Create the family directory

```
continual_benchmark/tasks/my_family/
    __init__.py
    generator.py
    verifier.py
    canonicalize.py
    specs/          # Optional YAML specs
```

### 2. Implement the generator

Your `generator.py` should export a function:

```python
def generate_my_instance(
    spec: Spec,
    stream_id: str,
    index: int,
    split: str,
    seed: int,
    rng: random.Random,
) -> Instance:
    ...
```

Key requirements:
- Generation must be **deterministic** given the seed and rng
- Return a fully populated `Instance` with prompt, target, and metadata
- Include difficulty and drift info in metadata

### 3. Implement the verifier

```python
def verify_my_task(
    prediction: str,
    target: str,
    metadata: dict[str, Any],
) -> tuple[bool, float]:
    ...
```

Returns `(correct, score)` where score is 0.0-1.0.

### 4. Implement canonicalization

```python
def canonicalize_my_output(output: str) -> str:
    ...
```

Handles harmless formatting variation while preserving semantic content.

### 5. Register the family

In `continual_benchmark/tasks/families.py`:

```python
@register_family("my_family")
class MyFamily(TaskFamily):
    @property
    def name(self) -> str:
        return "my_family"

    def generate_instance(self, spec, stream_id, index, split, seed, rng):
        from continual_benchmark.tasks.my_family.generator import generate_my_instance
        return generate_my_instance(spec, stream_id, index, split, seed, rng)

    def verify(self, prediction, target, metadata):
        from continual_benchmark.tasks.my_family.verifier import verify_my_task
        return verify_my_task(prediction, target, metadata)

    def canonicalize(self, output):
        from continual_benchmark.tasks.my_family.canonicalize import canonicalize_my_output
        return canonicalize_my_output(output)
```

### 6. Add to FamilyName enum

In `continual_benchmark/core/constants.py`, add your family to `FamilyName`.

### 7. Import in tasks/__init__.py

Add your family class to the imports in `continual_benchmark/tasks/__init__.py`.

### 8. Create stream definitions

Add stages using your family to YAML stream definitions.

### 9. Write tests

Add tests in `tests/test_my_family.py` covering:
- Generator produces valid instances
- Verifier correctly validates reference targets (self-check)
- Canonicalization handles edge cases
- Concept drift works as expected

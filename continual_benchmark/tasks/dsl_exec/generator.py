"""DSL execution task generator.

Generates valid DSL programs with configurable difficulty, operator sets,
and concept drift via semantic overrides.
"""

from __future__ import annotations

import json
import random
from typing import Any

from continual_benchmark.core.constants import FamilyName, Split
from continual_benchmark.core.hashing import instance_uid
from continual_benchmark.core.schemas import Instance, Spec
from continual_benchmark.tasks.dsl_exec.interpreter import (
    DEFAULT_OPERATORS,
    Assignment,
    DSLInterpreter,
    OpCall,
    Program,
)


def generate_dsl_instance(
    spec: Spec,
    stream_id: str,
    index: int,
    split: str,
    seed: int,
    rng: random.Random,
) -> Instance:
    """Generate a single DSL execution instance.

    Creates a random program from the spec's operator set, generates
    random input bindings, executes the program with the interpreter
    to get the reference target, and formats everything into a prompt.
    """
    # Extract config
    config = spec.generator_config
    operator_names = config.get("operators", ["ADD", "SUB", "MUL"])
    semantic_overrides = config.get("semantic_overrides", {})

    # Difficulty parameters
    program_length = spec.difficulty.program_length or rng.randint(3, 6)
    nesting_depth = spec.difficulty.nesting_depth or 1

    # Create interpreter with appropriate semantics
    # Filter to only the operators we want for this stage
    active_ops = {
        name: DEFAULT_OPERATORS[name]
        for name in operator_names
        if name in DEFAULT_OPERATORS
    }
    interpreter = DSLInterpreter(
        operators=active_ops,
        semantic_overrides=semantic_overrides if semantic_overrides else None,
    )

    # Generate input variables
    num_inputs = rng.randint(2, min(4, program_length))
    input_names = [chr(ord("a") + i) for i in range(num_inputs)]
    inputs = {name: rng.randint(-20, 20) for name in input_names}

    # Generate program
    program = _generate_program(
        rng=rng,
        operator_names=operator_names,
        input_names=input_names,
        program_length=program_length,
        nesting_depth=nesting_depth,
        active_ops=active_ops,
    )

    # Execute to get reference answer
    result = interpreter.execute(program, {k: float(v) for k, v in inputs.items()})

    # Round result for clean output
    if result == int(result):
        result = int(result)
    else:
        result = round(result, 6)

    target = json.dumps({"result": result}, separators=(",", ":"))

    # Format prompt
    op_descriptions = interpreter.describe_operators(operator_names)
    prompt = _format_prompt(program, inputs, op_descriptions)

    # Metadata
    op_counts: dict[str, int] = {}
    for assign in program.assignments:
        op = assign.op_call.op_name
        op_counts[op] = op_counts.get(op, 0) + 1

    metadata: dict[str, Any] = {
        "difficulty": spec.difficulty.level,
        "program_length": len(program.assignments),
        "operators": sorted(set(a.op_call.op_name for a in program.assignments)),
        "operator_counts": op_counts,
        "drift_type": spec.drift.drift_type.value if spec.drift else None,
        "has_semantic_override": bool(semantic_overrides),
        "num_inputs": num_inputs,
    }

    uid = instance_uid(spec.suite, FamilyName.DSL_EXEC.value, spec.stage, split, index, seed)

    return Instance(
        uid=uid,
        suite=spec.suite,
        stream_id=stream_id,
        family=FamilyName.DSL_EXEC,
        spec_id=spec.spec_id,
        stage=spec.stage,
        split=Split(split),
        prompt=prompt,
        target=target,
        seed=seed,
        metadata=metadata,
    )


def _generate_program(
    rng: random.Random,
    operator_names: list[str],
    input_names: list[str],
    program_length: int,
    nesting_depth: int,
    active_ops: dict,
) -> Program:
    """Generate a random valid DSL program."""
    available_vars = list(input_names)
    assignments: list[Assignment] = []

    for i in range(program_length):
        var_name = f"z{i + 1}" if i < program_length - 1 else "out"

        # Pick an operator
        op_name = rng.choice(operator_names)
        while op_name not in active_ops:
            op_name = rng.choice(operator_names)

        _, arity = active_ops[op_name]

        # Pick arguments from available variables (and sometimes literals)
        args: list[str | int | float] = []
        for _ in range(arity):
            if rng.random() < 0.2 and arity <= 2:
                # Use a literal value sometimes
                args.append(rng.randint(1, 10))
            else:
                args.append(rng.choice(available_vars))

        assignments.append(Assignment(
            var_name=var_name,
            op_call=OpCall(op_name=op_name, args=args),
        ))
        available_vars.append(var_name)

    return Program(
        assignments=assignments,
        output_var="out",
    )


def _format_prompt(
    program: Program,
    inputs: dict[str, int],
    op_descriptions: list[str],
) -> str:
    """Format a DSL execution prompt."""
    lines = ["Task: Execute the following program according to the DSL rules.", ""]
    lines.append("Rules:")
    for desc in op_descriptions:
        lines.append(desc)
    lines.append("")

    lines.append("Program:")
    for assign in program.assignments:
        op = assign.op_call
        args_str = ", ".join(str(a) for a in op.args)
        lines.append(f"{assign.var_name} = {op.op_name}({args_str})")
    lines.append("")

    lines.append("Input:")
    for name, val in sorted(inputs.items()):
        lines.append(f"{name} = {val}")
    lines.append("")

    lines.append(
        f'Return the final value of {program.output_var} as JSON '
        f'with key "result". Example: {{"result": 42}}'
    )

    return "\n".join(lines)

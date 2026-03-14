"""DSL interpreter: AST representation and execution engine.

The DSL supports a set of operators that take numeric arguments and produce
numeric results. Programs are sequences of variable assignments using these
operators. The interpreter executes programs given input variable bindings
and returns the final result.

Operators can have their semantics overridden per-stage to implement concept drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OpCall:
    """An operator invocation: op_name(args...)"""

    op_name: str
    args: list[str | int | float]  # variable names or literal values


@dataclass(frozen=True)
class Assignment:
    """A variable assignment: var_name = op_call"""

    var_name: str
    op_call: OpCall


@dataclass
class Program:
    """A DSL program: a sequence of assignments and a designated output variable."""

    assignments: list[Assignment]
    output_var: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default operator implementations
# ---------------------------------------------------------------------------

def _op_add(x: float, y: float) -> float:
    return x + y

def _op_sub(x: float, y: float) -> float:
    return x - y

def _op_mul(x: float, y: float) -> float:
    return x * y

def _op_neg(x: float) -> float:
    return -x

def _op_clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))

def _op_max(x: float, y: float) -> float:
    return max(x, y)

def _op_min(x: float, y: float) -> float:
    return min(x, y)

def _op_mod(x: float, y: float) -> float:
    if y == 0:
        return 0.0
    return x % y

def _op_abs(x: float) -> float:
    return abs(x)


# Operator registry: name -> (function, arity)
DEFAULT_OPERATORS: dict[str, tuple[Callable[..., float], int]] = {
    "ADD": (_op_add, 2),
    "SUB": (_op_sub, 2),
    "MUL": (_op_mul, 2),
    "NEG": (_op_neg, 1),
    "CLIP": (_op_clip, 3),
    "MAX": (_op_max, 2),
    "MIN": (_op_min, 2),
    "MOD": (_op_mod, 2),
    "ABS": (_op_abs, 1),
}


# ---------------------------------------------------------------------------
# Semantic override factory for concept drift
# ---------------------------------------------------------------------------

def make_semantic_override(description: str) -> Callable[..., float]:
    """Create an operator function from a drift description.

    Supported drift descriptions:
        "x + y + 1"   -> ADD with offset
        "x + y + 2"   -> ADD with larger offset
        "x * y + 1"   -> MUL with offset
    """
    # Parse simple arithmetic drift patterns
    desc = description.strip()

    if desc == "x + y + 1":
        return lambda x, y: x + y + 1
    elif desc == "x + y + 2":
        return lambda x, y: x + y + 2
    elif desc == "x * y + 1":
        return lambda x, y: x * y + 1
    elif desc == "x - y + 1":
        return lambda x, y: x - y + 1
    elif desc == "x * y - 1":
        return lambda x, y: x * y - 1
    else:
        raise ValueError(f"Unknown semantic override: {desc}")


# ---------------------------------------------------------------------------
# Interpreter
# ---------------------------------------------------------------------------

class DSLInterpreter:
    """Execute DSL programs with configurable operator semantics."""

    def __init__(
        self,
        operators: dict[str, tuple[Callable[..., float], int]] | None = None,
        semantic_overrides: dict[str, str] | None = None,
    ):
        """Initialize the interpreter.

        Args:
            operators: Operator name -> (function, arity). Defaults to DEFAULT_OPERATORS.
            semantic_overrides: Operator name -> drift description string.
                Overrides the default implementation for concept drift.
        """
        self.operators = dict(operators or DEFAULT_OPERATORS)

        # Apply semantic overrides (concept drift)
        if semantic_overrides:
            for op_name, desc in semantic_overrides.items():
                if op_name not in self.operators:
                    raise ValueError(
                        f"Cannot override unknown operator '{op_name}'. "
                        f"Known: {sorted(self.operators)}"
                    )
                original_arity = self.operators[op_name][1]
                override_fn = make_semantic_override(desc)
                self.operators[op_name] = (override_fn, original_arity)

    def execute(self, program: Program, inputs: dict[str, float]) -> float:
        """Execute a DSL program with given input bindings.

        Args:
            program: The program to execute.
            inputs: Variable name -> value bindings for input variables.

        Returns:
            The numeric value of the output variable.

        Raises:
            KeyError: If a referenced variable is undefined.
            ValueError: If an operator gets wrong number of arguments.
        """
        env: dict[str, float] = dict(inputs)

        for assign in program.assignments:
            op_name = assign.op_call.op_name
            if op_name not in self.operators:
                raise ValueError(f"Unknown operator: {op_name}")

            func, expected_arity = self.operators[op_name]

            # Resolve arguments: variable names to values, literals stay as-is
            resolved_args: list[float] = []
            for arg in assign.op_call.args:
                if isinstance(arg, str):
                    if arg not in env:
                        raise KeyError(
                            f"Undefined variable '{arg}' in assignment to '{assign.var_name}'"
                        )
                    resolved_args.append(float(env[arg]))
                else:
                    resolved_args.append(float(arg))

            if len(resolved_args) != expected_arity:
                raise ValueError(
                    f"Operator {op_name} expects {expected_arity} args, "
                    f"got {len(resolved_args)}"
                )

            env[assign.var_name] = func(*resolved_args)

        if program.output_var not in env:
            raise KeyError(f"Output variable '{program.output_var}' not found in environment")

        return env[program.output_var]

    def get_operator_names(self) -> list[str]:
        """Return sorted list of available operator names."""
        return sorted(self.operators.keys())

    def describe_operators(self, operator_names: list[str] | None = None) -> list[str]:
        """Generate human-readable operator descriptions for prompts.

        Args:
            operator_names: Which operators to describe (None = all active).

        Returns:
            List of description strings like "ADD(x, y) returns x + y".
        """
        names = operator_names or self.get_operator_names()
        descriptions: list[str] = []

        for name in sorted(names):
            if name not in self.operators:
                continue
            _, arity = self.operators[name]
            arg_names = ["x", "y", "z"][:arity]
            sig = f"{name}({', '.join(arg_names)})"

            # Generate description from actual function behavior using test values
            test_vals = [3.0, 7.0, 10.0][:arity]
            try:
                result = self.operators[name][0](*test_vals)
                # Infer a readable description
                desc = _infer_description(name, arg_names, test_vals, result)
                descriptions.append(f"- {sig} returns {desc}")
            except Exception:
                descriptions.append(f"- {sig}")

        return descriptions


def _infer_description(
    name: str, arg_names: list[str], test_vals: list[float], result: float
) -> str:
    """Infer a human-readable description from operator behavior."""
    x, y, z = (test_vals + [0, 0, 0])[:3]

    # Try common patterns
    if len(arg_names) == 1:
        if result == -x:
            return f"-{arg_names[0]}"
        if result == abs(x):
            return f"the absolute value of {arg_names[0]}"
        return f"{result} (for {arg_names[0]}={x})"

    if len(arg_names) == 2:
        if result == x + y:
            return f"{arg_names[0]} + {arg_names[1]}"
        if result == x + y + 1:
            return f"{arg_names[0]} + {arg_names[1]} + 1"
        if result == x + y + 2:
            return f"{arg_names[0]} + {arg_names[1]} + 2"
        if result == x - y:
            return f"{arg_names[0]} - {arg_names[1]}"
        if result == x * y:
            return f"{arg_names[0]} * {arg_names[1]}"
        if result == x * y + 1:
            return f"{arg_names[0]} * {arg_names[1]} + 1"
        if result == max(x, y):
            return f"the maximum of {arg_names[0]} and {arg_names[1]}"
        if result == min(x, y):
            return f"the minimum of {arg_names[0]} and {arg_names[1]}"
        if y != 0 and result == x % y:
            return f"{arg_names[0]} mod {arg_names[1]}"

    if len(arg_names) == 3 and result == max(y, min(x, z)):
        return f"{arg_names[0]} clipped to [{arg_names[1]}, {arg_names[2]}]"

    return f"(see examples)"

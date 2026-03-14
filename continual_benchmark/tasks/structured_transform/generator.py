"""Structured transformation task generator.

Generates random JSON objects and compositional transformation rule chains.
A reference solver applies the rules to produce the target output.
"""

from __future__ import annotations

import json
import random
from typing import Any

from continual_benchmark.core.constants import FamilyName, Split
from continual_benchmark.core.hashing import instance_uid
from continual_benchmark.core.schemas import Instance, Spec


# ---------------------------------------------------------------------------
# Rule types and their implementations
# ---------------------------------------------------------------------------

class TransformRule:
    """Base for transformation rules."""

    def describe(self) -> str:
        raise NotImplementedError

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RenameFieldRule(TransformRule):
    def __init__(self, old_name: str, new_name: str):
        self.old_name = old_name
        self.new_name = new_name

    def describe(self) -> str:
        return f'Rename field "{self.old_name}" to "{self.new_name}"'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        if self.old_name in result:
            result[self.new_name] = result.pop(self.old_name)
        return result


class RemoveFieldRule(TransformRule):
    def __init__(self, field_name: str):
        self.field_name = field_name

    def describe(self) -> str:
        return f'Remove the field "{self.field_name}"'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        result.pop(self.field_name, None)
        return result


class AddComputedFieldRule(TransformRule):
    """Adds a new field computed from existing fields."""

    def __init__(self, new_field: str, source_fields: list[str], operation: str):
        self.new_field = new_field
        self.source_fields = source_fields
        self.operation = operation  # "concat", "sum", "count", "join"

    def describe(self) -> str:
        sources = ", ".join(f'"{f}"' for f in self.source_fields)
        if self.operation == "concat":
            return f'Create a new field "{self.new_field}" by concatenating {sources} with a space'
        elif self.operation == "sum":
            return f'Create a new field "{self.new_field}" as the sum of {sources}'
        elif self.operation == "count":
            return f'Create a new field "{self.new_field}" as the count of items in {sources}'
        elif self.operation == "join":
            return f'Create a new field "{self.new_field}" by joining {sources} with ", "'
        return f'Create "{self.new_field}" from {sources} using {self.operation}'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        vals = [result.get(f) for f in self.source_fields]
        if self.operation == "concat":
            result[self.new_field] = " ".join(str(v) for v in vals if v is not None)
        elif self.operation == "sum":
            result[self.new_field] = sum(v for v in vals if isinstance(v, (int, float)))
        elif self.operation == "count":
            val = vals[0] if vals else []
            result[self.new_field] = len(val) if isinstance(val, list) else 0
        elif self.operation == "join":
            items = vals[0] if isinstance(vals[0], list) else vals
            result[self.new_field] = ", ".join(str(v) for v in items if v is not None)
        return result


class FilterListRule(TransformRule):
    """Filters a list field based on a condition."""

    def __init__(self, field_name: str, operation: str, threshold: int | float):
        self.field_name = field_name
        self.operation = operation  # "gte", "lte", "gt", "lt"
        self.threshold = threshold

    def describe(self) -> str:
        op_map = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<"}
        op_str = op_map.get(self.operation, self.operation)
        return f'In field "{self.field_name}", keep only values {op_str} {self.threshold}'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        if self.field_name in result and isinstance(result[self.field_name], list):
            ops = {
                "gte": lambda x: x >= self.threshold,
                "lte": lambda x: x <= self.threshold,
                "gt": lambda x: x > self.threshold,
                "lt": lambda x: x < self.threshold,
            }
            fn = ops.get(self.operation, lambda x: True)
            result[self.field_name] = [
                v for v in result[self.field_name]
                if isinstance(v, (int, float)) and fn(v)
            ]
        return result


class FlattenNestedRule(TransformRule):
    """Flattens a nested dict into the parent with prefixed keys."""

    def __init__(self, field_name: str, prefix: str = ""):
        self.field_name = field_name
        self.prefix = prefix

    def describe(self) -> str:
        if self.prefix:
            return (
                f'Flatten the nested object "{self.field_name}" into the parent, '
                f'prefixing keys with "{self.prefix}_"'
            )
        return f'Flatten the nested object "{self.field_name}" into the parent object'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        if self.field_name in result and isinstance(result[self.field_name], dict):
            nested = result.pop(self.field_name)
            for k, v in nested.items():
                key = f"{self.prefix}_{k}" if self.prefix else k
                result[key] = v
        return result


class ConditionalTransformRule(TransformRule):
    """Applies a transformation conditionally."""

    def __init__(
        self,
        condition_field: str,
        condition_op: str,
        condition_value: Any,
        action_field: str,
        action: str,
        action_value: Any = None,
    ):
        self.condition_field = condition_field
        self.condition_op = condition_op  # "eq", "gt", "lt", "contains"
        self.condition_value = condition_value
        self.action_field = action_field
        self.action = action  # "set", "remove", "multiply"
        self.action_value = action_value

    def describe(self) -> str:
        op_map = {"eq": "equals", "gt": "is greater than", "lt": "is less than", "contains": "contains"}
        op_str = op_map.get(self.condition_op, self.condition_op)
        act_map = {"set": f'set "{self.action_field}" to {self.action_value}',
                    "remove": f'remove "{self.action_field}"',
                    "multiply": f'multiply "{self.action_field}" by {self.action_value}'}
        act_str = act_map.get(self.action, self.action)
        return f'If "{self.condition_field}" {op_str} {self.condition_value}, then {act_str}'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        val = result.get(self.condition_field)
        condition_met = False

        if self.condition_op == "eq":
            condition_met = val == self.condition_value
        elif self.condition_op == "gt" and isinstance(val, (int, float)):
            condition_met = val > self.condition_value
        elif self.condition_op == "lt" and isinstance(val, (int, float)):
            condition_met = val < self.condition_value
        elif self.condition_op == "contains" and isinstance(val, (str, list)):
            condition_met = self.condition_value in val

        if condition_met:
            if self.action == "set":
                result[self.action_field] = self.action_value
            elif self.action == "remove":
                result.pop(self.action_field, None)
            elif self.action == "multiply" and isinstance(result.get(self.action_field), (int, float)):
                result[self.action_field] = result[self.action_field] * self.action_value

        return result


class AggregateValuesRule(TransformRule):
    """Aggregates numeric values from a list field."""

    def __init__(self, field_name: str, operation: str, target_field: str):
        self.field_name = field_name
        self.operation = operation  # "sum", "avg", "max", "min"
        self.target_field = target_field

    def describe(self) -> str:
        return f'Create "{self.target_field}" as the {self.operation} of values in "{self.field_name}"'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        vals = result.get(self.field_name, [])
        nums = [v for v in vals if isinstance(v, (int, float))] if isinstance(vals, list) else []

        if nums:
            if self.operation == "sum":
                result[self.target_field] = sum(nums)
            elif self.operation == "avg":
                result[self.target_field] = round(sum(nums) / len(nums), 2)
            elif self.operation == "max":
                result[self.target_field] = max(nums)
            elif self.operation == "min":
                result[self.target_field] = min(nums)
        else:
            result[self.target_field] = 0

        return result


class TypeCastRule(TransformRule):
    """Casts a field to a different type."""

    def __init__(self, field_name: str, target_type: str):
        self.field_name = field_name
        self.target_type = target_type  # "string", "int", "float"

    def describe(self) -> str:
        return f'Convert field "{self.field_name}" to {self.target_type}'

    def apply(self, obj: dict[str, Any]) -> dict[str, Any]:
        result = dict(obj)
        if self.field_name in result:
            val = result[self.field_name]
            try:
                if self.target_type == "string":
                    result[self.field_name] = str(val)
                elif self.target_type == "int":
                    result[self.field_name] = int(val)
                elif self.target_type == "float":
                    result[self.field_name] = float(val)
            except (ValueError, TypeError):
                pass
        return result


# ---------------------------------------------------------------------------
# Input object generation
# ---------------------------------------------------------------------------

# Name pools for realistic-looking data
FIRST_NAMES = ["Ada", "Bob", "Cara", "Dan", "Eva", "Frank", "Grace", "Hank", "Iris", "Jack"]
LAST_NAMES = ["Smith", "Chen", "Patel", "Lopez", "Kim", "Brown", "Davis", "Wilson", "Moore", "Taylor"]
DEPARTMENTS = ["Research", "Sales", "Engineering", "Marketing", "Finance", "HR", "Legal", "Support"]
CITIES = ["London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto", "Mumbai", "Lagos"]
TAGS = ["urgent", "low", "medium", "high", "critical", "review", "approved", "pending"]


def _generate_input_object(rng: random.Random, schema_depth: int) -> dict[str, Any]:
    """Generate a random JSON object with realistic structure."""
    obj: dict[str, Any] = {}

    # Always include some string fields
    fname = rng.choice(FIRST_NAMES)
    lname = rng.choice(LAST_NAMES)
    obj["fname"] = fname
    obj["lname"] = lname

    # Numeric fields
    obj["age"] = rng.randint(20, 65)
    obj["salary"] = rng.randint(30, 150) * 1000

    # List field
    obj["scores"] = [rng.randint(10, 100) for _ in range(rng.randint(3, 7))]

    # Optional string field (target for removal)
    obj["temp"] = rng.choice(["remove_me", "discard", "temporary"])

    # Location
    obj["city"] = rng.choice(CITIES)
    obj["department"] = rng.choice(DEPARTMENTS)

    # Tags list
    num_tags = rng.randint(1, 4)
    obj["tags"] = rng.sample(TAGS, min(num_tags, len(TAGS)))

    # Nested objects at higher depths
    if schema_depth >= 2:
        obj["address"] = {
            "street": f"{rng.randint(1, 999)} Main St",
            "city": rng.choice(CITIES),
            "zip": str(rng.randint(10000, 99999)),
        }

    if schema_depth >= 3:
        obj["contact"] = {
            "email": f"{fname.lower()}.{lname.lower()}@example.com",
            "phone": f"+1-{rng.randint(200,999)}-{rng.randint(100,999)}-{rng.randint(1000,9999)}",
        }
        obj["metrics"] = {
            "performance": round(rng.uniform(1.0, 5.0), 1),
            "attendance": rng.randint(80, 100),
        }

    return obj


# ---------------------------------------------------------------------------
# Rule generation
# ---------------------------------------------------------------------------

RULE_GENERATORS: dict[str, Any] = {}


def _generate_rules(
    rng: random.Random,
    rule_types: list[str],
    input_obj: dict[str, Any],
    rule_count: int,
) -> list[TransformRule]:
    """Generate a list of applicable transformation rules."""
    rules: list[TransformRule] = []
    used_fields: set[str] = set()

    for _ in range(rule_count):
        rule_type = rng.choice(rule_types)
        rule = _generate_single_rule(rng, rule_type, input_obj, used_fields)
        if rule:
            rules.append(rule)

    return rules


def _generate_single_rule(
    rng: random.Random,
    rule_type: str,
    input_obj: dict[str, Any],
    used_fields: set[str],
) -> TransformRule | None:
    """Generate a single transformation rule of the given type."""
    string_fields = [
        k for k, v in input_obj.items()
        if isinstance(v, str) and k not in used_fields
    ]
    numeric_fields = [
        k for k, v in input_obj.items()
        if isinstance(v, (int, float)) and k not in used_fields
    ]
    list_fields = [
        k for k, v in input_obj.items()
        if isinstance(v, list) and k not in used_fields
    ]
    dict_fields = [
        k for k, v in input_obj.items()
        if isinstance(v, dict) and k not in used_fields
    ]

    if rule_type == "rename_field" and string_fields:
        field = rng.choice(string_fields)
        new_name = field + "_renamed"
        used_fields.add(field)
        return RenameFieldRule(field, new_name)

    elif rule_type == "remove_field":
        removable = [k for k in input_obj if k not in used_fields and k in ("temp", "tags")]
        if removable:
            field = rng.choice(removable)
            used_fields.add(field)
            return RemoveFieldRule(field)

    elif rule_type == "add_computed_field" and len(string_fields) >= 2:
        sources = rng.sample(string_fields, 2)
        new_field = "full_name" if "fname" in sources else f"{'_'.join(sources)}_combined"
        used_fields.add(new_field)
        return AddComputedFieldRule(new_field, sources, "concat")

    elif rule_type == "filter_list" and list_fields:
        field = rng.choice(list_fields)
        vals = input_obj[field]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            threshold = sorted(vals)[len(vals) // 2]  # median-ish
            op = rng.choice(["gte", "lte"])
            used_fields.add(field)
            return FilterListRule(field, op, threshold)

    elif rule_type == "flatten_nested" and dict_fields:
        field = rng.choice(dict_fields)
        used_fields.add(field)
        prefix = field[:3]
        return FlattenNestedRule(field, prefix)

    elif rule_type == "conditional_transform" and numeric_fields:
        cond_field = rng.choice(numeric_fields)
        cond_val = input_obj[cond_field]
        threshold = cond_val - rng.randint(1, 5)  # Ensure condition is likely true
        action_field = rng.choice([f for f in numeric_fields if f != cond_field] or [cond_field])
        return ConditionalTransformRule(
            condition_field=cond_field,
            condition_op="gt",
            condition_value=threshold,
            action_field=action_field,
            action="multiply",
            action_value=2,
        )

    elif rule_type == "aggregate_values" and list_fields:
        field = rng.choice(list_fields)
        vals = input_obj[field]
        if vals and all(isinstance(v, (int, float)) for v in vals):
            op = rng.choice(["sum", "avg", "max", "min"])
            target = f"{field}_{op}"
            return AggregateValuesRule(field, op, target)

    elif rule_type == "type_cast" and numeric_fields:
        field = rng.choice(numeric_fields)
        used_fields.add(field)
        return TypeCastRule(field, "string")

    return None


# ---------------------------------------------------------------------------
# Reference solver
# ---------------------------------------------------------------------------

def apply_rules(obj: dict[str, Any], rules: list[TransformRule]) -> dict[str, Any]:
    """Apply transformation rules sequentially to produce the target output."""
    result = dict(obj)
    for rule in rules:
        result = rule.apply(result)
    return result


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_transform_instance(
    spec: Spec,
    stream_id: str,
    index: int,
    split: str,
    seed: int,
    rng: random.Random,
) -> Instance:
    """Generate a single structured transformation instance."""
    config = spec.generator_config
    rule_types = config.get("rule_types", ["rename_field", "remove_field", "add_computed_field"])

    schema_depth = spec.difficulty.schema_depth or 1
    rule_count = spec.difficulty.rule_count or 3

    # Generate input object
    input_obj = _generate_input_object(rng, schema_depth)

    # Generate rules
    rules = _generate_rules(rng, rule_types, input_obj, rule_count)

    # Apply rules to get target
    target_obj = apply_rules(input_obj, rules)
    target = json.dumps(target_obj, sort_keys=True, separators=(",", ":"))

    # Format prompt
    prompt = _format_transform_prompt(input_obj, rules)

    # Metadata
    metadata: dict[str, Any] = {
        "difficulty": spec.difficulty.level,
        "schema_depth": schema_depth,
        "rule_count": len(rules),
        "rule_types": [type(r).__name__ for r in rules],
        "drift_type": spec.drift.drift_type.value if spec.drift else None,
        "input_field_count": len(input_obj),
    }

    uid = instance_uid(
        spec.suite, FamilyName.STRUCTURED_TRANSFORM.value,
        spec.stage, split, index, seed,
    )

    return Instance(
        uid=uid,
        suite=spec.suite,
        stream_id=stream_id,
        family=FamilyName.STRUCTURED_TRANSFORM,
        spec_id=spec.spec_id,
        stage=spec.stage,
        split=Split(split),
        prompt=prompt,
        target=target,
        seed=seed,
        metadata=metadata,
    )


def _format_transform_prompt(
    input_obj: dict[str, Any],
    rules: list[TransformRule],
) -> str:
    """Format a structured transformation prompt."""
    lines = ["Task: Transform the input JSON according to the rules.", ""]
    lines.append("Rules:")
    for i, rule in enumerate(rules, 1):
        lines.append(f"{i}. {rule.describe()}")
    lines.append("")

    lines.append("Input:")
    lines.append(json.dumps(input_obj, indent=2))
    lines.append("")

    lines.append("Return valid JSON only. Use sorted keys in the output.")

    return "\n".join(lines)

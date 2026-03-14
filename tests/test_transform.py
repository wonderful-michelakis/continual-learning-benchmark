"""Tests for the structured transformation task family."""

import json
import random

from continual_benchmark.core.constants import FamilyName
from continual_benchmark.core.schemas import DifficultyConfig, Spec
from continual_benchmark.tasks.structured_transform.canonicalize import (
    canonicalize_json_output,
    compare_json_outputs,
)
from continual_benchmark.tasks.structured_transform.generator import (
    AddComputedFieldRule,
    FilterListRule,
    RemoveFieldRule,
    RenameFieldRule,
    apply_rules,
)


class TestTransformRules:
    def test_rename(self):
        rule = RenameFieldRule("fname", "first_name")
        result = rule.apply({"fname": "Ada", "lname": "Lovelace"})
        assert result == {"first_name": "Ada", "lname": "Lovelace"}

    def test_remove(self):
        rule = RemoveFieldRule("temp")
        result = rule.apply({"name": "Ada", "temp": "discard"})
        assert result == {"name": "Ada"}

    def test_filter_list(self):
        rule = FilterListRule("scores", "gte", 50)
        result = rule.apply({"scores": [40, 50, 88]})
        assert result == {"scores": [50, 88]}

    def test_computed_field_concat(self):
        rule = AddComputedFieldRule("full_name", ["fname", "lname"], "concat")
        result = rule.apply({"fname": "Ada", "lname": "Lovelace"})
        assert result["full_name"] == "Ada Lovelace"

    def test_rule_chain(self):
        rules = [
            RenameFieldRule("fname", "first_name"),
            RenameFieldRule("lname", "last_name"),
            RemoveFieldRule("temp"),
            FilterListRule("scores", "gte", 50),
        ]
        obj = {"fname": "Ada", "lname": "Lovelace", "temp": "x", "scores": [40, 50, 88]}
        result = apply_rules(obj, rules)
        assert "first_name" in result
        assert "temp" not in result
        assert result["scores"] == [50, 88]


class TestCanonicalization:
    def test_key_sorting(self):
        assert canonicalize_json_output('{"b":2,"a":1}') == '{"a":1,"b":2}'

    def test_numeric_normalization(self):
        assert canonicalize_json_output('{"x": 1.0}') == '{"x":1}'

    def test_comparison(self):
        ok, _ = compare_json_outputs('{"a":1,"b":2}', '{"b":2,"a":1}')
        assert ok is True


class TestGeneration:
    def test_generate_instance(self):
        from continual_benchmark.core.registry import get_family
        import continual_benchmark.tasks  # noqa: F401

        spec = Spec(
            spec_id="structured_transform:v1:stage04",
            family=FamilyName.STRUCTURED_TRANSFORM,
            stage=4,
            difficulty=DifficultyConfig(level=2, schema_depth=1, rule_count=3),
            generator_config={"rule_types": ["rename_field", "remove_field", "add_computed_field"]},
        )
        family = get_family("structured_transform")
        rng = random.Random(42)
        inst = family.generate_instance(spec, "test", 0, "train", 42, rng)

        assert inst.prompt
        assert inst.target
        # Verify self-check
        ok, score = family.verify(inst.target, inst.target, inst.metadata)
        assert ok

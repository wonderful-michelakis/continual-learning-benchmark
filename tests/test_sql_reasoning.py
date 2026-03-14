"""Tests for the SQL reasoning task family."""

import random

from continual_benchmark.core.constants import FamilyName
from continual_benchmark.core.schemas import DifficultyConfig, Spec
from continual_benchmark.tasks.sql_reasoning.canonicalize import (
    canonicalize_table_output,
    compare_table_outputs,
)
from continual_benchmark.tasks.sql_reasoning.db_builder import (
    DatabaseInstance,
    build_employees_schema,
)


class TestDBBuilder:
    def test_build_basic_schema(self):
        rng = random.Random(42)
        schema, data = build_employees_schema(rng, num_tables=2)
        assert len(schema.tables) == 2
        assert "departments" in data
        assert "employees" in data
        assert len(data["employees"]) >= 5

    def test_database_query(self):
        rng = random.Random(42)
        schema, data = build_employees_schema(rng, num_tables=2)
        db = DatabaseInstance(schema=schema, data=data)
        result = db.execute_query("SELECT COUNT(*) as cnt FROM employees")
        assert result[0]["cnt"] == len(data["employees"])
        db.close()

    def test_schema_v2(self):
        rng = random.Random(42)
        schema, data = build_employees_schema(rng, num_tables=2, schema_variant="employees_v2")
        assert any(c.name == "full_name" for t in schema.tables for c in t.columns if t.name == "employees")


class TestCanonicalization:
    def test_order_insensitive(self):
        ok, _ = compare_table_outputs('["Alice","Bob"]', '["Bob","Alice"]', order_matters=False)
        assert ok is True

    def test_order_sensitive(self):
        ok, _ = compare_table_outputs('["Alice","Bob"]', '["Bob","Alice"]', order_matters=True)
        assert ok is False

    def test_scalar_normalization(self):
        assert canonicalize_table_output("42.0") == "42"


class TestGeneration:
    def test_generate_instance(self):
        from continual_benchmark.core.registry import get_family
        import continual_benchmark.tasks  # noqa: F401

        spec = Spec(
            spec_id="sql_reasoning:v1:stage07",
            family=FamilyName.SQL_REASONING,
            stage=7,
            difficulty=DifficultyConfig(level=2, num_tables=2),
            generator_config={"schema_family": "employees", "query_types": ["select", "filter"]},
        )
        family = get_family("sql_reasoning")
        rng = random.Random(42)
        inst = family.generate_instance(spec, "test", 0, "train", 42, rng)

        assert inst.prompt
        assert inst.target
        ok, score = family.verify(inst.target, inst.target, inst.metadata)
        assert ok

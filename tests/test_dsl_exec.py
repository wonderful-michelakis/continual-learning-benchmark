"""Tests for the DSL execution task family."""

import random

from continual_benchmark.core.constants import FamilyName
from continual_benchmark.core.schemas import DifficultyConfig, Spec
from continual_benchmark.tasks.dsl_exec.canonicalize import (
    canonicalize_dsl_output,
    compare_dsl_outputs,
)
from continual_benchmark.tasks.dsl_exec.interpreter import (
    Assignment,
    DSLInterpreter,
    OpCall,
    Program,
)


class TestInterpreter:
    def test_simple_add(self):
        interp = DSLInterpreter()
        prog = Program(
            assignments=[
                Assignment("out", OpCall("ADD", ["a", "b"])),
            ],
            output_var="out",
        )
        result = interp.execute(prog, {"a": 3.0, "b": 5.0})
        assert result == 8.0

    def test_chained_operations(self):
        interp = DSLInterpreter()
        prog = Program(
            assignments=[
                Assignment("z1", OpCall("ADD", ["a", "b"])),
                Assignment("z2", OpCall("MUL", ["z1", "c"])),
                Assignment("out", OpCall("CLIP", ["z2", 0, 20])),
            ],
            output_var="out",
        )
        result = interp.execute(prog, {"a": 2.0, "b": 3.0, "c": 5.0})
        assert result == 20.0  # (2+3)*5=25, clipped to 20

    def test_semantic_override(self):
        """Test concept drift: ADD returns x + y + 1."""
        interp = DSLInterpreter(semantic_overrides={"ADD": "x + y + 1"})
        prog = Program(
            assignments=[
                Assignment("out", OpCall("ADD", ["a", "b"])),
            ],
            output_var="out",
        )
        result = interp.execute(prog, {"a": 3.0, "b": 5.0})
        assert result == 9.0  # 3 + 5 + 1

    def test_neg_operator(self):
        interp = DSLInterpreter()
        prog = Program(
            assignments=[Assignment("out", OpCall("NEG", ["a"]))],
            output_var="out",
        )
        assert interp.execute(prog, {"a": 7.0}) == -7.0

    def test_literal_args(self):
        interp = DSLInterpreter()
        prog = Program(
            assignments=[Assignment("out", OpCall("ADD", [3, 4]))],
            output_var="out",
        )
        assert interp.execute(prog, {}) == 7.0


class TestCanonicalization:
    def test_json_normalization(self):
        assert canonicalize_dsl_output('{"result": 42}') == '{"result":42}'
        assert canonicalize_dsl_output('{"result":42.0}') == '{"result":42}'

    def test_comparison(self):
        ok, score = compare_dsl_outputs('{"result": 42}', '{"result":42.0}')
        assert ok is True
        assert score == 1.0

    def test_wrong_answer(self):
        ok, score = compare_dsl_outputs('{"result": 41}', '{"result": 42}')
        assert ok is False
        assert score == 0.0


class TestGeneration:
    def test_generate_instance(self):
        from continual_benchmark.core.registry import get_family
        import continual_benchmark.tasks  # noqa: F401

        spec = Spec(
            spec_id="dsl_exec:v1:stage01",
            family=FamilyName.DSL_EXEC,
            stage=1,
            difficulty=DifficultyConfig(level=2, program_length=3),
            generator_config={"operators": ["ADD", "SUB", "MUL"]},
        )
        family = get_family("dsl_exec")
        rng = random.Random(42)
        inst = family.generate_instance(spec, "test", 0, "train", 42, rng)

        assert inst.uid
        assert inst.prompt
        assert inst.target
        # Verify the target is correct
        ok, score = family.verify(inst.target, inst.target, inst.metadata)
        assert ok

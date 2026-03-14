"""Integration tests for the build pipeline."""

import json
from pathlib import Path

import pytest

from continual_benchmark.build.build_suite import build
from continual_benchmark.build.qa_suite import run_qa
from continual_benchmark.core.io import read_instances, read_json


@pytest.fixture
def mini_suite(tmp_path):
    """Build a mini 2-stage suite for testing."""
    # Create a minimal stream definition
    stream_dir = tmp_path / "streams"
    stream_dir.mkdir()
    stream_def = {
        "stream_id": "mini_test",
        "suite": "test",
        "description": "Minimal test stream",
        "stages": [
            {
                "stage": 1,
                "spec_id": "dsl_exec:test:s1",
                "family": "dsl_exec",
                "difficulty": {"level": 2, "program_length": 3, "operator_count": 3},
                "split_sizes": {"train": 10, "dev": 5, "public_test": 5, "private_test": 5},
                "generator_config": {"operators": ["ADD", "SUB", "MUL"]},
            },
            {
                "stage": 2,
                "spec_id": "structured_transform:test:s2",
                "family": "structured_transform",
                "difficulty": {"level": 2, "schema_depth": 1, "rule_count": 3},
                "split_sizes": {"train": 10, "dev": 5, "public_test": 5, "private_test": 5},
                "generator_config": {"rule_types": ["rename_field", "remove_field"]},
            },
        ],
    }
    import yaml
    with open(stream_dir / "mini_test.yaml", "w") as f:
        yaml.dump(stream_def, f)

    out = tmp_path / "suite"
    build(suite="test", stream_id="mini_test", output_dir=out, global_seed=42, stream_dir=stream_dir)
    return out


class TestBuildPipeline:
    def test_build_creates_files(self, mini_suite):
        assert (mini_suite / "manifest.json").exists()
        assert (mini_suite / "stream.json").exists()
        assert (mini_suite / "stage_001" / "train.jsonl").exists()
        assert (mini_suite / "stage_002" / "train.jsonl").exists()

    def test_manifest_content(self, mini_suite):
        manifest = read_json(mini_suite / "manifest.json")
        assert manifest["suite"] == "test"
        assert manifest["stage_count"] == 2
        assert "dsl_exec" in manifest["families"]

    def test_instances_valid(self, mini_suite):
        instances = read_instances(mini_suite / "stage_001" / "train.jsonl")
        assert len(instances) == 10
        for inst in instances:
            assert inst.family.value == "dsl_exec"
            assert inst.stage == 1
            assert inst.prompt
            assert inst.target

    def test_qa_passes(self, mini_suite):
        report = run_qa(mini_suite)
        # Allow some warnings but no critical failures on basic checks
        # Exclude diversity/duplicate checks which can fail on tiny suites
        failed = [
            c for c in report.checks
            if not c.passed
            and "duplicate" not in c.name
            and "diversity" not in c.name
        ]
        assert len(failed) == 0, f"QA failures: {[c.name for c in failed]}"


class TestReproducibility:
    def test_deterministic_generation(self, tmp_path):
        """Building the same suite twice produces identical output."""
        stream_dir = tmp_path / "streams"
        stream_dir.mkdir()
        stream_def = {
            "stream_id": "repro_test",
            "suite": "test",
            "stages": [{
                "stage": 1,
                "spec_id": "dsl_exec:test:s1",
                "family": "dsl_exec",
                "difficulty": {"level": 2, "program_length": 3},
                "split_sizes": {"train": 5, "dev": 2, "public_test": 2, "private_test": 2},
                "generator_config": {"operators": ["ADD", "SUB"]},
            }],
        }
        import yaml
        with open(stream_dir / "repro_test.yaml", "w") as f:
            yaml.dump(stream_def, f)

        out1 = tmp_path / "suite1"
        out2 = tmp_path / "suite2"
        build(suite="test", stream_id="repro_test", output_dir=out1, global_seed=42, stream_dir=stream_dir)
        build(suite="test", stream_id="repro_test", output_dir=out2, global_seed=42, stream_dir=stream_dir)

        # Compare train.jsonl from both builds
        insts1 = read_instances(out1 / "stage_001" / "train.jsonl")
        insts2 = read_instances(out2 / "stage_001" / "train.jsonl")

        assert len(insts1) == len(insts2)
        for a, b in zip(insts1, insts2):
            assert a.uid == b.uid
            assert a.prompt == b.prompt
            assert a.target == b.target

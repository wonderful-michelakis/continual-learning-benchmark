"""Tests for stream loading and materialization."""

from continual_benchmark.streams.builder import materialize_specs
from continual_benchmark.streams.loader import list_available_streams, load_stream
from continual_benchmark.streams.similarity import (
    compute_similarity_matrix,
    summarize_stream_structure,
)


class TestStreamLoader:
    def test_list_streams(self):
        streams = list_available_streams()
        assert "v1_clustered" in streams
        assert "v1_interleaved" in streams
        assert "v1_drift" in streams

    def test_load_clustered(self):
        stream = load_stream("v1_clustered")
        assert stream.stream_id == "v1_clustered"
        assert len(stream.stages) == 12

    def test_load_interleaved(self):
        stream = load_stream("v1_interleaved")
        assert len(stream.stages) == 12

    def test_load_drift(self):
        stream = load_stream("v1_drift")
        assert len(stream.stages) == 16


class TestStreamBuilder:
    def test_materialize_specs(self):
        stream = load_stream("v1_clustered")
        specs = materialize_specs(stream)
        assert len(specs) == 12
        assert specs[0].spec_id == "dsl_exec:v1:stage01"
        assert specs[0].family.value == "dsl_exec"


class TestSimilarity:
    def test_similarity_matrix(self):
        stream = load_stream("v1_clustered")
        specs = materialize_specs(stream)
        matrix = compute_similarity_matrix(specs)
        assert len(matrix) == 12
        # Diagonal should be 1.0
        for i in range(12):
            assert matrix[i][i] == 1.0

    def test_stream_structure_summary(self):
        stream = load_stream("v1_clustered")
        summary = summarize_stream_structure(stream)
        assert summary["total_stages"] == 12
        assert "dsl_exec" in summary["family_distribution"]

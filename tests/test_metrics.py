"""Tests for continual learning metrics."""

from continual_benchmark.core.schemas import PerformanceMatrix
from continual_benchmark.eval.metrics import compute_cl_metrics


class TestMetrics:
    def _make_matrix(self):
        """Create a sample performance matrix for testing.

        Scenario: 3 stages, model degrades on earlier tasks.
        """
        return PerformanceMatrix(
            training_stages=[1, 2, 3],
            eval_stages=[1, 2, 3],
            matrix=[
                [0.90, None, None],  # After stage 1: only task 1 seen
                [0.70, 0.85, None],  # After stage 2: task 1 dropped, task 2 learned
                [0.60, 0.75, 0.80],  # After stage 3: further degradation
            ],
            families={1: "dsl_exec", 2: "structured_transform", 3: "sql_reasoning"},
        )

    def test_average_accuracy(self):
        pm = self._make_matrix()
        report = compute_cl_metrics(pm)
        # After stage 1: only task 1 = 0.90
        assert abs(report.average_accuracy[0] - 0.90) < 1e-6
        # After stage 2: tasks 1,2 = (0.70 + 0.85) / 2 = 0.775
        assert abs(report.average_accuracy[1] - 0.775) < 1e-6
        # After stage 3: tasks 1,2,3 = (0.60 + 0.75 + 0.80) / 3
        assert abs(report.average_accuracy[2] - 0.71667) < 1e-3

    def test_forgetting(self):
        pm = self._make_matrix()
        report = compute_cl_metrics(pm)
        # Task 1: max was 0.90, current is 0.60 -> forgetting = 0.30
        assert abs(report.forgetting[1] - 0.30) < 1e-6
        # Task 2: max was 0.85, current is 0.75 -> forgetting = 0.10
        assert abs(report.forgetting[2] - 0.10) < 1e-6
        # Task 3: never dropped (just learned), forgetting = 0
        assert report.forgetting[3] == 0.0

    def test_average_forgetting(self):
        pm = self._make_matrix()
        report = compute_cl_metrics(pm)
        expected_avg = (0.30 + 0.10 + 0.0) / 3
        assert abs(report.average_forgetting - expected_avg) < 1e-6

    def test_backward_transfer(self):
        pm = self._make_matrix()
        report = compute_cl_metrics(pm)
        # Task 1: perf after learning = 0.90, final = 0.60 -> BWT = -0.30
        assert abs(report.backward_transfer[1] - (-0.30)) < 1e-6
        # Task 2: perf after learning = 0.85, final = 0.75 -> BWT = -0.10
        assert abs(report.backward_transfer[2] - (-0.10)) < 1e-6

    def test_family_breakdown(self):
        pm = self._make_matrix()
        report = compute_cl_metrics(pm)
        assert "dsl_exec" in report.family_breakdown
        assert "structured_transform" in report.family_breakdown

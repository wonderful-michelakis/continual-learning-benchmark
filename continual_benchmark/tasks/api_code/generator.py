"""API code generation task generator.

Generates tasks where the model must write functions using a toy API.
The API changes across stages (concept drift via new/modified functions).
"""

from __future__ import annotations

import json
import random
import textwrap
from typing import Any

from continual_benchmark.core.constants import FamilyName, Split
from continual_benchmark.core.hashing import instance_uid
from continual_benchmark.core.schemas import Instance, Spec
from continual_benchmark.tasks.api_code.test_builder import (
    build_test_code,
    generate_test_cases,
)


# ---------------------------------------------------------------------------
# API definitions by version (lazy-loaded)
# ---------------------------------------------------------------------------

_API_DEFS_CACHE: dict[str, dict[str, Any]] | None = None


def _build_api_definitions() -> dict[str, dict[str, Any]]:
    """Build API definitions with proper code formatting."""
    return {
        "1.0": {
            "description": "Data Processing Library v1.0",
            "functions": {
                "normalize": {
                    "signature": "normalize(xs: list[float]) -> list[float]",
                    "description": "Returns a list scaled so the sum is 1.0. If sum is 0, returns zeros.",
                    "impl": textwrap.dedent("""\
                        def normalize(xs):
                            total = sum(xs)
                            if total == 0:
                                return [0.0] * len(xs)
                            return [x / total for x in xs]
                    """),
                },
                "top_k": {
                    "signature": "top_k(xs: list[float], k: int) -> list[float]",
                    "description": "Returns the k largest values in descending order.",
                    "impl": textwrap.dedent("""\
                        def top_k(xs, k):
                            return sorted(xs, reverse=True)[:k]
                    """),
                },
                "pairwise_sum": {
                    "signature": "pairwise_sum(xs: list[float]) -> list[float]",
                    "description": (
                        "Returns [xs[0]+xs[1], xs[2]+xs[3], ...]. "
                        "If odd length, last element is kept."
                    ),
                    "impl": textwrap.dedent("""\
                        def pairwise_sum(xs):
                            result = []
                            for i in range(0, len(xs) - 1, 2):
                                result.append(xs[i] + xs[i+1])
                            if len(xs) % 2 == 1:
                                result.append(xs[-1])
                            return result
                    """),
                },
                "running_mean": {
                    "signature": "running_mean(xs: list[float], window: int) -> list[float]",
                    "description": (
                        "Returns running mean with given window size. "
                        "For positions with fewer than 'window' elements, uses available elements."
                    ),
                    "impl": textwrap.dedent("""\
                        def running_mean(xs, window):
                            result = []
                            for i in range(len(xs)):
                                start = max(0, i - window + 1)
                                chunk = xs[start:i+1]
                                result.append(round(sum(chunk) / len(chunk), 6))
                            return result
                    """),
                },
                "cumulative_sum": {
                    "signature": "cumulative_sum(xs: list[float]) -> list[float]",
                    "description": "Returns cumulative sum: [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2], ...].",
                    "impl": textwrap.dedent("""\
                        def cumulative_sum(xs):
                            result = []
                            total = 0
                            for x in xs:
                                total += x
                                result.append(total)
                            return result
                    """),
                },
            },
            "tasks": [
                {
                    "function_name": "summarize",
                    "signature": "def summarize(xs: list[int], k: int) -> list[float]",
                    "instructions": (
                        "1. Normalize xs\n"
                        "2. Compute pairwise_sum on the normalized list\n"
                        "3. Return top_k of the result with parameter k"
                    ),
                    "spec": {"type": "summarize"},
                },
                {
                    "function_name": "smooth_and_rank",
                    "signature": "def smooth_and_rank(xs: list[int], window: int, k: int) -> list[float]",
                    "instructions": (
                        "1. Compute running_mean of xs with the given window size\n"
                        "2. Return top_k of the smoothed result with parameter k"
                    ),
                    "spec": {"type": "smooth_and_rank"},
                },
                {
                    "function_name": "cumulative_normalize",
                    "signature": "def cumulative_normalize(xs: list[int]) -> list[float]",
                    "instructions": (
                        "1. Compute cumulative_sum of xs\n"
                        "2. Normalize the cumulative sum\n"
                        "3. Return the result"
                    ),
                    "spec": {"type": "cumulative_normalize"},
                },
                {
                    "function_name": "paired_topk_smooth",
                    "signature": "def paired_topk_smooth(xs: list[int], k: int, window: int) -> list[float]",
                    "instructions": (
                        "1. Compute pairwise_sum of xs\n"
                        "2. Apply running_mean with the given window\n"
                        "3. Return top_k with parameter k"
                    ),
                    "spec": {"type": "paired_topk_smooth"},
                },
                {
                    "function_name": "normalize_and_accumulate",
                    "signature": "def normalize_and_accumulate(xs: list[int]) -> list[float]",
                    "instructions": (
                        "1. Normalize xs\n"
                        "2. Compute cumulative_sum on the normalized list\n"
                        "3. Return the result (each value rounded to 6 decimal places)"
                    ),
                    "spec": {"type": "normalize_and_accumulate"},
                },
            ],
        },
        "1.1": {
            "description": "Data Processing Library v1.1",
            "functions": {
                "normalize": {
                    "signature": "normalize(xs: list[float]) -> list[float]",
                    "description": "Returns a list scaled so the sum is 1.0. If sum is 0, returns zeros.",
                    "impl": textwrap.dedent("""\
                        def normalize(xs):
                            total = sum(xs)
                            if total == 0:
                                return [0.0] * len(xs)
                            return [x / total for x in xs]
                    """),
                },
                "top_k": {
                    "signature": "top_k(xs: list[float], k: int) -> list[float]",
                    "description": "Returns the k largest values in descending order.",
                    "impl": textwrap.dedent("""\
                        def top_k(xs, k):
                            return sorted(xs, reverse=True)[:k]
                    """),
                },
                "pairwise_sum": {
                    "signature": "pairwise_sum(xs: list[float]) -> list[float]",
                    "description": (
                        "Returns [xs[0]+xs[1], xs[2]+xs[3], ...]. "
                        "If odd length, last element is kept."
                    ),
                    "impl": textwrap.dedent("""\
                        def pairwise_sum(xs):
                            result = []
                            for i in range(0, len(xs) - 1, 2):
                                result.append(xs[i] + xs[i+1])
                            if len(xs) % 2 == 1:
                                result.append(xs[-1])
                            return result
                    """),
                },
                "running_mean": {
                    "signature": "running_mean(xs: list[float], window: int) -> list[float]",
                    "description": (
                        "Returns running mean with given window size. "
                        "For positions with fewer than 'window' elements, uses available elements."
                    ),
                    "impl": textwrap.dedent("""\
                        def running_mean(xs, window):
                            result = []
                            for i in range(len(xs)):
                                start = max(0, i - window + 1)
                                chunk = xs[start:i+1]
                                result.append(round(sum(chunk) / len(chunk), 6))
                            return result
                    """),
                },
                "cumulative_sum": {
                    "signature": "cumulative_sum(xs: list[float]) -> list[float]",
                    "description": "Returns cumulative sum: [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2], ...].",
                    "impl": textwrap.dedent("""\
                        def cumulative_sum(xs):
                            result = []
                            total = 0
                            for x in xs:
                                total += x
                                result.append(total)
                            return result
                    """),
                },
                "filter_above": {
                    "signature": "filter_above(xs: list[dict], key: str, threshold: float) -> list[dict]",
                    "description": "Filters items where item[key] >= threshold.",
                    "impl": textwrap.dedent("""\
                        def filter_above(xs, key, threshold):
                            return [x for x in xs if x.get(key, 0) >= threshold]
                    """),
                },
                "sort_by": {
                    "signature": "sort_by(xs: list[dict], key: str, reverse: bool = True) -> list[dict]",
                    "description": "Sorts a list of dicts by the given key.",
                    "impl": textwrap.dedent("""\
                        def sort_by(xs, key, reverse=True):
                            return sorted(xs, key=lambda x: x.get(key, 0), reverse=reverse)
                    """),
                },
                "group_by": {
                    "signature": "group_by(xs: list[dict], key: str) -> dict[str, list[dict]]",
                    "description": "Groups a list of dicts by the value of the given key.",
                    "impl": textwrap.dedent("""\
                        def group_by(xs, key):
                            groups = {}
                            for x in xs:
                                k = str(x.get(key, ''))
                                groups.setdefault(k, []).append(x)
                            return groups
                    """),
                },
            },
            "tasks": [
                {
                    "function_name": "summarize",
                    "signature": "def summarize(xs: list[int], k: int) -> list[float]",
                    "instructions": (
                        "1. Normalize xs\n"
                        "2. Compute pairwise_sum on the normalized list\n"
                        "3. Return top_k of the result with parameter k"
                    ),
                    "spec": {"type": "summarize"},
                },
                {
                    "function_name": "process_batch",
                    "signature": "def process_batch(items: list[dict], threshold: int) -> list[dict]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Sort the result by 'value' in descending order using sort_by\n"
                        "3. Return the sorted list"
                    ),
                    "spec": {"type": "process_batch"},
                },
                {
                    "function_name": "smooth_and_rank",
                    "signature": "def smooth_and_rank(xs: list[int], window: int, k: int) -> list[float]",
                    "instructions": (
                        "1. Compute running_mean of xs with the given window size\n"
                        "2. Return top_k of the smoothed result with parameter k"
                    ),
                    "spec": {"type": "smooth_and_rank"},
                },
                {
                    "function_name": "cumulative_normalize",
                    "signature": "def cumulative_normalize(xs: list[int]) -> list[float]",
                    "instructions": (
                        "1. Compute cumulative_sum of xs\n"
                        "2. Normalize the cumulative sum\n"
                        "3. Return the result"
                    ),
                    "spec": {"type": "cumulative_normalize"},
                },
                {
                    "function_name": "group_and_count",
                    "signature": "def group_and_count(items: list[dict], key: str) -> dict[str, int]",
                    "instructions": (
                        "1. Group the items by the given key using group_by\n"
                        "2. Return a dict mapping each group key to the count of items in that group"
                    ),
                    "spec": {"type": "group_and_count"},
                },
                {
                    "function_name": "filter_group_sort",
                    "signature": "def filter_group_sort(items: list[dict], threshold: int, group_key: str) -> dict[str, list[dict]]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Group the filtered items by group_key using group_by\n"
                        "3. Sort items within each group by 'value' descending using sort_by\n"
                        "4. Return the grouped result"
                    ),
                    "spec": {"type": "filter_group_sort"},
                },
            ],
        },
        "2.0": {
            "description": "Data Processing Library v2.0 (BREAKING CHANGES)",
            "functions": {
                "normalize": {
                    "signature": "normalize(xs: list[float]) -> list[float]",
                    "description": "Returns a list scaled so the sum is 1.0. If sum is 0, returns zeros.",
                    "impl": textwrap.dedent("""\
                        def normalize(xs):
                            total = sum(xs)
                            if total == 0:
                                return [0.0] * len(xs)
                            return [x / total for x in xs]
                    """),
                },
                "top_k": {
                    "signature": "top_k(xs: list[float], k: int) -> list[float]",
                    "description": "Returns the k largest values in descending order.",
                    "impl": textwrap.dedent("""\
                        def top_k(xs, k):
                            return sorted(xs, reverse=True)[:k]
                    """),
                },
                "pairwise_sum": {
                    "signature": "pairwise_sum(xs: list[float]) -> list[float]",
                    "description": (
                        "CHANGED in v2.0: Returns [2*(xs[0]+xs[1]), 2*(xs[2]+xs[3]), ...]. "
                        "Values are doubled. If odd length, last element is doubled too."
                    ),
                    "impl": textwrap.dedent("""\
                        def pairwise_sum(xs):
                            result = []
                            for i in range(0, len(xs) - 1, 2):
                                result.append(2 * (xs[i] + xs[i+1]))
                            if len(xs) % 2 == 1:
                                result.append(2 * xs[-1])
                            return result
                    """),
                },
                "running_mean": {
                    "signature": "running_mean(xs: list[float], window: int) -> list[float]",
                    "description": (
                        "Returns running mean with given window size. "
                        "For positions with fewer than 'window' elements, uses available elements."
                    ),
                    "impl": textwrap.dedent("""\
                        def running_mean(xs, window):
                            result = []
                            for i in range(len(xs)):
                                start = max(0, i - window + 1)
                                chunk = xs[start:i+1]
                                result.append(round(sum(chunk) / len(chunk), 6))
                            return result
                    """),
                },
                "cumulative_sum": {
                    "signature": "cumulative_sum(xs: list[float]) -> list[float]",
                    "description": "Returns cumulative sum: [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2], ...].",
                    "impl": textwrap.dedent("""\
                        def cumulative_sum(xs):
                            result = []
                            total = 0
                            for x in xs:
                                total += x
                                result.append(total)
                            return result
                    """),
                },
                "filter_above": {
                    "signature": "filter_above(xs: list[dict], key: str, threshold: float) -> list[dict]",
                    "description": "Filters items where item[key] >= threshold.",
                    "impl": textwrap.dedent("""\
                        def filter_above(xs, key, threshold):
                            return [x for x in xs if x.get(key, 0) >= threshold]
                    """),
                },
                "sort_by": {
                    "signature": "sort_by(xs: list[dict], key: str, reverse: bool = True) -> list[dict]",
                    "description": "Sorts a list of dicts by the given key.",
                    "impl": textwrap.dedent("""\
                        def sort_by(xs, key, reverse=True):
                            return sorted(xs, key=lambda x: x.get(key, 0), reverse=reverse)
                    """),
                },
                "group_by": {
                    "signature": "group_by(xs: list[dict], key: str) -> dict[str, list[dict]]",
                    "description": "Groups a list of dicts by the value of the given key.",
                    "impl": textwrap.dedent("""\
                        def group_by(xs, key):
                            groups = {}
                            for x in xs:
                                k = str(x.get(key, ''))
                                groups.setdefault(k, []).append(x)
                            return groups
                    """),
                },
                "transform_values": {
                    "signature": "transform_values(data: dict, operation: str) -> dict",
                    "description": "Applies an operation ('double', 'negate', 'square') to all numeric values.",
                    "impl": textwrap.dedent("""\
                        def transform_values(data, operation):
                            ops = {'double': lambda v: v * 2, 'negate': lambda v: -v, 'square': lambda v: v * v}
                            fn = ops.get(operation, lambda v: v)
                            return {k: fn(v) for k, v in data.items() if isinstance(v, (int, float))}
                    """),
                },
            },
            "tasks": [
                {
                    "function_name": "summarize",
                    "signature": "def summarize(xs: list[int], k: int) -> list[float]",
                    "instructions": (
                        "1. Normalize xs\n"
                        "2. Compute pairwise_sum on the normalized list "
                        "(note: v2.0 doubles values)\n"
                        "3. Return top_k of the result with parameter k"
                    ),
                    "spec": {"type": "summarize"},
                },
                {
                    "function_name": "process_batch",
                    "signature": "def process_batch(items: list[dict], threshold: int) -> list[dict]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Sort the result by 'value' in descending order using sort_by\n"
                        "3. Add a 'rank' field to each item (1-indexed)\n"
                        "4. Return the result"
                    ),
                    "spec": {"type": "process_batch"},
                },
                {
                    "function_name": "transform_data",
                    "signature": "def transform_data(data: dict, operation: str) -> dict",
                    "instructions": (
                        "1. Apply transform_values to the data with the given operation\n"
                        "2. Return the result"
                    ),
                    "spec": {"type": "transform_data"},
                },
                {
                    "function_name": "smooth_and_rank",
                    "signature": "def smooth_and_rank(xs: list[int], window: int, k: int) -> list[float]",
                    "instructions": (
                        "1. Compute running_mean of xs with the given window size\n"
                        "2. Return top_k of the smoothed result with parameter k"
                    ),
                    "spec": {"type": "smooth_and_rank"},
                },
                {
                    "function_name": "group_and_count",
                    "signature": "def group_and_count(items: list[dict], key: str) -> dict[str, int]",
                    "instructions": (
                        "1. Group the items by the given key using group_by\n"
                        "2. Return a dict mapping each group key to the count of items in that group"
                    ),
                    "spec": {"type": "group_and_count"},
                },
                {
                    "function_name": "paired_topk_smooth",
                    "signature": "def paired_topk_smooth(xs: list[int], k: int, window: int) -> list[float]",
                    "instructions": (
                        "1. Compute pairwise_sum of xs (note: v2.0 doubles values)\n"
                        "2. Apply running_mean with the given window\n"
                        "3. Return top_k with parameter k"
                    ),
                    "spec": {"type": "paired_topk_smooth"},
                },
                {
                    "function_name": "filter_group_sort",
                    "signature": "def filter_group_sort(items: list[dict], threshold: int, group_key: str) -> dict[str, list[dict]]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Group the filtered items by group_key using group_by\n"
                        "3. Sort items within each group by 'value' descending using sort_by\n"
                        "4. Return the grouped result"
                    ),
                    "spec": {"type": "filter_group_sort"},
                },
            ],
        },
        "3.0": {
            "description": "Data Processing Library v3.0 (REVERTED + NEW)",
            "functions": {
                "normalize": {
                    "signature": "normalize(xs: list[float]) -> list[float]",
                    "description": "Returns a list scaled so the sum is 1.0. If sum is 0, returns zeros.",
                    "impl": textwrap.dedent("""\
                        def normalize(xs):
                            total = sum(xs)
                            if total == 0:
                                return [0.0] * len(xs)
                            return [x / total for x in xs]
                    """),
                },
                "top_k": {
                    "signature": "top_k(xs: list[float], k: int) -> list[float]",
                    "description": "Returns the k largest values in descending order.",
                    "impl": textwrap.dedent("""\
                        def top_k(xs, k):
                            return sorted(xs, reverse=True)[:k]
                    """),
                },
                "pairwise_sum": {
                    "signature": "pairwise_sum(xs: list[float]) -> list[float]",
                    "description": (
                        "REVERTED in v3.0 to v1.0 behavior: Returns [xs[0]+xs[1], xs[2]+xs[3], ...]. "
                        "No doubling. If odd length, last element is kept."
                    ),
                    "impl": textwrap.dedent("""\
                        def pairwise_sum(xs):
                            result = []
                            for i in range(0, len(xs) - 1, 2):
                                result.append(xs[i] + xs[i+1])
                            if len(xs) % 2 == 1:
                                result.append(xs[-1])
                            return result
                    """),
                },
                "running_mean": {
                    "signature": "running_mean(xs: list[float], window: int) -> list[float]",
                    "description": (
                        "Returns running mean with given window size. "
                        "For positions with fewer than 'window' elements, uses available elements."
                    ),
                    "impl": textwrap.dedent("""\
                        def running_mean(xs, window):
                            result = []
                            for i in range(len(xs)):
                                start = max(0, i - window + 1)
                                chunk = xs[start:i+1]
                                result.append(round(sum(chunk) / len(chunk), 6))
                            return result
                    """),
                },
                "cumulative_sum": {
                    "signature": "cumulative_sum(xs: list[float]) -> list[float]",
                    "description": "Returns cumulative sum: [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2], ...].",
                    "impl": textwrap.dedent("""\
                        def cumulative_sum(xs):
                            result = []
                            total = 0
                            for x in xs:
                                total += x
                                result.append(total)
                            return result
                    """),
                },
                "filter_above": {
                    "signature": "filter_above(xs: list[dict], key: str, threshold: float) -> list[dict]",
                    "description": "Filters items where item[key] >= threshold.",
                    "impl": textwrap.dedent("""\
                        def filter_above(xs, key, threshold):
                            return [x for x in xs if x.get(key, 0) >= threshold]
                    """),
                },
                "sort_by": {
                    "signature": "sort_by(xs: list[dict], key: str, reverse: bool = True) -> list[dict]",
                    "description": "Sorts a list of dicts by the given key.",
                    "impl": textwrap.dedent("""\
                        def sort_by(xs, key, reverse=True):
                            return sorted(xs, key=lambda x: x.get(key, 0), reverse=reverse)
                    """),
                },
                "group_by": {
                    "signature": "group_by(xs: list[dict], key: str) -> dict[str, list[dict]]",
                    "description": "Groups a list of dicts by the value of the given key.",
                    "impl": textwrap.dedent("""\
                        def group_by(xs, key):
                            groups = {}
                            for x in xs:
                                k = str(x.get(key, ''))
                                groups.setdefault(k, []).append(x)
                            return groups
                    """),
                },
                "transform_values": {
                    "signature": "transform_values(data: dict, operation: str) -> dict",
                    "description": "Applies an operation ('double', 'negate', 'square') to all numeric values.",
                    "impl": textwrap.dedent("""\
                        def transform_values(data, operation):
                            ops = {'double': lambda v: v * 2, 'negate': lambda v: -v, 'square': lambda v: v * v}
                            fn = ops.get(operation, lambda v: v)
                            return {k: fn(v) for k, v in data.items() if isinstance(v, (int, float))}
                    """),
                },
                "zip_with": {
                    "signature": "zip_with(xs: list[float], ys: list[float], op: str) -> list[float]",
                    "description": (
                        "Combines two lists element-wise. op is 'add', 'mul', or 'max'. "
                        "Lists must be same length."
                    ),
                    "impl": textwrap.dedent("""\
                        def zip_with(xs, ys, op):
                            ops = {'add': lambda a, b: a + b, 'mul': lambda a, b: a * b, 'max': lambda a, b: max(a, b)}
                            fn = ops.get(op, lambda a, b: a + b)
                            return [fn(x, y) for x, y in zip(xs, ys)]
                    """),
                },
            },
            "tasks": [
                {
                    "function_name": "summarize",
                    "signature": "def summarize(xs: list[int], k: int) -> list[float]",
                    "instructions": (
                        "1. Normalize xs\n"
                        "2. Compute pairwise_sum on the normalized list "
                        "(note: v3.0 reverted to v1.0 — no doubling)\n"
                        "3. Return top_k of the result with parameter k"
                    ),
                    "spec": {"type": "summarize"},
                },
                {
                    "function_name": "process_batch",
                    "signature": "def process_batch(items: list[dict], threshold: int) -> list[dict]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Sort the result by 'value' in descending order using sort_by\n"
                        "3. Return the sorted list (no rank field in v3.0)"
                    ),
                    "spec": {"type": "process_batch"},
                },
                {
                    "function_name": "transform_data",
                    "signature": "def transform_data(data: dict, operation: str) -> dict",
                    "instructions": (
                        "1. Apply transform_values to the data with the given operation\n"
                        "2. Return the result"
                    ),
                    "spec": {"type": "transform_data"},
                },
                {
                    "function_name": "smooth_and_rank",
                    "signature": "def smooth_and_rank(xs: list[int], window: int, k: int) -> list[float]",
                    "instructions": (
                        "1. Compute running_mean of xs with the given window size\n"
                        "2. Return top_k of the smoothed result with parameter k"
                    ),
                    "spec": {"type": "smooth_and_rank"},
                },
                {
                    "function_name": "cumulative_normalize",
                    "signature": "def cumulative_normalize(xs: list[int]) -> list[float]",
                    "instructions": (
                        "1. Compute cumulative_sum of xs\n"
                        "2. Normalize the cumulative sum\n"
                        "3. Return the result"
                    ),
                    "spec": {"type": "cumulative_normalize"},
                },
                {
                    "function_name": "blend_and_summarize",
                    "signature": "def blend_and_summarize(xs: list[int], ys: list[int], k: int) -> list[float]",
                    "instructions": (
                        "1. Normalize xs and ys separately\n"
                        "2. Combine them element-wise using zip_with with 'add' operation\n"
                        "3. Return top_k of the result with parameter k"
                    ),
                    "spec": {"type": "blend_and_summarize"},
                },
                {
                    "function_name": "filter_group_sort",
                    "signature": "def filter_group_sort(items: list[dict], threshold: int, group_key: str) -> dict[str, list[dict]]",
                    "instructions": (
                        "1. Filter items where 'value' >= threshold using filter_above\n"
                        "2. Group the filtered items by group_key using group_by\n"
                        "3. Sort items within each group by 'value' descending using sort_by\n"
                        "4. Return the grouped result"
                    ),
                    "spec": {"type": "filter_group_sort"},
                },
            ],
        },
    }


def get_api_definition(version: str) -> dict[str, Any]:
    """Get the API definition for a specific version."""
    global _API_DEFS_CACHE
    if _API_DEFS_CACHE is None:
        _API_DEFS_CACHE = _build_api_definitions()
    return _API_DEFS_CACHE.get(version, _API_DEFS_CACHE["2.0"])


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_api_code_instance(
    spec: Spec,
    stream_id: str,
    index: int,
    split: str,
    seed: int,
    rng: random.Random,
) -> Instance:
    """Generate a single API code generation instance."""
    config = spec.generator_config
    api_version = config.get("api_version", "1.0")
    test_count = spec.difficulty.test_count or 5

    api_def = get_api_definition(api_version)
    tasks = api_def.get("tasks", [])

    if not tasks:
        api_def = get_api_definition("1.0")
        tasks = api_def["tasks"]

    # Pick a task for this instance
    task = tasks[index % len(tasks)]

    # Generate test cases
    test_cases = generate_test_cases(
        rng=rng,
        api_version=api_version,
        function_name=task["function_name"],
        function_spec=task["spec"],
        count=test_count,
    )

    # Build the API code (implementations)
    api_code_parts = []
    for fname, fdef in api_def["functions"].items():
        api_code_parts.append(fdef["impl"])
    api_impl = "\n".join(api_code_parts)

    # Build test code
    test_code = build_test_code(task["function_name"], test_cases)

    # Build the reference solution
    reference_solution = _build_reference_solution(task, api_version)

    # Compute expected target via actual execution
    from continual_benchmark.tasks.api_code.sandbox import execute_code_with_tests
    exec_result = execute_code_with_tests(
        user_code=reference_solution,
        api_code=api_impl,
        test_code=test_code,
        timeout=10,
    )

    target = json.dumps({
        "tests_passed": exec_result.tests_passed,
        "tests_total": exec_result.tests_total,
        "all_passed": exec_result.success,
    })

    # Format prompt — include example test cases to make each instance unique
    prompt = _format_api_prompt(api_def, task, test_cases[:2])

    metadata = {
        "difficulty": spec.difficulty.level,
        "api_version": api_version,
        "function_name": task["function_name"],
        "test_count": len(test_cases),
        "drift_type": spec.drift.drift_type.value if spec.drift else None,
        "api_surface_size": len(api_def["functions"]),
        "_api_impl": api_impl,
        "_test_code": test_code,
    }

    uid = instance_uid(
        spec.suite, FamilyName.API_CODE.value,
        spec.stage, split, index, seed,
    )

    return Instance(
        uid=uid,
        suite=spec.suite,
        stream_id=stream_id,
        family=FamilyName.API_CODE,
        spec_id=spec.spec_id,
        stage=spec.stage,
        split=Split(split),
        prompt=prompt,
        target=target,
        seed=seed,
        metadata=metadata,
    )


def _build_reference_solution(task: dict[str, Any], api_version: str) -> str:
    """Build a reference solution for the task."""
    fname = task["function_name"]
    spec_type = task["spec"].get("type", "")

    if spec_type == "summarize":
        return textwrap.dedent(f"""\
            def {fname}(xs, k):
                n = normalize(xs)
                p = pairwise_sum(n)
                return [round(v, 6) for v in top_k(p, k)]
        """)

    elif spec_type == "process_batch":
        if api_version == "2.0":
            return textwrap.dedent(f"""\
                def {fname}(items, threshold):
                    filtered = filter_above(items, 'value', threshold)
                    sorted_items = sort_by(filtered, 'value', reverse=True)
                    for i, item in enumerate(sorted_items):
                        item['rank'] = i + 1
                    return sorted_items
            """)
        return textwrap.dedent(f"""\
            def {fname}(items, threshold):
                filtered = filter_above(items, 'value', threshold)
                return sort_by(filtered, 'value', reverse=True)
        """)

    elif spec_type == "transform_data":
        return textwrap.dedent(f"""\
            def {fname}(data, operation):
                return transform_values(data, operation)
        """)

    elif spec_type == "smooth_and_rank":
        return textwrap.dedent(f"""\
            def {fname}(xs, window, k):
                smoothed = running_mean(xs, window)
                return [round(v, 6) for v in top_k(smoothed, k)]
        """)

    elif spec_type == "cumulative_normalize":
        return textwrap.dedent(f"""\
            def {fname}(xs):
                c = cumulative_sum(xs)
                return normalize(c)
        """)

    elif spec_type == "paired_topk_smooth":
        return textwrap.dedent(f"""\
            def {fname}(xs, k, window):
                p = pairwise_sum(xs)
                smoothed = running_mean(p, window)
                return [round(v, 6) for v in top_k(smoothed, k)]
        """)

    elif spec_type == "normalize_and_accumulate":
        return textwrap.dedent(f"""\
            def {fname}(xs):
                n = normalize(xs)
                c = cumulative_sum(n)
                return [round(v, 6) for v in c]
        """)

    elif spec_type == "group_and_count":
        return textwrap.dedent(f"""\
            def {fname}(items, key):
                groups = group_by(items, key)
                return {{k: len(v) for k, v in groups.items()}}
        """)

    elif spec_type == "filter_group_sort":
        return textwrap.dedent(f"""\
            def {fname}(items, threshold, group_key):
                filtered = filter_above(items, 'value', threshold)
                groups = group_by(filtered, group_key)
                return {{k: sort_by(v, 'value', reverse=True) for k, v in groups.items()}}
        """)

    elif spec_type == "blend_and_summarize":
        return textwrap.dedent(f"""\
            def {fname}(xs, ys, k):
                nx = normalize(xs)
                ny = normalize(ys)
                blended = zip_with(nx, ny, 'add')
                return [round(v, 6) for v in top_k(blended, k)]
        """)

    return f"def {fname}(*args, **kwargs): pass"


def _format_api_prompt(
    api_def: dict[str, Any],
    task: dict[str, Any],
    example_tests: list[dict[str, Any]] | None = None,
) -> str:
    """Format an API code generation prompt."""
    lines = [f"You are given a library: {api_def['description']}", ""]
    lines.append("Available API functions:")
    lines.append("")

    for fname, fdef in api_def["functions"].items():
        lines.append(f"  {fdef['signature']}")
        lines.append(f"    {fdef['description']}")
        lines.append("")

    lines.append("Write a Python function:")
    lines.append(f"  {task['signature']}")
    lines.append("")
    lines.append("It should:")
    for line in task["instructions"].split("\n"):
        lines.append(f"  {line}")
    lines.append("")

    # Include example test cases to make each instance unique
    if example_tests:
        lines.append("Examples:")
        for tc in example_tests:
            args_str = ", ".join(repr(a) for a in tc["args"])
            lines.append(
                f"  {task['function_name']}({args_str}) -> {repr(tc['expected'])}"
            )
        lines.append("")

    lines.append(
        "Return only the function code. "
        "Do not include imports or the API implementations."
    )

    return "\n".join(lines)

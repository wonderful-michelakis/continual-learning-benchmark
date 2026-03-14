"""Replay buffer implementations for limited-memory continual learning.

Supports:
- FIFO (first-in-first-out) buffer
- Reservoir sampling buffer
- Token-budget-aware buffer

Implements Track B (limited-memory replay): methods may store a bounded
replay buffer from previous tasks.
"""

from __future__ import annotations

import random
from typing import Any

from continual_benchmark.core.schemas import Instance


class ReplayBuffer:
    """Base replay buffer with configurable capacity."""

    def __init__(self, max_examples: int = 1000):
        self.max_examples = max_examples
        self._buffer: list[Instance] = []
        self._total_seen: int = 0

    def add_stage(self, instances: list[Instance]) -> None:
        """Add a stage's worth of instances to the buffer."""
        raise NotImplementedError

    def get_replay_data(self) -> list[Instance]:
        """Return the current replay buffer contents."""
        return list(self._buffer)

    def size(self) -> int:
        """Current number of instances in the buffer."""
        return len(self._buffer)

    def stats(self) -> dict[str, Any]:
        """Return buffer statistics."""
        stages = set(i.stage for i in self._buffer)
        families = set(i.family.value for i in self._buffer)
        return {
            "size": self.size(),
            "max_examples": self.max_examples,
            "total_seen": self._total_seen,
            "stages_represented": sorted(stages),
            "families_represented": sorted(families),
        }


class FIFOBuffer(ReplayBuffer):
    """First-in-first-out replay buffer.

    Keeps the most recent examples up to max_examples.
    """

    def add_stage(self, instances: list[Instance]) -> None:
        self._total_seen += len(instances)
        self._buffer.extend(instances)
        # Keep only the most recent max_examples
        if len(self._buffer) > self.max_examples:
            self._buffer = self._buffer[-self.max_examples:]


class ReservoirBuffer(ReplayBuffer):
    """Reservoir sampling replay buffer.

    Maintains a uniform random sample over all seen instances.
    Each instance has an equal probability of being in the buffer.
    """

    def __init__(self, max_examples: int = 1000, seed: int = 42):
        super().__init__(max_examples)
        self._rng = random.Random(seed)

    def add_stage(self, instances: list[Instance]) -> None:
        for inst in instances:
            self._total_seen += 1
            if len(self._buffer) < self.max_examples:
                self._buffer.append(inst)
            else:
                # Replace with probability max_examples / total_seen
                j = self._rng.randint(0, self._total_seen - 1)
                if j < self.max_examples:
                    self._buffer[j] = inst


class TokenBudgetBuffer(ReplayBuffer):
    """Token-budget-aware replay buffer.

    Bounds the total token count rather than example count.
    Uses a simple word-count approximation for tokens.
    """

    def __init__(self, max_tokens: int = 100_000, seed: int = 42):
        super().__init__(max_examples=0)  # No example limit
        self.max_tokens = max_tokens
        self._rng = random.Random(seed)
        self._current_tokens: int = 0

    def _estimate_tokens(self, inst: Instance) -> int:
        """Rough token estimate: word count * 1.3."""
        words = len(inst.prompt.split()) + len(inst.target.split())
        return int(words * 1.3)

    def add_stage(self, instances: list[Instance]) -> None:
        self._total_seen += len(instances)
        for inst in instances:
            tokens = self._estimate_tokens(inst)
            if self._current_tokens + tokens <= self.max_tokens:
                self._buffer.append(inst)
                self._current_tokens += tokens
            else:
                # Try to replace a random existing instance
                if self._buffer:
                    idx = self._rng.randint(0, len(self._buffer) - 1)
                    old_tokens = self._estimate_tokens(self._buffer[idx])
                    if self._current_tokens - old_tokens + tokens <= self.max_tokens:
                        self._current_tokens = self._current_tokens - old_tokens + tokens
                        self._buffer[idx] = inst

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base["current_tokens"] = self._current_tokens
        base["max_tokens"] = self.max_tokens
        return base

"""Deterministic RNG utilities for reproducible benchmark generation.

Seed hierarchy:
    global_seed -> stage_seed -> split_seed -> instance_seed

All generation is deterministic given (suite_version, global_seed, stage, split, index).
"""

import hashlib
import random
from dataclasses import dataclass

from continual_benchmark.core.constants import SPLIT_SEED_OFFSETS, Split


def _hash_to_seed(data: str) -> int:
    """Hash a string to a 32-bit seed for reproducibility across platforms."""
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def derive_stage_seed(global_seed: int, stage: int) -> int:
    """Derive a deterministic seed for a specific stage."""
    return _hash_to_seed(f"stage:{global_seed}:{stage}")


def derive_split_seed(stage_seed: int, split: Split) -> int:
    """Derive a deterministic seed for a specific split within a stage."""
    offset = SPLIT_SEED_OFFSETS[split]
    return _hash_to_seed(f"split:{stage_seed}:{offset}")


def derive_instance_seed(split_seed: int, index: int) -> int:
    """Derive a deterministic seed for a specific instance within a split."""
    return _hash_to_seed(f"instance:{split_seed}:{index}")


@dataclass(frozen=True)
class SeedChain:
    """Manages the full seed derivation chain for a benchmark build.

    Usage:
        chain = SeedChain(global_seed=42)
        stage_rng = chain.stage_rng(stage=3, split=Split.TRAIN)
        instance_seed = chain.instance_seed(stage=3, split=Split.TRAIN, index=5)
    """

    global_seed: int

    def stage_seed(self, stage: int) -> int:
        """Get the seed for a given stage."""
        return derive_stage_seed(self.global_seed, stage)

    def split_seed(self, stage: int, split: Split) -> int:
        """Get the seed for a given stage + split."""
        stage_s = self.stage_seed(stage)
        return derive_split_seed(stage_s, split)

    def instance_seed(self, stage: int, split: Split, index: int) -> int:
        """Get the seed for a specific instance."""
        split_s = self.split_seed(stage, split)
        return derive_instance_seed(split_s, index)

    def stage_rng(self, stage: int, split: Split) -> random.Random:
        """Get a seeded Random instance for generating a split within a stage."""
        seed = self.split_seed(stage, split)
        return random.Random(seed)

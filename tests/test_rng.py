"""Tests for deterministic RNG utilities."""

from continual_benchmark.core.constants import Split
from continual_benchmark.core.rng import SeedChain, derive_instance_seed, derive_stage_seed


class TestSeedDerivation:
    def test_stage_seed_deterministic(self):
        """Same inputs always produce the same seed."""
        s1 = derive_stage_seed(42, 1)
        s2 = derive_stage_seed(42, 1)
        assert s1 == s2

    def test_different_stages_different_seeds(self):
        s1 = derive_stage_seed(42, 1)
        s2 = derive_stage_seed(42, 2)
        assert s1 != s2

    def test_different_global_seeds(self):
        s1 = derive_stage_seed(42, 1)
        s2 = derive_stage_seed(43, 1)
        assert s1 != s2


class TestSeedChain:
    def test_instance_seed_deterministic(self):
        chain = SeedChain(global_seed=42)
        s1 = chain.instance_seed(1, Split.TRAIN, 0)
        s2 = chain.instance_seed(1, Split.TRAIN, 0)
        assert s1 == s2

    def test_different_splits_different_seeds(self):
        chain = SeedChain(global_seed=42)
        s_train = chain.instance_seed(1, Split.TRAIN, 0)
        s_dev = chain.instance_seed(1, Split.DEV, 0)
        assert s_train != s_dev

    def test_different_indices_different_seeds(self):
        chain = SeedChain(global_seed=42)
        s0 = chain.instance_seed(1, Split.TRAIN, 0)
        s1 = chain.instance_seed(1, Split.TRAIN, 1)
        assert s0 != s1

    def test_stage_rng_deterministic(self):
        chain = SeedChain(global_seed=42)
        rng1 = chain.stage_rng(1, Split.TRAIN)
        val1 = rng1.random()

        rng2 = chain.stage_rng(1, Split.TRAIN)
        val2 = rng2.random()

        assert val1 == val2

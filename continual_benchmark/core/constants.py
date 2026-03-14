"""Global constants for the benchmark framework."""

from enum import Enum


class Split(str, Enum):
    """Dataset split types."""

    TRAIN = "train"
    DEV = "dev"
    PUBLIC_TEST = "public_test"
    PRIVATE_TEST = "private_test"


class FamilyName(str, Enum):
    """Registered task family identifiers."""

    DSL_EXEC = "dsl_exec"
    STRUCTURED_TRANSFORM = "structured_transform"
    SQL_REASONING = "sql_reasoning"
    API_CODE = "api_code"


class DriftType(str, Enum):
    """Types of concept drift between stages."""

    NONE = "none"
    SEMANTIC = "semantic"          # Operator/rule semantics change
    STRUCTURAL = "structural"      # Schema/structure changes
    ADDITIVE = "additive"          # New operators/rules added
    REMOVAL = "removal"            # Operators/rules removed


class Track(str, Enum):
    """Continual learning evaluation tracks."""

    NO_REPLAY = "no_replay"           # Track A: no data replay
    LIMITED_MEMORY = "limited_memory"  # Track B: bounded replay buffer


# Default split sizes per stage
DEFAULT_SPLIT_SIZES: dict[Split, int] = {
    Split.TRAIN: 1000,
    Split.DEV: 200,
    Split.PUBLIC_TEST: 200,
    Split.PRIVATE_TEST: 200,
}

# Seed offsets for deterministic split generation
SPLIT_SEED_OFFSETS: dict[Split, int] = {
    Split.TRAIN: 0,
    Split.DEV: 1_000_000,
    Split.PUBLIC_TEST: 2_000_000,
    Split.PRIVATE_TEST: 3_000_000,
}

DEFAULT_GLOBAL_SEED = 42
SUITE_VERSION = "v1"

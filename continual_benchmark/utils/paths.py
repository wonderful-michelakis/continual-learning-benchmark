"""Path utilities for locating benchmark resources."""

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Default locations for stream definitions
STREAM_DEFINITIONS_DIR = PACKAGE_ROOT / "streams" / "definitions"


def stage_dir(output_root: Path, stage: int) -> Path:
    """Return the directory path for a specific stage's artifacts."""
    return output_root / f"stage_{stage:03d}"


def split_path(output_root: Path, stage: int, split: str) -> Path:
    """Return the JSONL file path for a specific stage + split."""
    return stage_dir(output_root, stage) / f"{split}.jsonl"

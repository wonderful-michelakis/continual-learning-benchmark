"""Logging configuration for the benchmark framework."""

import logging
import sys


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure and return the benchmark logger."""
    logger = logging.getLogger("continual_benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


logger = setup_logging()

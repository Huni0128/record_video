"""File system helpers for output/session directories."""
from __future__ import annotations

import os
from datetime import datetime


__all__ = ["ensure_out_dir", "timestamped_subdir"]


DEFAULT_BASE_DIR = "save/farm_record"


def ensure_out_dir(base: str) -> str:
    """Create a timestamped output directory and return its absolute path."""
    return timestamped_subdir(base or DEFAULT_BASE_DIR)


def timestamped_subdir(base: str) -> str:
    """Create a timestamped sub-directory inside *base* and return its path."""
    base = (base or DEFAULT_BASE_DIR).strip() or DEFAULT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_dir)

"""Core domain models and helpers for recording sessions."""
from __future__ import annotations

from .config import RecordConfig
from .paths import DEFAULT_BASE_DIR, ensure_out_dir, timestamped_subdir
from .video import fourcc_mp4v

__all__ = [
    "RecordConfig",
    "DEFAULT_BASE_DIR",
    "ensure_out_dir",
    "timestamped_subdir",
    "fourcc_mp4v",
]

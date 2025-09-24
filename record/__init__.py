"""High level package for recording and analysing RealSense depth data."""
from __future__ import annotations

from .analysis import DEFAULT_ANALYSIS_BASE, analyze_npy
from .core import (
    DEFAULT_BASE_DIR,
    RecordConfig,
    ensure_out_dir,
    fourcc_mp4v,
    timestamped_subdir,
)
from .recording import RecordThread

__all__ = [
    "DEFAULT_BASE_DIR",
    "RecordConfig",
    "RecordThread",
    "analyze_npy",
    "DEFAULT_ANALYSIS_BASE",
    "ensure_out_dir",
    "fourcc_mp4v",
    "timestamped_subdir",
    "main",
    "MainWindow",
    "FuncThread",
    "NpyViewerWidget",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy loader
    if name in {"main", "MainWindow", "FuncThread", "NpyViewerWidget"}:
        from . import gui

        return getattr(gui, name)
    raise AttributeError(name)

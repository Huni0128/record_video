"""Video related utilities."""
from __future__ import annotations

import cv2

__all__ = ["fourcc_mp4v"]


def fourcc_mp4v() -> int:
    """Return the integer FourCC for MP4V videos."""
    return cv2.VideoWriter_fourcc(*"mp4v")

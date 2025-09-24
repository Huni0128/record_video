"""Qt related helpers such as OpenCV → QImage conversion."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5 import QtGui

__all__ = ["qimage_from_cv"]


def qimage_from_cv(mat_bgr_or_gray: np.ndarray) -> QtGui.QImage:
    """Convert a BGR or grayscale ``numpy.ndarray`` into a ``QImage``."""
    if mat_bgr_or_gray is None:
        raise ValueError("입력이 None 입니다.")

    if mat_bgr_or_gray.dtype != np.uint8:
        raise ValueError(
            f"지원 dtype은 uint8 뿐입니다. got={mat_bgr_or_gray.dtype}"
        )

    mat = (
        mat_bgr_or_gray
        if mat_bgr_or_gray.flags["C_CONTIGUOUS"]
        else np.ascontiguousarray(mat_bgr_or_gray)
    )

    if mat.ndim == 2:
        h, w = mat.shape
        bytes_per_line = w
        qimg = QtGui.QImage(
            mat.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8
        )
        return qimg.copy()

    if mat.ndim == 3 and mat.shape[2] == 3:
        h, w, _ = mat.shape
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        rgb = rgb if rgb.flags["C_CONTIGUOUS"] else np.ascontiguousarray(rgb)
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return qimg.copy()

    raise ValueError(
        "지원 형상은 GRAY (H, W, uint8) 또는 BGR (H, W, 3, uint8) 입니다. "
        f"got shape={mat.shape}"
    )

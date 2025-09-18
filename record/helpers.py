# -*- coding: utf-8 -*-
"""
공용 유틸: 출력 폴더 생성, OpenCV Mat → QImage 변환, MP4V FourCC.

Docstring 스타일: Google Style
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtGui


__all__ = ["ensure_out_dir", "qimage_from_cv", "fourcc_mp4v"]


def ensure_out_dir(base: str) -> str:
    """타임스탬프 하위 폴더를 생성하여 절대 경로를 반환합니다.

    Args:
        base: 출력 베이스 폴더 경로. 공백/None이면 "save/farm_record" 사용.

    Returns:
        생성된 출력 폴더의 절대 경로.
    """
    base = (base or "").strip() or "save/farm_record"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.abspath(out_dir)


def qimage_from_cv(mat_bgr_or_gray: np.ndarray) -> QtGui.QImage:
    """OpenCV Mat(BGR 또는 GRAY)을 Qt QImage로 변환합니다.

    - 입력 배열이 비연속(non-contiguous)인 경우 안전을 위해 복사합니다.
    - 반환 시 .copy()로 Qt 내부 버퍼를 소유하게 합니다(수명 안전).

    Args:
        mat_bgr_or_gray: 2D (H, W) uint8 GRAY 또는 3D (H, W, 3) BGR 배열.

    Returns:
        QtGui.QImage 객체.

    Raises:
        ValueError: 지원하지 않는 dtype/채널/차원일 때.
    """
    if mat_bgr_or_gray is None:
        raise ValueError("입력이 None 입니다.")

    if mat_bgr_or_gray.dtype != np.uint8:
        raise ValueError(
            f"지원 dtype은 uint8 뿐입니다. got={mat_bgr_or_gray.dtype}"
        )

    # 메모리 연속성 보장
    mat = (
        mat_bgr_or_gray
        if mat_bgr_or_gray.flags["C_CONTIGUOUS"]
        else np.ascontiguousarray(mat_bgr_or_gray)
    )

    if mat.ndim == 2:
        h, w = mat.shape
        bytes_per_line = w  # uint8 GRAY
        qimg = QtGui.QImage(
            mat.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8
        )
        return qimg.copy()

    if mat.ndim == 3 and mat.shape[2] == 3:
        h, w, _ = mat.shape
        # BGR → RGB
        rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
        rgb = (
            rgb if rgb.flags["C_CONTIGUOUS"] else np.ascontiguousarray(rgb)
        )
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return qimg.copy()

    raise ValueError(
        "지원 형상은 GRAY (H, W, uint8) 또는 BGR (H, W, 3, uint8) 입니다. "
        f"got shape={mat.shape}"
    )


def fourcc_mp4v() -> int:
    """MP4V FourCC 정수 값을 반환합니다.

    Returns:
        OpenCV VideoWriter_fourcc 값.
    """
    return cv2.VideoWriter_fourcc(*"mp4v")

# -*- coding: utf-8 -*-
"""
기록(Record) 설정 데이터 클래스 모듈.

Docstring 스타일: Google Style
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RecordConfig:
    """기록(Record) 설정값.

    Start 버튼을 누르면 Stop 버튼을 누를 때까지 계속 기록됩니다.

    Attributes:
        out_dir: 출력 폴더 경로.
        width: 프레임 너비 (픽셀 단위).
        height: 프레임 높이 (픽셀 단위).
        fps: 초당 프레임 수.
        save_videos: 비디오 파일 저장 여부.
        save_depth_npy: NPY 파일 저장 여부 (depth만 저장).
        save_frames: 프레임을 이미지/NPY로 분리 저장 여부.
        save_bag: 실시간으로 .bag 파일을 함께 저장할지 여부.
        frame_stride: 이미지/NPY 저장 프레임 간격 (1이면 모든 프레임 저장).
    """

    out_dir: str
    width: int
    height: int
    fps: int
    save_videos: bool
    save_depth_npy: bool
    save_frames: bool = False
    save_bag: bool = False
    frame_stride: int = 1

    def __post_init__(self) -> None:
        """입력값 검증 및 보정."""

        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")

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
    """

    out_dir: str
    width: int
    height: int
    fps: int
    save_videos: bool
    save_depth_npy: bool

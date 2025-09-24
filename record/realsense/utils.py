# -*- coding: utf-8 -*-
"""
RealSense 관련 유틸.

- pyrealsense2는 런타임에만 의존하도록 임포트 지연(_import_rs).
- 센서 옵션/프레임 메타데이터/내·외부 파라미터 접근 헬퍼 제공.

Docstring 스타일: Google Style
"""
from __future__ import annotations

from typing import Any, Dict, Optional


__all__ = [
    "_import_rs",
    "get_available_metadata",
    "sensor_options_to_dict",
    "stream_intrinsics",
    "extrinsics_between",
]


def _import_rs() -> Any:
    """pyrealsense2 모듈을 지연 임포트합니다.

    Returns:
        임포트된 pyrealsense2 모듈 객체.

    Raises:
        ImportError: pyrealsense2가 설치되어 있지 않은 경우.
    """
    try:
        import pyrealsense2 as rs  # type: ignore
        return rs
    except ImportError as exc:
        raise ImportError(
            "pyrealsense2가 설치되어 있지 않습니다.\n\n`pip install pyrealsense2`를 실행하세요."
        ) from exc


def get_available_metadata(frame: Any) -> Dict[str, Any]:
    """프레임이 지원하는 메타데이터를 모두 조회하여 반환합니다.

    Args:
        frame: RealSense 프레임 객체.

    Returns:
        메타데이터 이름 → 값의 딕셔너리.
    """
    rs = _import_rs()
    md: Dict[str, Any] = {}
    for name in dir(rs.frame_metadata_value):
        if name.startswith("__"):
            continue
        try:
            enum_val = getattr(rs.frame_metadata_value, name)
            if frame.supports_frame_metadata(enum_val):
                md[name] = frame.get_frame_metadata(enum_val)
        except Exception:
            # 일부 메타데이터는 장치/프레임 타입에 따라 미지원일 수 있음
            pass
    return md


def sensor_options_to_dict(sensor: Any) -> Dict[str, Any]:
    """센서가 지원하는 모든 옵션의 현재 값과 범위를 조회합니다.

    Args:
        sensor: RealSense 센서 객체.

    Returns:
        옵션 이름 → {value, min, max, step, default} 또는 {error}의 딕셔너리.
    """
    rs = _import_rs()
    info: Dict[str, Any] = {}
    for name in dir(rs.option):
        if name.startswith("__"):
            continue
        try:
            opt = getattr(rs.option, name)
            if sensor.supports(opt):
                try:
                    val = sensor.get_option(opt)
                    rng = sensor.get_option_range(opt)
                    info[name] = {
                        "value": val,
                        "min": rng.min,
                        "max": rng.max,
                        "step": rng.step,
                        "default": rng.default,
                    }
                except Exception as exc:
                    info[name] = {"error": str(exc)}
        except Exception:
            pass
    return info


def stream_intrinsics(profile: Any) -> Optional[Dict[str, Any]]:
    """비디오 스트림 프로파일에서 내장 파라미터를 추출합니다.

    Args:
        profile: rs.stream_profile 또는 None.

    Returns:
        내장 파라미터 딕셔너리(width, height, ppx, ppy, fx, fy, model, coeffs),
        실패/미지원 시 None.
    """
    if profile is None:
        return None
    try:
        vs_profile = profile.as_video_stream_profile()
        intr = vs_profile.get_intrinsics()
        return {
            "width": intr.width,
            "height": intr.height,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "fx": intr.fx,
            "fy": intr.fy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
    except Exception:
        return None


def extrinsics_between(src_profile: Any, dst_profile: Any) -> Optional[Dict[str, Any]]:
    """두 스트림 프로파일 간 외부 파라미터(회전/평행이동)를 계산합니다.

    Args:
        src_profile: 원본 스트림 프로파일.
        dst_profile: 대상 스트림 프로파일.

    Returns:
        {"rotation": [...], "translation": [...]} 또는 실패 시 None.
    """
    if src_profile is None or dst_profile is None:
        return None
    try:
        ext = src_profile.get_extrinsics_to(dst_profile)
        return {
            "rotation": list(ext.rotation),
            "translation": list(ext.translation),
        }
    except Exception:
        return None

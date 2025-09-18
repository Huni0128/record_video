# -*- coding: utf-8 -*-
"""
GUI와 CLI에서 모두 사용할 수 있는 NPY(depth only) 분석 모듈.

기능
- 프레임별 통계 CSV 생성(frame_stats_from_npy.csv)

주의
- 픽셀 CSV는 용량이 매우 클 수 있습니다. 필요한 경우에만 사용하세요.

예시
    python -m farm_record.analyze_npy \
        path/to/depth_raw_frames.npy \
        --out runs/npy_analysis_out

Docstring 스타일: Google Style
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np

__all__ = ["analyze_npy"]  # 외부 임포트 보장


# ----------------------------- 내부 유틸 ------------------------------ #
def _ensure_timestamped_dir(base: str) -> str:
    """타임스탬프 하위 폴더를 생성하여 경로를 반환합니다.

    Args:
        base: 출력 베이스 폴더 경로.

    Returns:
        생성된 타임스탬프 하위 폴더의 절대 경로.
    """
    base = (base or "").strip() or "runs/npy_analysis_out"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _find_scale_from_json(npy_path: str, default: float = 0.001) -> float:
    """동일 폴더의 메타 JSON에서 깊이 스케일(m/단위)을 추정합니다.

    Args:
        npy_path: 입력 NPY 파일 경로.
        default: JSON에 값이 없거나 오류일 때 사용할 기본값.

    Returns:
        깊이 스케일(미터/단위).
    """
    base_dir = os.path.dirname(os.path.abspath(npy_path))
    json_path = os.path.join(base_dir, "device_stream_info.json")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        value = float(meta.get("depth_scale_m_per_unit", default))
        if not np.isfinite(value):
            return default
        return value
    except Exception:
        return default


# ----------------------------- 공개 API ------------------------------ #
def analyze_npy(
    npy_path: str,
    out_base_dir: str = "runs/npy_analysis_out",
    make_plots: bool = False,
) -> Dict[str, Any]:
    """NPY(depth_raw_frames.npy) 파일을 분석합니다.

    프레임별 통계 CSV를 생성합니다. (make_plots는 현재 미사용/placeholder)

    Args:
        npy_path: 분석할 depth_raw_frames.npy 경로
            (형상: (N, H, W) uint16).
        out_base_dir: 출력 베이스 폴더.
        make_plots: (옵션) 플롯 생성 여부. 현재 구현에서는 사용하지 않음.

    Returns:
        summary 딕셔너리.
            - type: "npy"
            - npy_path: 입력 NPY의 절대 경로
            - out_dir: 출력 폴더 절대 경로
            - frames: 프레임 수(N)
            - size: (W, H)
            - depth_scale_m_per_unit: 깊이 스케일(미터/단위)
            - csv_path: 통계 CSV 경로

    Raises:
        FileNotFoundError: 입력 파일이 없을 때.
        ValueError: 배열 차원이 (N, H, W)가 아닐 때.
    """
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"NPY not found: {npy_path}")

    out_dir = _ensure_timestamped_dir(out_base_dir)

    depth = np.load(npy_path, mmap_mode="r")  # (N, H, W) uint16
    if depth.ndim != 3:
        raise ValueError(
            "depth 배열 차원 오류: expected 3D (N, H, W), "
            f"got shape={depth.shape}"
        )

    num_frames, height, width = depth.shape
    depth_scale = _find_scale_from_json(npy_path, default=0.001)

    # --------------------------- 통계 CSV ---------------------------- #
    stats_csv = os.path.join(out_dir, "frame_stats_from_npy.csv")
    fieldnames = [
        "idx",
        "nz_ratio",
        "min_m",
        "p50_m",
        "p95_m",
        "max_m",
        "mean_m",
    ]

    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(num_frames):
            d = depth[i]
            nz = d[d > 0].astype(np.float32)

            if nz.size == 0:
                row = {
                    "idx": i,
                    "nz_ratio": 0.0,
                    "min_m": np.nan,
                    "p50_m": np.nan,
                    "p95_m": np.nan,
                    "max_m": np.nan,
                    "mean_m": np.nan,
                }
            else:
                dm = nz * depth_scale
                row = {
                    "idx": i,
                    "nz_ratio": float(nz.size) / d.size,
                    "min_m": float(dm.min()),
                    "p50_m": float(np.percentile(dm, 50)),
                    "p95_m": float(np.percentile(dm, 95)),
                    "max_m": float(dm.max()),
                    "mean_m": float(dm.mean()),
                }

            writer.writerow(row)

    # --------------------------- 요약 반환 --------------------------- #
    summary: Dict[str, Any] = {
        "type": "npy",
        "npy_path": os.path.abspath(npy_path),
        "out_dir": os.path.abspath(out_dir),
        "frames": int(num_frames),
        "size": (int(width), int(height)),
        "depth_scale_m_per_unit": float(depth_scale),
        "csv_path": os.path.abspath(stats_csv),
    }

    # (선택) make_plots 기능을 붙일 때 out_dir 하위에 저장하고,
    # summary에 figure 경로나 추가 산출물 경로를 넣어줍니다.
    # if make_plots:
    #     ...
    #     summary["plots"] = [abs_path1, abs_path2, ...]

    return summary


# ------------------------------- CLI -------------------------------- #
def _cli() -> None:
    """커맨드라인 인터페이스 진입점."""
    parser = argparse.ArgumentParser(
        description=(
            "NPY(depth_raw_frames.npy) 분석: 프레임별 통계 CSV 생성"
        )
    )
    parser.add_argument(
        "npy_path",
        help="분석할 depth_raw_frames.npy 경로",
    )
    parser.add_argument(
        "--out",
        default="runs/npy_analysis_out",
        help="출력 베이스 폴더",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="(옵션) 플롯 생성",
    )

    args = parser.parse_args()

    summary = analyze_npy(
        args.npy_path,
        out_base_dir=args.out,
        make_plots=bool(args.plots),
    )

    print("[NPY SUMMARY]")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _cli()

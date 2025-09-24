"""Frame range extraction utilities for recorded outputs and .bag files."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..core.paths import timestamped_subdir
from ..realsense import _import_rs, extrinsics_between, stream_intrinsics

__all__ = [
    "DEFAULT_CROP_BASE",
    "probe_npy_info",
    "probe_bag_info",
    "crop_saved_outputs",
    "crop_bag_frames",
]


DEFAULT_CROP_BASE = "save/crop_out"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ------------------------------- Probing --------------------------------- #

def probe_npy_info(npy_path: str) -> Dict[str, Any]:
    """Inspect a depth ``.npy`` file and return basic metadata."""

    npy_path = os.path.abspath(npy_path)
    if not os.path.isfile(npy_path):
        raise FileNotFoundError(f"NPY not found: {npy_path}")

    depth = np.load(npy_path, mmap_mode="r")
    if depth.ndim != 3:
        raise ValueError(
            "depth 배열 차원 오류: expected 3D (N, H, W), "
            f"got shape={depth.shape}"
        )

    frames, height, width = depth.shape
    dtype = str(depth.dtype)
    del depth

    return {
        "type": "npy",
        "path": npy_path,
        "frames": int(frames),
        "height": int(height),
        "width": int(width),
        "dtype": dtype,
    }


def probe_bag_info(bag_path: str) -> Dict[str, Any]:
    """Inspect a RealSense ``.bag`` file for stream dimensions and metadata."""

    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    rs = _import_rs()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    depth_info: Dict[str, Any] = {}
    color_info: Optional[Dict[str, Any]] = None
    depth_scale: Optional[float] = None
    duration_s: Optional[float] = None

    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        try:
            playback.set_real_time(False)
        except Exception:
            pass

        try:
            duration = playback.get_duration()
            duration_s = float(duration.total_seconds())
        except Exception:
            duration_s = None

        device = profile.get_device()
        depth_sensor = None
        for sensor in device.sensors:
            if sensor.is_depth_sensor():
                depth_sensor = sensor
                break
        if depth_sensor is not None:
            try:
                depth_scale = float(depth_sensor.get_depth_scale())
            except Exception:
                depth_scale = None

        try:
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            depth_info = {
                "width": int(depth_profile.width()),
                "height": int(depth_profile.height()),
                "fps": int(depth_profile.fps()),
            }
        except Exception:
            depth_info = {}

        try:
            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            color_info = {
                "width": int(color_profile.width()),
                "height": int(color_profile.height()),
                "fps": int(color_profile.fps()),
            }
        except Exception:
            color_info = None
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    frames_estimate: Optional[int] = None
    if duration_s and depth_info.get("fps"):
        frames_estimate = int(round(duration_s * depth_info["fps"]))

    return {
        "type": "bag",
        "path": bag_path,
        "depth": depth_info,
        "color": color_info,
        "depth_scale_m_per_unit": depth_scale,
        "duration_s": duration_s,
        "frames_estimate": frames_estimate,
    }


# ------------------------------ Internals -------------------------------- #

def _write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _resolve_frame_range(
    total_frames: Optional[int],
    frame_start: int,
    frame_end: Optional[int],
    frame_step: int,
) -> Tuple[int, Optional[int], int]:
    """Validate and normalize frame selection parameters."""

    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    start = int(frame_start)
    if start < 0:
        raise ValueError("frame_start must be >= 0")

    end = frame_end
    if end is not None:
        end = int(end)
        if end < 0:
            end = None

    if end is not None and end <= start:
        raise ValueError("frame_end must be greater than frame_start")

    if total_frames is not None:
        total_frames = int(total_frames)
        if start >= total_frames:
            raise ValueError(
                f"frame_start ({start})는 총 프레임 수({total_frames})보다 작아야 합니다."
            )
        if end is None or end > total_frames:
            end = total_frames

    return start, end, int(frame_step)


def _selected_indices(
    total_frames: int,
    frame_start: int,
    frame_end: Optional[int],
    frame_step: int,
) -> List[int]:
    stop = total_frames if frame_end is None else frame_end
    indices = list(range(frame_start, stop, frame_step))
    if not indices:
        raise ValueError("선택된 프레임이 없습니다. 범위를 다시 확인하세요.")
    return indices


def _iter_color_frames(source: str) -> Iterator[Tuple[np.ndarray, str]]:
    """Yield frames from a video file or directory of images."""

    source = os.path.abspath(source)
    if os.path.isdir(source):
        files = sorted(
            f
            for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        if not files:
            raise FileNotFoundError(f"No image files found in directory: {source}")
        for name in files:
            path = os.path.join(source, name)
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            yield img, name
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source}")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame, ""
        finally:
            cap.release()


def _save_color_frames_range(
    source: Optional[str],
    out_dir: str,
    indices: Sequence[int],
) -> Tuple[int, Optional[str]]:
    """Export selected frames from *source* into *out_dir*."""

    if not source or not indices:
        return 0, None

    source = os.path.abspath(source)
    if not os.path.exists(source):
        raise FileNotFoundError(f"Color source not found: {source}")

    os.makedirs(out_dir, exist_ok=True)
    mapping = {src_idx: out_idx for out_idx, src_idx in enumerate(indices)}
    last_idx = indices[-1]
    saved = 0

    for idx, (frame, _name) in enumerate(_iter_color_frames(source)):
        if idx > last_idx:
            break
        out_idx = mapping.get(idx)
        if out_idx is None:
            continue
        out_path = os.path.join(out_dir, f"frame_{out_idx:06d}.png")
        if not cv2.imwrite(out_path, frame):
            raise RuntimeError(f"Failed to write image: {out_path}")
        saved += 1

    if saved < len(indices):
        raise RuntimeError(
            f"Color source {source} ended early (expected {len(indices)} frames, saved {saved})."
        )

    return saved, out_dir


# ------------------------------ Public APIs ------------------------------ #

def crop_saved_outputs(
    depth_path: str,
    out_base_dir: str = DEFAULT_CROP_BASE,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
    *,
    color_source: Optional[str] = None,
    depth_color_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract a frame range from recorded depth outputs (and optional videos)."""

    depth_path = os.path.abspath(depth_path)
    if not os.path.isfile(depth_path):
        raise FileNotFoundError(f"depth_raw_frames.npy not found: {depth_path}")

    info = probe_npy_info(depth_path)
    total_frames = int(info["frames"])
    height = int(info["height"])
    width = int(info["width"])

    frame_start, frame_end, frame_step = _resolve_frame_range(
        total_frames, frame_start, frame_end, frame_step
    )
    indices = _selected_indices(total_frames, frame_start, frame_end, frame_step)
    actual_end = indices[-1] + 1

    depth = np.load(depth_path, mmap_mode="r")
    out_dir = timestamped_subdir(out_base_dir or DEFAULT_CROP_BASE)

    depth_raw_dir = os.path.join(out_dir, "depth_raw_frames")
    os.makedirs(depth_raw_dir, exist_ok=True)

    for out_idx, src_idx in enumerate(indices):
        frame = depth[src_idx]
        np.save(os.path.join(depth_raw_dir, f"frame_{out_idx:06d}.npy"), frame)

    del depth

    color_dir: Optional[str] = None
    color_saved = 0
    if color_source:
        color_dir = os.path.join(out_dir, "color_frames")
        color_saved, color_dir = _save_color_frames_range(
            color_source, color_dir, indices
        )

    depth_color_dir: Optional[str] = None
    depth_color_saved = 0
    if depth_color_source:
        depth_color_dir = os.path.join(out_dir, "depth_color_frames")
        depth_color_saved, depth_color_dir = _save_color_frames_range(
            depth_color_source, depth_color_dir, indices
        )

    summary: Dict[str, Any] = {
        "type": "crop_saved",
        "out_dir": os.path.abspath(out_dir),
        "depth_source": depth_path,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
        "actual_frame_end": actual_end,
        "frames": len(indices),
        "source_total_frames": total_frames,
        "depth_dtype": info.get("dtype"),
        "depth_raw_dir": depth_raw_dir,
        "color_source": os.path.abspath(color_source) if color_source else None,
        "color_frames": int(color_saved),
        "color_dir": color_dir,
        "depth_color_source": os.path.abspath(depth_color_source)
        if depth_color_source
        else None,
        "depth_color_frames": int(depth_color_saved),
        "depth_color_dir": depth_color_dir,
        "source_indices": list(map(int, indices)),
    }

    _write_json(os.path.join(out_dir, "crop_info.json"), summary)
    return summary


def crop_bag_frames(
    bag_path: str,
    out_base_dir: str = DEFAULT_CROP_BASE,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
    frame_step: int = 1,
) -> Dict[str, Any]:
    """Extract a frame range from a RealSense ``.bag`` recording."""

    bag_path = os.path.abspath(bag_path)
    if not os.path.isfile(bag_path):
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    frame_start, frame_end, frame_step = _resolve_frame_range(
        None, frame_start, frame_end, frame_step
    )

    rs = _import_rs()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    out_dir = timestamped_subdir(out_base_dir or DEFAULT_CROP_BASE)
    depth_raw_dir = os.path.join(out_dir, "depth_raw_frames")
    os.makedirs(depth_raw_dir, exist_ok=True)
    depth_color_dir = os.path.join(out_dir, "depth_color_frames")
    os.makedirs(depth_color_dir, exist_ok=True)

    color_dir: Optional[str] = None
    depth_frames: List[np.ndarray] = []
    selected_indices: List[int] = []
    depth_scale = 0.001
    depth_fps: Optional[int] = None
    color_saved = 0
    depth_color_saved = 0
    duration_s: Optional[float] = None

    align: Optional[Any] = None
    color_profile = None
    depth_profile = None

    try:
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        try:
            playback.set_real_time(False)
        except Exception:
            pass
        try:
            duration = playback.get_duration()
            duration_s = float(duration.total_seconds())
        except Exception:
            duration_s = None

        device = profile.get_device()
        depth_sensor = None
        for sensor in device.sensors:
            if sensor.is_depth_sensor():
                depth_sensor = sensor
                break
        if depth_sensor is not None:
            try:
                scale = float(depth_sensor.get_depth_scale())
                if np.isfinite(scale):
                    depth_scale = scale
            except Exception:
                pass

        depth_profile = profile.get_stream(rs.stream.depth)
        try:
            depth_vs = depth_profile.as_video_stream_profile()
            depth_fps = int(depth_vs.fps())
        except Exception:
            depth_fps = None

        try:
            color_profile = profile.get_stream(rs.stream.color)
            align = rs.align(rs.stream.color)
            color_dir = os.path.join(out_dir, "color_frames")
            os.makedirs(color_dir, exist_ok=True)
        except Exception:
            align = None
            color_profile = None
            color_dir = None

        colorizer = rs.colorizer()
        frame_idx = 0

        while True:
            if frame_end is not None and frame_idx >= frame_end:
                break
            try:
                frameset = pipeline.wait_for_frames()
            except RuntimeError:
                break

            if align is not None:
                frameset = align.process(frameset)

            depth_frame = frameset.get_depth_frame()
            if not depth_frame:
                frame_idx += 1
                continue

            if frame_idx < frame_start or ((frame_idx - frame_start) % frame_step != 0):
                frame_idx += 1
                continue

            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_frames.append(depth_raw.copy())
            out_idx = len(depth_frames) - 1

            np.save(os.path.join(depth_raw_dir, f"frame_{out_idx:06d}.npy"), depth_raw)

            depth_color = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            depth_color_path = os.path.join(depth_color_dir, f"frame_{out_idx:06d}.png")
            if not cv2.imwrite(depth_color_path, depth_color):
                raise RuntimeError(f"Failed to write image: {depth_color_path}")
            depth_color_saved += 1

            if color_dir is not None:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    color_img = np.asanyarray(color_frame.get_data())
                    color_path = os.path.join(color_dir, f"frame_{out_idx:06d}.png")
                    if not cv2.imwrite(color_path, color_img):
                        raise RuntimeError(f"Failed to write image: {color_path}")
                    color_saved += 1

            selected_indices.append(frame_idx)
            frame_idx += 1

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass

    if not depth_frames:
        raise RuntimeError("선택된 범위에 해당하는 프레임이 없습니다.")

    actual_end = selected_indices[-1] + 1

    summary: Dict[str, Any] = {
        "type": "crop_bag",
        "out_dir": os.path.abspath(out_dir),
        "bag_path": bag_path,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_step": frame_step,
        "actual_frame_end": actual_end,
        "frames": int(len(depth_frames)),
        "depth_raw_dir": depth_raw_dir,
        "depth_color_dir": depth_color_dir,
        "depth_color_frames": depth_color_saved,
        "color_dir": color_dir,
        "color_frames": color_saved,
        "depth_scale_m_per_unit": float(depth_scale),
        "duration_s": duration_s,
        "fps": depth_fps,
        "source_indices": list(map(int, selected_indices)),
    }

    _write_json(os.path.join(out_dir, "crop_info.json"), summary)
    return summary

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
    "summarize_crop_output",
]


DEFAULT_CROP_BASE = "save/crop_out"
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ----------------------------- Util helpers ------------------------------ #


def _frame_index_from_name(name: str) -> Optional[int]:
    """Extract the integer index from file names such as ``frame_000123``."""

    stem, _ = os.path.splitext(name)
    if not stem.startswith("frame_"):
        return None
    suffix = stem[6:]
    if not suffix:
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _format_indices(indices: Sequence[int], limit: int = 12) -> str:
    """Return a short preview string for a list of indices."""

    if not indices:
        return ""
    if len(indices) <= limit:
        return ", ".join(map(str, indices))
    head = ", ".join(map(str, indices[:limit]))
    return f"{head} … (+{len(indices) - limit})"


def _collect_frame_files(
    directory: str, allowed_exts: Sequence[str]
) -> Tuple[Dict[int, str], List[int], List[str]]:
    """Collect ``frame_XXXX`` files inside *directory*.

    Args:
        directory: Directory that may contain per-frame files.
        allowed_exts: Acceptable lowercase extensions including the dot.

    Returns:
        ``(files, indices, ignored)`` tuple where ``files`` maps frame indices
        to absolute paths, ``indices`` is the sorted list of indices and
        ``ignored`` contains names skipped because they do not follow the
        ``frame_`` pattern.
    """

    files: Dict[int, str] = {}
    indices: List[int] = []
    ignored: List[str] = []

    if not os.path.isdir(directory):
        return files, indices, ignored

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        _stem, ext = os.path.splitext(name)
        if allowed_exts and ext.lower() not in allowed_exts:
            continue
        idx = _frame_index_from_name(name)
        if idx is None:
            ignored.append(name)
            continue
        files[idx] = path
        indices.append(idx)

    indices.sort()
    return files, indices, ignored


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


def summarize_crop_output(out_dir: str) -> Dict[str, Any]:
    """Inspect a cropped output directory and highlight mismatches.

    The function checks ``depth_raw_frames`` ``.npy`` files and optionally the
    ``color_frames`` / ``depth_color_frames`` image folders. It also inspects the
    optional aggregated ``depth_raw_frames.npy`` file and ``crop_info.json``
    metadata when available.

    Args:
        out_dir: Crop output directory produced by :func:`crop_saved_outputs` or
            :func:`crop_bag_frames`.

    Returns:
        Dictionary summarizing the findings. Keys include ``depth``/``color``/
        ``depth_color`` sections describing counts, missing indices and whether
        the files match. ``consistent`` will be ``True`` only when every
        required asset is present and aligned.
    """

    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        raise NotADirectoryError(f"Crop output directory not found: {out_dir}")

    info_path = os.path.join(out_dir, "crop_info.json")
    info_data: Optional[Dict[str, Any]] = None
    if os.path.isfile(info_path):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info_data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            info_data = {"error": str(exc)}

    expected_frames: Optional[int] = None
    if isinstance(info_data, dict):
        frames_val = info_data.get("frames")
        if isinstance(frames_val, int) and frames_val >= 0:
            expected_frames = frames_val

    summary: Dict[str, Any] = {
        "out_dir": out_dir,
        "info_path": info_path if os.path.isfile(info_path) else None,
        "info": info_data,
        "expected_frames": expected_frames,
        "messages": [],
    }

    depth_dir = os.path.join(out_dir, "depth_raw_frames")
    depth_files, depth_indices, depth_ignored = _collect_frame_files(
        depth_dir, (".npy",)
    )
    depth_expected = (
        list(range(expected_frames))
        if expected_frames is not None
        else list(range(len(depth_indices)))
    )
    depth_missing = sorted(set(depth_expected) - set(depth_indices))
    depth_unexpected = sorted(set(depth_indices) - set(depth_expected))

    depth_summary = {
        "dir": depth_dir if os.path.isdir(depth_dir) else None,
        "count": len(depth_indices),
        "indices": depth_indices,
        "files": depth_files,
        "ignored": depth_ignored,
        "missing": depth_missing,
        "unexpected": depth_unexpected,
        "match": False,
    }
    summary["depth"] = depth_summary

    if not os.path.isdir(depth_dir):
        summary["messages"].append("depth_raw_frames 디렉터리를 찾을 수 없습니다.")
    elif not depth_indices:
        summary["messages"].append("depth_raw_frames 폴더에 frame_*.npy 파일이 없습니다.")
    depth_summary["match"] = (
        bool(depth_indices)
        and not depth_missing
        and not depth_unexpected
        and (
            expected_frames is None
            or len(depth_indices) == expected_frames
        )
    )

    color_dir = os.path.join(out_dir, "color_frames")
    color_files, color_indices, color_ignored = _collect_frame_files(
        color_dir, _IMAGE_EXTS
    )
    color_missing = sorted(set(depth_indices) - set(color_indices)) if depth_indices else []
    color_unexpected = sorted(set(color_indices) - set(depth_indices)) if depth_indices else []
    color_summary = {
        "dir": color_dir if os.path.isdir(color_dir) else None,
        "count": len(color_indices),
        "indices": color_indices,
        "files": color_files,
        "ignored": color_ignored,
        "missing": color_missing,
        "unexpected": color_unexpected,
        "match": None,
    }
    summary["color"] = color_summary
    if os.path.isdir(color_dir):
        color_summary["match"] = (
            bool(color_indices)
            and not color_missing
            and not color_unexpected
            and (len(color_indices) == len(depth_indices) if depth_indices else True)
        )
        if not color_summary["match"] and depth_indices:
            msg = "color_frames 폴더의 프레임 수가 depth와 일치하지 않습니다."
            if color_missing:
                msg += f" missing: { _format_indices(color_missing) }"
            if color_unexpected:
                msg += f" unexpected: { _format_indices(color_unexpected) }"
            summary["messages"].append(msg)
    elif depth_indices:
        summary["messages"].append(
            "color_frames 디렉터리가 없어 이미지 매칭을 확인할 수 없습니다."
        )

    depth_color_dir = os.path.join(out_dir, "depth_color_frames")
    depth_color_files, depth_color_indices, depth_color_ignored = _collect_frame_files(
        depth_color_dir, _IMAGE_EXTS
    )
    depth_color_missing = (
        sorted(set(depth_indices) - set(depth_color_indices)) if depth_indices else []
    )
    depth_color_unexpected = (
        sorted(set(depth_color_indices) - set(depth_indices)) if depth_indices else []
    )
    depth_color_summary = {
        "dir": depth_color_dir if os.path.isdir(depth_color_dir) else None,
        "count": len(depth_color_indices),
        "indices": depth_color_indices,
        "files": depth_color_files,
        "ignored": depth_color_ignored,
        "missing": depth_color_missing,
        "unexpected": depth_color_unexpected,
        "match": None,
    }
    summary["depth_color"] = depth_color_summary
    if os.path.isdir(depth_color_dir):
        depth_color_summary["match"] = (
            bool(depth_color_indices)
            and not depth_color_missing
            and not depth_color_unexpected
            and (len(depth_color_indices) == len(depth_indices) if depth_indices else True)
        )
        if not depth_color_summary["match"] and depth_indices:
            msg = "depth_color_frames 폴더의 프레임 수가 depth와 일치하지 않습니다."
            if depth_color_missing:
                msg += f" missing: { _format_indices(depth_color_missing) }"
            if depth_color_unexpected:
                msg += f" unexpected: { _format_indices(depth_color_unexpected) }"
            summary["messages"].append(msg)

    depth_npy_path = os.path.join(out_dir, "depth_raw_frames.npy")
    depth_npy_summary: Dict[str, Any] = {
        "path": depth_npy_path if os.path.isfile(depth_npy_path) else None,
        "shape": None,
        "frames": None,
        "match": None,
        "error": None,
    }
    summary["depth_npy"] = depth_npy_summary
    if os.path.isfile(depth_npy_path):
        try:
            depth_np = np.load(depth_npy_path, mmap_mode="r")
            depth_npy_summary["shape"] = tuple(int(x) for x in depth_np.shape)
            if depth_np.ndim == 3:
                depth_npy_summary["frames"] = int(depth_np.shape[0])
                depth_npy_summary["match"] = (
                    depth_indices
                    and int(depth_np.shape[0]) == len(depth_indices)
                )
            else:
                depth_npy_summary["match"] = False
                summary["messages"].append(
                    f"depth_raw_frames.npy 형상이 예상과 다릅니다: shape={depth_np.shape}"
                )
        except Exception as exc:  # noqa: BLE001
            depth_npy_summary["error"] = str(exc)
            summary["messages"].append(f"depth_raw_frames.npy 읽기 오류: {exc}")
        finally:
            try:
                del depth_np
            except Exception:
                pass

    consistent = depth_summary["match"]
    if depth_summary["dir"] is None:
        consistent = False

    if depth_indices:
        if color_summary["match"] is False:
            consistent = False
        if color_summary["dir"] is None:
            consistent = False
        if depth_color_summary["match"] is False:
            consistent = False
    if depth_npy_summary["match"] is False:
        consistent = False

    summary["consistent"] = bool(consistent)

    if depth_summary["ignored"]:
        summary["messages"].append(
            "다음 depth 파일은 frame_* 패턴이 아니므로 무시되었습니다: "
            + ", ".join(depth_summary["ignored"])
        )
    if color_summary["ignored"]:
        summary["messages"].append(
            "다음 color 파일은 frame_* 패턴이 아니므로 무시되었습니다: "
            + ", ".join(color_summary["ignored"])
        )
    if depth_color_summary["ignored"]:
        summary["messages"].append(
            "다음 depth_color 파일은 frame_* 패턴이 아니므로 무시되었습니다: "
            + ", ".join(depth_color_summary["ignored"])
        )

    return summary


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
# ----------------------------- Util helpers ------------------------------ #


def _frame_index_from_name(name: str) -> Optional[int]:
    """Extract integer frame index from file name like ``frame_000123``."""

    stem, _ext = os.path.splitext(name)
    if not stem.startswith("frame_"):
        return None
    idx_txt = stem[6:]
    if not idx_txt:
        return None
    try:
        return int(idx_txt)
    except ValueError:
        return None


def _format_indices(indices: Sequence[int], limit: int = 12) -> str:
    """Return a human readable preview of integer indices."""

    if not indices:
        return ""
    if len(indices) <= limit:
        return ", ".join(map(str, indices))
    head = ", ".join(map(str, indices[:limit]))
    return f"{head} … (+{len(indices) - limit})"


def _collect_frame_files(
    directory: str,
    allowed_exts: Sequence[str],
) -> Tuple[Dict[int, str], List[int], List[str]]:
    """Return mapping of ``frame_XXXX`` files in ``directory``.

    Args:
        directory: Directory to scan.
        allowed_exts: Allowed lowercase extensions including the leading dot.

    Returns:
        Tuple of ``(files_map, indices, ignored)`` where ``files_map`` maps frame
        indices to absolute file paths, ``indices`` is the sorted list of
        detected indices and ``ignored`` contains file names that did not match
        the ``frame_`` pattern.
    """

    files: Dict[int, str] = {}
    indices: List[int] = []
    ignored: List[str] = []

    if not os.path.isdir(directory):
        return files, indices, ignored

    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if allowed_exts and ext.lower() not in allowed_exts:
            continue
        idx = _frame_index_from_name(name)
        if idx is None:
            ignored.append(name)
            continue
        indices.append(idx)
        files[idx] = path

    indices.sort()
    return files, indices, ignored


# ------------------------------- Probing --------------------------------- #

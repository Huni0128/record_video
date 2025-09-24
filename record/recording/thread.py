# -*- coding: utf-8 -*-
"""
Intel RealSense 기록 쓰레드.

- Start 누르면 Stop 누를 때까지 프레임을 계속 수집합니다.
- 저장 산출물:
  - depth_raw_frames.npy : (N, H, W) uint16, 깊이 프레임만 저장
  - device_stream_info.json     : 장치/센서/내·외부 파라미터 + depth_scale
  - depth_colorized.mp4 / color.mp4 (컬러 센서가 있을 때)
  - depth_color_frames/frame_XXXXXX.png : 색상화된 depth 이미지 (선택)
  - color_frames/frame_XXXXXX.png : 컬러 이미지 (선택)
  - depth_raw_frames/frame_XXXXXX.npy : 프레임별 depth raw (선택)

Docstring 스타일: Google Style
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt5 import QtCore

from ..core import RecordConfig, fourcc_mp4v
from ..realsense import (
    _import_rs,
    extrinsics_between,
    sensor_options_to_dict,
    stream_intrinsics,
)


__all__ = ["RecordThread"]


class RecordThread(QtCore.QThread):
    """RealSense 기록용 QThread.

    Signals:
        sig_preview(np.ndarray, np.ndarray): depth(BGR colorized), color(BGR).
        sig_status(str): 진행 상태 메시지.
        sig_finished(str): 완료 시 출력 폴더 경로.
    """

    # 미리보기: depth(BGR 컬러라이즈) + color(BGR)
    sig_preview = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    sig_status = QtCore.pyqtSignal(str)
    sig_finished = QtCore.pyqtSignal(str)

    def __init__(self, cfg: RecordConfig, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self._stop = False

        # RealSense/비디오 핸들
        self.pipeline: Any = None
        self.depth_video: Optional[cv2.VideoWriter] = None
        self.color_video: Optional[cv2.VideoWriter] = None
        self.colorizer: Any = None

        # 메타/버퍼
        self.depth_scale: Optional[float] = None
        self.depth_raw_frames: List[np.ndarray] = []

        # 프레임 저장 관련
        self._frame_idx: int = 0
        self._frame_stride: int = max(1, int(self.cfg.frame_stride))
        self._depth_color_frame_dir: Optional[str] = None
        self._color_frame_dir: Optional[str] = None
        self._depth_raw_frame_dir: Optional[str] = None
        self._bag_path: Optional[str] = None
        if self.cfg.save_frames:
            self._depth_color_frame_dir = os.path.join(
                self.cfg.out_dir, "depth_color_frames"
            )
            os.makedirs(self._depth_color_frame_dir, exist_ok=True)
            self._color_frame_dir = os.path.join(self.cfg.out_dir, "color_frames")
            os.makedirs(self._color_frame_dir, exist_ok=True)
            self._depth_raw_frame_dir = os.path.join(
                self.cfg.out_dir, "depth_raw_frames"
            )
            os.makedirs(self._depth_raw_frame_dir, exist_ok=True)
        self._frame_save_error_reported: bool = False

        # 내부 상태
        self._color_enabled: bool = False
        self._color_size: Optional[Tuple[int, int]] = None  # (w, h)

        # 출력 폴더 보장
        os.makedirs(self.cfg.out_dir, exist_ok=True)

    # ------------------------------ Control ------------------------------ #
    def stop(self) -> None:
        """루프 종료 요청."""
        self._stop = True

    # ------------------------------ Internals ---------------------------- #
    def _try_enable_color(self, rs: Any, config: Any) -> None:
        """컬러 센서가 있으면 color 스트림을 활성화.

        해상도 후보군을 순서대로 시도합니다. 모두 실패하면 비활성 상태 유지.
        """
        candidates = [
            (self.cfg.width, self.cfg.height),
            (640, 480),
            (848, 480),
            (1280, 720),
            (1920, 1080),
        ]
        for w, h in candidates:
            try:
                config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, self.cfg.fps)
                self._color_enabled = True
                self._color_size = (w, h)
                return
            except Exception:
                continue
        self._color_enabled = False
        self._color_size = None

    # --------------------------------- Run -------------------------------- #
    def run(self) -> None:
        try:
            rs = _import_rs()

            self._frame_idx = 0
            self._frame_save_error_reported = False

            self.sig_status.emit("Starting pipeline...")
            self.pipeline = rs.pipeline()
            config = rs.config()

            # Depth 필수 + Color(가능하면). IR은 사용하지 않음.
            config.enable_stream(
                rs.stream.depth, self.cfg.width, self.cfg.height, rs.format.z16, self.cfg.fps
            )
            self._try_enable_color(rs, config)

            if self.cfg.save_bag:
                try:
                    bag_path = os.path.join(self.cfg.out_dir, "session.bag")
                    # 기존 파일이 남아있으면 제거
                    if os.path.exists(bag_path):
                        os.remove(bag_path)
                    config.enable_record_to_file(bag_path)
                    self._bag_path = bag_path
                    self.sig_status.emit(f"Recording .bag → {bag_path}")
                except Exception as exc:  # noqa: BLE001
                    self.sig_status.emit(f".bag enable failed: {exc}")
                    self._bag_path = None

            # 파이프라인 시작
            profile = self.pipeline.start(config)

            # 장치/센서 정보 수집
            device = profile.get_device()
            dev_info: Dict[str, str] = {
                "name": device.get_info(rs.camera_info.name)
                if device.supports(rs.camera_info.name)
                else "",
                "firmware_version": device.get_info(rs.camera_info.firmware_version)
                if device.supports(rs.camera_info.firmware_version)
                else "",
                "product_line": device.get_info(rs.camera_info.product_line)
                if device.supports(rs.camera_info.product_line)
                else "",
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor)
                if device.supports(rs.camera_info.usb_type_descriptor)
                else "",
            }

            sensors_dump: Dict[str, Dict[str, Any]] = {}
            depth_sensor = None
            for s in device.sensors:
                s_name = (
                    s.get_info(rs.camera_info.name)
                    if s.supports(rs.camera_info.name)
                    else "sensor"
                )
                sensors_dump[s_name] = sensor_options_to_dict(s)
                if s.is_depth_sensor():
                    depth_sensor = s

            # 깊이 스케일
            if depth_sensor is not None:
                try:
                    self.depth_scale = float(depth_sensor.get_depth_scale())
                except Exception:
                    self.depth_scale = None
            if self.depth_scale is None or not np.isfinite(self.depth_scale):
                self.depth_scale = 0.001  # m/unit 기본값(1 unit = 1 mm)

            # 스트림 프로파일
            depth_profile = profile.get_stream(rs.stream.depth)
            color_profile = None
            if self._color_enabled:
                try:
                    color_profile = profile.get_stream(rs.stream.color)
                except Exception:
                    color_profile = None
                    self._color_enabled = False
                    self._color_size = None

            intrinsics = {
                "depth": stream_intrinsics(depth_profile),
                "color": stream_intrinsics(color_profile) if color_profile else None,
            }
            extrinsics = {
                "color_to_depth": extrinsics_between(color_profile, depth_profile)
                if color_profile
                else None,
            }

            # 메타 저장
            meta_path = os.path.join(self.cfg.out_dir, "device_stream_info.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "device": dev_info,
                        "depth_scale_m_per_unit": self.depth_scale,
                        "sensors_options": sensors_dump,
                        "intrinsics": intrinsics,
                        "extrinsics": extrinsics,
                        "resolution": [self.cfg.width, self.cfg.height],
                        "fps": self.cfg.fps,
                        "color_enabled": bool(self._color_enabled),
                        "color_size": list(self._color_size) if self._color_size else None,
                        "bag_path": os.path.abspath(self._bag_path)
                        if self._bag_path
                        else None,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # 비디오 라이터
            if self.cfg.save_videos:
                self.depth_video = cv2.VideoWriter(
                    os.path.join(self.cfg.out_dir, "depth_colorized.mp4"),
                    fourcc_mp4v(),
                    self.cfg.fps,
                    (self.cfg.width, self.cfg.height),
                )
                if color_profile is not None:
                    try:
                        vs = color_profile.as_video_stream_profile()
                        c_w, c_h = vs.width(), vs.height()
                    except Exception:
                        c_w, c_h = self._color_size or (self.cfg.width, self.cfg.height)
                    self.color_video = cv2.VideoWriter(
                        os.path.join(self.cfg.out_dir, "color.mp4"),
                        fourcc_mp4v(),
                        self.cfg.fps,
                        (int(c_w), int(c_h)),
                    )

            # 컬러라이저
            self.colorizer = rs.colorizer()

            self.sig_status.emit(
                "Recording... (press Stop to end)"
                + ("" if not self._color_enabled else " [color enabled]")
            )

            # 메인 루프
            while not self._stop:
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame() if self._color_enabled else None
                if not depth_frame:
                    continue

                # Depth raw
                depth_raw = np.asanyarray(depth_frame.get_data())  # (H, W) uint16

                # 프리뷰용
                depth_color = np.asanyarray(
                    self.colorizer.colorize(depth_frame).get_data()
                )
                if color_frame is not None:
                    color_img = np.asanyarray(color_frame.get_data())  # (Hc, Wc, 3) BGR8
                else:
                    # 컬러가 없으면 depth 크기 기준으로 검정 화면 제공
                    color_img = np.zeros(
                        (self.cfg.height, self.cfg.width, 3), dtype=np.uint8
                    )

                frame_idx = self._frame_idx

                if self.cfg.save_frames and (frame_idx % self._frame_stride == 0):
                    frame_base = f"frame_{frame_idx:06d}"
                    try:
                        if self._depth_color_frame_dir is not None:
                            depth_path = os.path.join(
                                self._depth_color_frame_dir, f"{frame_base}.png"
                            )
                            cv2.imwrite(depth_path, depth_color)
                        if (
                            self._color_frame_dir is not None
                            and color_frame is not None
                        ):
                            color_path = os.path.join(
                                self._color_frame_dir, f"{frame_base}.png"
                            )
                            cv2.imwrite(color_path, color_img)
                        if self._depth_raw_frame_dir is not None:
                            raw_path = os.path.join(
                                self._depth_raw_frame_dir, f"{frame_base}.npy"
                            )
                            np.save(raw_path, depth_raw)
                    except Exception as exc:  # noqa: BLE001
                        if not self._frame_save_error_reported:
                            self._frame_save_error_reported = True
                            self.sig_status.emit(f"Frame save error: {exc}")

                # 비디오 저장
                if self.cfg.save_videos:
                    if self.depth_video is not None:
                        self.depth_video.write(depth_color)
                    if self.color_video is not None and color_frame is not None:
                        self.color_video.write(color_img)

                # NPY 저장 버퍼 (종료 시 일괄 저장)
                if self.cfg.save_depth_npy:
                    self.depth_raw_frames.append(depth_raw.copy())

                # 미리보기 송출
                self.sig_preview.emit(depth_color, color_img)

                self._frame_idx += 1

            self.sig_status.emit("Stopping...")

        except Exception as exc:  # noqa: BLE001
            self.sig_status.emit(f"Error: {exc}")

        finally:
            # 파이프라인/비디오 리소스 정리
            try:
                if self.pipeline:
                    self.pipeline.stop()
            except Exception:
                pass

            try:
                for writer in (self.depth_video, self.color_video):
                    if writer is not None:
                        writer.release()
            except Exception:
                pass

            # NPY 저장
            try:
                if self.cfg.save_depth_npy and self.depth_raw_frames:
                    npy_path = os.path.join(self.cfg.out_dir, "depth_raw_frames.npy")
                    arr = np.stack(self.depth_raw_frames, axis=0)  # (N, H, W) uint16
                    np.save(npy_path, arr)
            except Exception as exc:  # 저장 실패는 상태로 알림만
                self.sig_status.emit(f"NPY save error: {exc}")

            # 완료 알림
            self.sig_finished.emit(self.cfg.out_dir)

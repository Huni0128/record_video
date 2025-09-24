"""Widgets for verifying cropped depth/image frame consistency."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ..analysis.crop import summarize_crop_output
from .image import qimage_from_cv

__all__ = ["CropVerifyWidget"]


class CropVerifyWidget(QtWidgets.QWidget):
    """UI helper to inspect cropped outputs and preview frames."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self._summary: Optional[Dict[str, Any]] = None
        self._updating_index = False
        self._frame_positions: List[int] = []
        self._depth_pixmap: Optional[QtGui.QPixmap] = None
        self._color_pixmap: Optional[QtGui.QPixmap] = None

        self._build_ui()

    # ------------------------------------------------------------------ UI #
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.editOutDir = QtWidgets.QLineEdit()
        self.btnBrowse = QtWidgets.QPushButton("Browse…")
        self.btnBrowse.clicked.connect(self._on_browse)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.editOutDir, 1)
        hb.addWidget(self.btnBrowse)
        form.addRow("Crop output dir:", hb)

        self.btnLoad = QtWidgets.QPushButton("Load summary")
        self.btnLoad.clicked.connect(self._on_load_clicked)
        form.addRow("", self.btnLoad)

        layout.addLayout(form)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Depth frames:"), 0, 0)
        self.lblDepthStats = QtWidgets.QLabel("-")
        grid.addWidget(self.lblDepthStats, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Color frames:"), 1, 0)
        self.lblColorStats = QtWidgets.QLabel("-")
        grid.addWidget(self.lblColorStats, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Depth-color frames:"), 2, 0)
        self.lblDepthColorStats = QtWidgets.QLabel("-")
        grid.addWidget(self.lblDepthColorStats, 2, 1)

        grid.addWidget(QtWidgets.QLabel("depth_raw_frames.npy:"), 3, 0)
        self.lblNpyStats = QtWidgets.QLabel("-")
        grid.addWidget(self.lblNpyStats, 3, 1)

        layout.addLayout(grid)

        self.lblStatus = QtWidgets.QLabel("Status: -")
        font = self.lblStatus.font()
        font.setBold(True)
        self.lblStatus.setFont(font)
        layout.addWidget(self.lblStatus)

        preview_group = QtWidgets.QGroupBox("Preview")
        preview_layout = QtWidgets.QVBoxLayout(preview_group)

        images_layout = QtWidgets.QHBoxLayout()
        self.lblDepthPreview = QtWidgets.QLabel("Depth")
        self.lblDepthPreview.setAlignment(QtCore.Qt.AlignCenter)
        self.lblDepthPreview.setMinimumSize(280, 200)
        self.lblDepthPreview.setStyleSheet(
            "background:#202020; color:#dddddd; border:1px solid #444;"
        )
        images_layout.addWidget(self.lblDepthPreview, 1)

        self.lblColorPreview = QtWidgets.QLabel("Color")
        self.lblColorPreview.setAlignment(QtCore.Qt.AlignCenter)
        self.lblColorPreview.setMinimumSize(280, 200)
        self.lblColorPreview.setStyleSheet(
            "background:#202020; color:#dddddd; border:1px solid #444;"
        )
        images_layout.addWidget(self.lblColorPreview, 1)

        preview_layout.addLayout(images_layout)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Frame index:"))
        self.spinFrame = QtWidgets.QSpinBox()
        self.spinFrame.setRange(0, 0)
        self.spinFrame.valueChanged.connect(self._on_spin_changed)
        controls.addWidget(self.spinFrame)

        self.sliderFrame = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderFrame.setRange(0, 0)
        self.sliderFrame.setEnabled(False)
        self.sliderFrame.valueChanged.connect(self._on_slider_changed)
        controls.addWidget(self.sliderFrame, 1)

        preview_layout.addLayout(controls)

        self.lblFrameInfo = QtWidgets.QLabel("-")
        self.lblFrameInfo.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        preview_layout.addWidget(self.lblFrameInfo)

        layout.addWidget(preview_group, 1)

        self.txtLog = QtWidgets.QTextEdit()
        self.txtLog.setReadOnly(True)
        self.txtLog.setPlaceholderText("Verification details will appear here…")
        layout.addWidget(self.txtLog, 1)

        layout.setStretchFactor(preview_group, 2)
        layout.setStretchFactor(self.txtLog, 1)

    # --------------------------------------------------------------- helpers #
    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._apply_pixmap(self.lblDepthPreview, self._depth_pixmap)
        self._apply_pixmap(self.lblColorPreview, self._color_pixmap)

    def _apply_pixmap(
        self, label: QtWidgets.QLabel, pixmap: Optional[QtGui.QPixmap]
    ) -> None:
        if pixmap is None or label.width() <= 0 or label.height() <= 0:
            label.clear()
            if pixmap is None:
                label.setText("No image")
            return
        scaled = pixmap.scaled(
            label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        label.setPixmap(scaled)

    def _on_browse(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select crop output directory", ""
        )
        if directory:
            self.load_directory(directory)

    def _on_load_clicked(self) -> None:
        self.load_directory(self.editOutDir.text().strip())

    def load_directory(self, directory: str) -> None:
        directory = directory.strip()
        if not directory:
            QtWidgets.QMessageBox.information(
                self, "Verify", "Select a crop output directory first."
            )
            return
        self.editOutDir.setText(directory)
        try:
            summary = summarize_crop_output(directory)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Verify", str(exc))
            return

        self._summary = summary
        depth_info = summary.get("depth", {})
        self._frame_positions = list(depth_info.get("indices", []))
        self.sliderFrame.setEnabled(bool(self._frame_positions))
        self.sliderFrame.setMaximum(max(0, len(self._frame_positions) - 1))
        self.spinFrame.setEnabled(bool(self._frame_positions))
        self.spinFrame.setMaximum(max(0, len(self._frame_positions) - 1))
        self.spinFrame.setValue(0 if self._frame_positions else 0)
        self.sliderFrame.setValue(0)

        self._depth_pixmap = None
        self._color_pixmap = None
        self.lblDepthPreview.clear()
        self.lblColorPreview.clear()
        self.lblDepthPreview.setText("Depth")
        self.lblColorPreview.setText("Color")
        self.lblFrameInfo.setText("-")

        self._update_labels()
        if self._frame_positions:
            self._show_frame_at(0)

    def _status_text(self, ok: Optional[bool], text: str) -> str:
        if ok is True:
            return f"✅ {text}"
        if ok is False:
            return f"❌ {text}"
        return f"⚠️ {text}"

    def _update_labels(self) -> None:
        if not self._summary:
            return

        depth = self._summary.get("depth", {})
        depth_txt = f"{depth.get('count', 0)} files"
        expected = self._summary.get("expected_frames")
        if expected is not None:
            depth_txt += f" / expected {expected}"
        self.lblDepthStats.setText(self._status_text(depth.get("match"), depth_txt))

        color = self._summary.get("color", {})
        color_txt = f"{color.get('count', 0)} files"
        if depth.get("count"):
            color_txt += f" (depth={depth.get('count')})"
        color_match = color.get("match") if color.get("dir") else None
        self.lblColorStats.setText(self._status_text(color_match, color_txt))

        depth_color = self._summary.get("depth_color", {})
        depth_color_txt = f"{depth_color.get('count', 0)} files"
        depth_color_match = (
            depth_color.get("match") if depth_color.get("dir") else None
        )
        self.lblDepthColorStats.setText(
            self._status_text(depth_color_match, depth_color_txt)
        )

        depth_npy = self._summary.get("depth_npy", {})
        if depth_npy.get("path"):
            npy_txt = str(depth_npy.get("shape"))
            self.lblNpyStats.setText(self._status_text(depth_npy.get("match"), npy_txt))
        else:
            self.lblNpyStats.setText(self._status_text(None, "not found"))

        overall = self._summary.get("consistent")
        status_txt = "Files are consistent." if overall else "Mismatch detected."
        self.lblStatus.setText(self._status_text(overall, status_txt))

        self.txtLog.clear()
        info = self._summary.get("info")
        if isinstance(info, dict) and info:
            frames = info.get("frames")
            src_indices = info.get("source_indices")
            msg = f"crop_info.json frames={frames}"
            if isinstance(src_indices, list) and src_indices:
                msg += f", source indices: {src_indices[0]} … {src_indices[-1]}"
            self.txtLog.append(msg)
        for msg in self._summary.get("messages", []):
            self.txtLog.append(msg)
        if not self._summary.get("messages"):
            self.txtLog.append("No mismatches detected.")

    # --------------------------------------------------------------- preview #
    def _on_slider_changed(self, value: int) -> None:
        if self._updating_index:
            return
        self._updating_index = True
        self.spinFrame.setValue(value)
        self._updating_index = False
        self._show_frame_at(value)

    def _on_spin_changed(self, value: int) -> None:
        if self._updating_index:
            return
        self._updating_index = True
        self.sliderFrame.setValue(value)
        self._updating_index = False
        self._show_frame_at(value)

    def _show_frame_at(self, position: int) -> None:
        if not self._summary or position < 0:
            return
        if position >= len(self._frame_positions):
            return
        frame_index = self._frame_positions[position]
        depth = self._summary.get("depth", {})
        depth_path = depth.get("files", {}).get(frame_index)
        color_path: Optional[str] = None
        color = self._summary.get("color", {})
        if color.get("dir"):
            color_path = color.get("files", {}).get(frame_index)
        if not color_path:
            depth_color = self._summary.get("depth_color", {})
            if depth_color.get("dir"):
                color_path = depth_color.get("files", {}).get(frame_index)

        depth_img_txt = "Depth"
        color_img_txt = "Color"

        if depth_path and os.path.isfile(depth_path):
            try:
                depth_data = np.load(depth_path, allow_pickle=False)
                depth_bgr = self._depth_to_bgr(depth_data)
                qimg = qimage_from_cv(depth_bgr)
                pix = QtGui.QPixmap.fromImage(qimg)
                self._depth_pixmap = pix
                self._apply_pixmap(self.lblDepthPreview, pix)
                depth_img_txt = (
                    f"Depth: {os.path.basename(depth_path)} {depth_data.shape}"
                )
            except Exception as exc:  # noqa: BLE001
                self._depth_pixmap = None
                self.lblDepthPreview.clear()
                depth_img_txt = f"Depth load error: {exc}"
        else:
            self._depth_pixmap = None
            self.lblDepthPreview.clear()
        if self._depth_pixmap is None:
            self.lblDepthPreview.setText("No depth frame")

        if color_path and os.path.isfile(color_path):
            try:
                color_img = cv2.imread(color_path, cv2.IMREAD_COLOR)
                if color_img is None:
                    raise RuntimeError("Failed to read image")
                qimg = qimage_from_cv(color_img)
                pix = QtGui.QPixmap.fromImage(qimg)
                self._color_pixmap = pix
                self._apply_pixmap(self.lblColorPreview, pix)
                h, w = color_img.shape[:2]
                color_img_txt = (
                    f"Color: {os.path.basename(color_path)} {w}x{h}"  # width x height
                )
            except Exception as exc:  # noqa: BLE001
                self._color_pixmap = None
                self.lblColorPreview.clear()
                color_img_txt = f"Color load error: {exc}"
        else:
            self._color_pixmap = None
            self.lblColorPreview.clear()
        if self._color_pixmap is None:
            self.lblColorPreview.setText("No color frame")

        self.lblFrameInfo.setText(
            f"Frame {position} (source index {frame_index})\n{depth_img_txt}\n{color_img_txt}"
        )

    def _depth_to_bgr(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth frame, got shape={depth.shape}")
        depth = depth.astype(np.float32)
        valid = depth > 0
        if np.any(valid):
            vmin = float(depth[valid].min())
            vmax = float(depth[valid].max())
        else:
            vmin = float(depth.min())
            vmax = float(depth.max())
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        depth_clip = np.clip(depth, vmin, vmax)
        norm = (depth_clip - vmin) / (vmax - vmin)
        depth_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
        return colored


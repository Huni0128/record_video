# -*- coding: utf-8 -*-
"""메인 윈도우 로직과 분석 실행 스레드를 포함한 PyQt5 GUI 모듈."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from ..analysis import DEFAULT_ANALYSIS_BASE, analyze_npy as run_analyze_npy
from ..analysis.crop import (
    DEFAULT_CROP_BASE,
    crop_bag_frames,
    crop_saved_outputs,
    probe_bag_info,
    probe_npy_info,
)
from ..core import DEFAULT_BASE_DIR, RecordConfig, ensure_out_dir
from ..recording import RecordThread
from .crop_verify import CropVerifyWidget
from .image import qimage_from_cv
from .threads import FuncThread
from .viewer import NpyViewerWidget


__all__ = ["MainWindow"]


class MainWindow(QtWidgets.QMainWindow):
    """애플리케이션의 메인 윈도우."""

    def __init__(self) -> None:
        super().__init__()

        ui_path = Path(__file__).resolve().parents[2] / "ui" / "main_window.ui"
        uic.loadUi(str(ui_path), self)

        # ---- Record 버튼/라벨 초기화 ----
        if hasattr(self, "btnStart"):
            self.btnStart.setText("Record")
            self.btnStart.clicked.connect(self.on_start)
        if hasattr(self, "btnStop"):
            self.btnStop.clicked.connect(self.on_stop)
        if hasattr(self, "btnBrowse"):
            self.btnBrowse.clicked.connect(self.on_browse)

        if hasattr(self, "editOutDir"):
            self.editOutDir.setText("save")
        if hasattr(self, "comboRes"):
            self.comboRes.setCurrentText("1280x720")
        if hasattr(self, "comboFps"):
            self.comboFps.setCurrentText("15")
        if hasattr(self, "chkSaveDepthRaw"):
            self.chkSaveDepthRaw.setText("Save Depth (.npy)")
        if hasattr(self, "chkSaveBag"):
            self.chkSaveBag.setText("Save .bag")
            self.chkSaveBag.setChecked(False)
        if hasattr(self, "chkSaveFrames"):
            self.chkSaveFrames.setText("Save Frames (images + depth .npy)")
            self.chkSaveFrames.setChecked(True)
        if hasattr(self, "spinFrameStep"):
            self.spinFrameStep.setMinimum(1)
            self.spinFrameStep.setMaximum(120)
            self.spinFrameStep.setValue(1)
        self.rec_thread: Optional[RecordThread] = None
        self._is_recording: bool = False

        if hasattr(self, "chkSaveFrames") and hasattr(self, "spinFrameStep"):
            self.spinFrameStep.setEnabled(self.chkSaveFrames.isChecked())
            self.chkSaveFrames.toggled.connect(self._on_save_frames_toggled)

        # ---- Viewer (NPY 임베드) ----
        self._viewer: Optional[NpyViewerWidget] = None
        self._viewerDock: Optional[QtWidgets.QDockWidget] = None
        self._last_csv: Optional[str] = None
        self._npy_thread: Optional[FuncThread] = None
        self._crop_thread: Optional[FuncThread] = None
        self._last_crop_out: Optional[str] = None
        self._saved_depth_shape: Optional[Tuple[int, int, int]] = None
        self._bag_info: Optional[Dict[str, Any]] = None
        self._crop_context: Optional[str] = None
        self._crop_running: bool = False
        self._crop_verify_widget: Optional[CropVerifyWidget] = None

        self._build_analysis_ui()

    # ------------------------------ Record ------------------------------ #
    def set_status(self, text: str) -> None:
        """상태 바/라벨에 상태 텍스트를 표시합니다."""
        if hasattr(self, "lblStatus"):
            self.lblStatus.setText(f"Status: {text}")
        if self.statusBar():
            self.statusBar().showMessage(text, 5000)

    def set_running(self, running: bool) -> None:
        """녹화 중에는 Stop만 활성화, 나머지는 비활성화."""
        self._is_recording = running
        widget_names = (
            "btnStart",
            "btnStop",
            "btnBrowse",
            "editOutDir",
            "comboRes",
            "comboFps",
            "chkSaveVideos",
            "chkSaveDepthRaw",
            "chkSaveBag",
            "chkSaveFrames",
            "spinFrameStep",
        )
        for name in widget_names:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if name in ("btnStart", "btnBrowse"):
                widget.setEnabled(not running)
            elif name == "btnStop":
                widget.setEnabled(running)
            elif name == "spinFrameStep":
                chk = getattr(self, "chkSaveFrames", None)
                enabled = not running and (chk.isChecked() if chk else True)
                widget.setEnabled(enabled)
            else:
                widget.setEnabled(not running)

    @QtCore.pyqtSlot(bool)
    def _on_save_frames_toggled(self, checked: bool) -> None:
        """Save Frames 옵션 토글 시 프레임 간격 스핀박스 활성화 상태를 갱신."""

        if hasattr(self, "spinFrameStep"):
            self.spinFrameStep.setEnabled(checked and not self._is_recording)

    def on_browse(self) -> None:
        """출력 베이스 디렉터리 선택."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Base Directory", ""
        )
        if directory and hasattr(self, "editOutDir"):
            self.editOutDir.setText(directory)

    def on_start(self) -> None:
        """녹화 시작."""
        try:
            width, height = map(int, self.comboRes.currentText().split("x"))
            fps = int(self.comboFps.currentText())

            out_dir = ensure_out_dir(self.editOutDir.text())

            save_videos = (
                self.chkSaveVideos.isChecked()
                if hasattr(self, "chkSaveVideos")
                else True
            )
            save_depth_npy = (
                self.chkSaveDepthRaw.isChecked()
                if hasattr(self, "chkSaveDepthRaw")
                else True
            )
            save_frames = (
                self.chkSaveFrames.isChecked()
                if hasattr(self, "chkSaveFrames")
                else False
            )
            save_bag = (
                self.chkSaveBag.isChecked()
                if hasattr(self, "chkSaveBag")
                else False
            )
            frame_stride = (
                max(1, int(self.spinFrameStep.value()))
                if hasattr(self, "spinFrameStep")
                else 1
            )

            cfg = RecordConfig(
                out_dir=out_dir,
                width=width,
                height=height,
                fps=fps,
                save_videos=save_videos,
                save_depth_npy=save_depth_npy,
                save_frames=save_frames,
                save_bag=save_bag,
                frame_stride=frame_stride,
            )

            self.rec_thread = RecordThread(cfg)
            self.rec_thread.sig_preview.connect(self.update_previews)
            self.rec_thread.sig_status.connect(self.set_status)
            self.rec_thread.sig_finished.connect(self.on_finished)
            self.rec_thread.start()

            self.set_running(True)
            self.set_status("Recording.")
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Start Error", str(exc))
            self.set_running(False)

    def on_stop(self) -> None:
        """녹화 중지."""
        if self.rec_thread and self.rec_thread.isRunning():
            self.rec_thread.stop()

    @QtCore.pyqtSlot(str)
    def on_finished(self, out_dir: str) -> None:
        """녹화 종료 후 후처리."""
        self.set_running(False)
        self.set_status(f"Done. Output: {out_dir}")
        QtWidgets.QMessageBox.information(
            self,
            "Finished",
            f"저장이 완료되었습니다.\n\n{out_dir}",
        )

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def update_previews(
        self, depth_bgr: np.ndarray, color_bgr: np.ndarray
    ) -> None:
        """미리보기(Depth/Color) 위젯 갱신."""
        if hasattr(self, "viewDepth"):
            qimg_depth = qimage_from_cv(depth_bgr)
            self._set_pixmap_scaled(self.viewDepth, qimg_depth)

        if hasattr(self, "viewColor"):
            qimg_color = qimage_from_cv(color_bgr)
            self._set_pixmap_scaled(self.viewColor, qimg_color)

    def _set_pixmap_scaled(
        self, label: QtWidgets.QLabel, qimg: QtGui.QImage
    ) -> None:
        """라벨에 QImage를 라벨 크기에 맞춰 비율 유지로 표시."""
        pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
            label.width(),
            label.height(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    # ----------------------------- Qt Events ----------------------------- #
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """윈도우 닫힐 때 백그라운드 스레드 정리."""
        try:
            if self.rec_thread and self.rec_thread.isRunning():
                self.rec_thread.stop()
                self.rec_thread.wait(2000)
        except Exception:
            pass
        event.accept()

    # ---------------------------- Analysis UI ---------------------------- #
    def _build_analysis_ui(self) -> None:
        """NPY 뷰어/분석 도크 UI 구성."""
        # 뷰어 임베드(placeholder가 있으면 그 위치에 삽입)
        placeholder = getattr(self, "viewerNpyContainer", None)
        if isinstance(placeholder, QtWidgets.QWidget):
            self._viewer = NpyViewerWidget(self)
            layout = placeholder.layout()
            if layout is None:
                layout = QtWidgets.QVBoxLayout(placeholder)
                placeholder.setLayout(layout)
            layout.addWidget(self._viewer)
        else:
            self._viewer = NpyViewerWidget(self)
            self._viewerDock = QtWidgets.QDockWidget("Viewer (NPY)", self)
            self._viewerDock.setWidget(self._viewer)
            self._viewerDock.setObjectName("dockViewerNpy")
            self._viewerDock.setAllowedAreas(
                QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
            )
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._viewerDock)

        # 분석/크롭 도크
        dock = QtWidgets.QDockWidget("Analysis / Crop", self)
        dock.setObjectName("dockAnalysisTools")
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        tabs = QtWidgets.QTabWidget(container)
        tabs.setObjectName("tabAnalysisTools")
        container_layout.addWidget(tabs)

        analysis_tab = self._create_npy_analysis_tab()
        tabs.addTab(analysis_tab, "Analyze NPY")

        crop_tab = self._create_crop_tab()
        tabs.addTab(crop_tab, "Crop Frames")

        dock.setWidget(container)
        self._analysis_tabs = tabs

    # ----------------------- Helpers for analysis dock -------------------- #
    def _create_npy_analysis_tab(self) -> QtWidgets.QWidget:
        """기존 NPY 분석 UI를 생성하여 탭으로 반환."""

        widget = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(widget)

        btn_latest = QtWidgets.QPushButton("Find Latest Session")
        btn_latest.clicked.connect(self._fill_latest_session_paths)
        vbox.addWidget(btn_latest)

        form = QtWidgets.QFormLayout()
        self.editNpyPath = QtWidgets.QLineEdit()
        btnBrowseNpy = QtWidgets.QPushButton("Browse...")
        btnBrowseNpy.clicked.connect(self._browse_npy)

        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.editNpyPath, 1)
        hb.addWidget(btnBrowseNpy)
        form.addRow("NPY file:", hb)

        self.btnAnalyzeNpy = QtWidgets.QPushButton("Analyze .npy")
        self.btnAnalyzeNpy.clicked.connect(self._run_analyze_npy)
        form.addRow("", self.btnAnalyzeNpy)

        vbox.addLayout(form)

        gb_out = QtWidgets.QGroupBox("Results / Open")
        grid = QtWidgets.QGridLayout(gb_out)

        self.lblOutDir = QtWidgets.QLabel("-")
        self.btnOpenOut = QtWidgets.QPushButton("Open Output Folder")
        self.btnOpenOut.clicked.connect(self._open_out_dir)
        self.btnOpenOut.setEnabled(False)

        self.btnOpenCSV = QtWidgets.QPushButton("Open CSV")
        self.btnOpenCSV.clicked.connect(self._open_csv)
        self.btnOpenCSV.setEnabled(False)

        grid.addWidget(QtWidgets.QLabel("Output:"), 0, 0)
        grid.addWidget(self.lblOutDir, 0, 1)
        grid.addWidget(self.btnOpenOut, 1, 0)
        grid.addWidget(self.btnOpenCSV, 1, 1)
        vbox.addWidget(gb_out)

        self.txtAnalyzeLog = QtWidgets.QTextEdit()
        self.txtAnalyzeLog.setReadOnly(True)
        self.txtAnalyzeLog.setPlaceholderText(
            "Analysis logs will appear here..."
        )
        vbox.addWidget(self.txtAnalyzeLog, 1)

        return widget

    def _append_log(self, msg: str) -> None:
        """분석 로그 텍스트 박스에 메시지 추가."""
        if hasattr(self, "txtAnalyzeLog"):
            self.txtAnalyzeLog.append(msg)

    def _browse_npy(self) -> None:
        """NPY 파일 선택."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select .npy", "", "NumPy NPY (*.npy)"
        )
        if path and hasattr(self, "editNpyPath"):
            self.editNpyPath.setText(path)

    def _fill_latest_session_paths(self) -> None:
        """가장 최근 세션 폴더를 찾아 NPY 경로 채우기 및 뷰어 로드."""
        base = DEFAULT_BASE_DIR
        if not os.path.isdir(base):
            QtWidgets.QMessageBox.information(
                self, "Info", f"세션 폴더가 없습니다: {base}"
            )
            return

        subdirs = [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ]
        if not subdirs:
            QtWidgets.QMessageBox.information(
                self, "Info", f"세션 폴더가 없습니다: {base}"
            )
            return

        latest = max(subdirs, key=os.path.getmtime)
        npy_path = os.path.join(latest, "depth_raw_frames.npy")
        if os.path.isfile(npy_path):
            self.editNpyPath.setText(npy_path)
            try:
                self._load_in_viewer()
            except Exception:
                pass

        self._append_log(f"[Latest] {latest}")

    def _disable_analyze_ui(self, disable: bool) -> None:
        """분석 UI 비활성/활성 제어."""
        for widget in (getattr(self, "btnAnalyzeNpy", None),
                       getattr(self, "editNpyPath", None)):
            if widget is not None:
                widget.setEnabled(not disable)

    def _open_out_dir(self) -> None:
        """출력 폴더 열기."""
        path = self.lblOutDir.text()
        if path and path != "-":
            self._open_path(path)

    def _open_csv(self) -> None:
        """CSV 열기."""
        if self._last_csv:
            self._open_path(self._last_csv)

    def _open_path(self, path: str) -> None:
        """플랫폼별 기본 앱으로 경로 열기."""
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Open Error", str(exc))

    def _run_analyze_npy(self) -> None:
        """NPY 분석 실행."""
        npy_path = self.editNpyPath.text().strip()
        make_plots = True

        if not npy_path:
            QtWidgets.QMessageBox.information(
                self, "Info", "분석할 .npy 파일을 선택하세요."
            )
            return

        self._append_log(f"[NPY] Start: {npy_path} (plots={make_plots})")
        self._disable_analyze_ui(True)

        self._npy_thread = FuncThread(
            run_analyze_npy, npy_path, DEFAULT_ANALYSIS_BASE, make_plots
        )
        self._npy_thread.sig_done.connect(self._on_npy_done)
        self._npy_thread.sig_error.connect(self._on_analyze_error)
        self._npy_thread.start()

    def _on_npy_done(self, summary: Dict[str, Any]) -> None:
        """분석 완료 처리."""
        self._disable_analyze_ui(False)

        out_dir = summary.get("out_dir", "-")
        self.lblOutDir.setText(out_dir)

        self._last_csv = summary.get("csv_path")
        self.btnOpenOut.setEnabled(True)
        self.btnOpenCSV.setEnabled(bool(self._last_csv))

        self._append_log("[NPY SUMMARY]")
        for key, value in summary.items():
            self._append_log(f"  {key}: {value}")

    def _on_analyze_error(self, msg: str) -> None:
        """분석 오류 처리."""
        self._disable_analyze_ui(False)
        QtWidgets.QMessageBox.critical(self, "Analyze Error", msg)
        self._append_log(f"[ERROR] {msg}")

    def _load_in_viewer(self) -> None:
        """선택된 NPY를 뷰어에 로드."""
        npy_path = self.editNpyPath.text().strip()
        if not npy_path:
            QtWidgets.QMessageBox.information(
                self, "Info", "열 .npy 파일을 선택하세요."
            )
            return

        try:
            if self._viewer is None:
                raise RuntimeError("Viewer가 초기화되지 않았습니다.")
            self._viewer.load_file(npy_path)
            self._append_log(f"[VIEW] loaded: {npy_path}")

            if self._viewerDock is not None:
                self._viewerDock.show()
                self._viewerDock.raise_()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Viewer Error", str(exc))
            self._append_log(f"[ERROR] viewer: {exc}")

    # ---------------------------- Crop UI helpers --------------------------- #
    def _create_crop_tab(self) -> QtWidgets.QWidget:
        """크롭 관련 탭 컨테이너를 생성."""

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self._crop_tabs = QtWidgets.QTabWidget()
        self._crop_tabs.setObjectName("tabCropModes")
        layout.addWidget(self._crop_tabs)

        saved_tab = self._create_crop_saved_tab()
        self._crop_tabs.addTab(saved_tab, "Saved outputs")

        bag_tab = self._create_crop_bag_tab()
        self._crop_tabs.addTab(bag_tab, "From .bag")

        verify_tab = self._create_crop_verify_tab()
        self._crop_tabs.addTab(verify_tab, "Verify output")

        self.txtCropLog = QtWidgets.QTextEdit()
        self.txtCropLog.setReadOnly(True)
        self.txtCropLog.setPlaceholderText(
            "Crop logs will appear here..."
        )
        layout.addWidget(self.txtCropLog, 1)

        gb_out = QtWidgets.QGroupBox("Output")
        grid = QtWidgets.QGridLayout(gb_out)

        self.lblCropOut = QtWidgets.QLabel("-")
        self.lblCropOut.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.btnCropOpenOut = QtWidgets.QPushButton("Open Output Folder")
        self.btnCropOpenOut.clicked.connect(self._open_crop_out)
        self.btnCropOpenOut.setEnabled(False)

        grid.addWidget(QtWidgets.QLabel("Last output:"), 0, 0)
        grid.addWidget(self.lblCropOut, 0, 1)
        grid.addWidget(self.btnCropOpenOut, 1, 0, 1, 2)
        layout.addWidget(gb_out)

        return widget

    def _create_crop_verify_tab(self) -> QtWidgets.QWidget:
        """Create the verification tab that checks cropped outputs."""

        self._crop_verify_widget = CropVerifyWidget(self)
        return self._crop_verify_widget

    def _create_crop_saved_tab(self) -> QtWidgets.QWidget:
        """저장된 산출물에서 크롭하는 탭 UI 구성."""

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()

        self.editCropDepthPath = QtWidgets.QLineEdit()
        self.editCropDepthPath.editingFinished.connect(self._update_saved_depth_info)
        self.btnBrowseCropDepth = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropDepth.clicked.connect(self._browse_crop_depth)
        hb_depth = QtWidgets.QHBoxLayout()
        hb_depth.addWidget(self.editCropDepthPath, 1)
        hb_depth.addWidget(self.btnBrowseCropDepth)
        form.addRow("Depth NPY:", hb_depth)

        self.lblCropSavedInfo = QtWidgets.QLabel("Depth: -")
        self.lblCropSavedInfo.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        form.addRow("Info:", self.lblCropSavedInfo)

        self.editCropColorPath = QtWidgets.QLineEdit()
        self.btnBrowseCropColor = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropColor.clicked.connect(self._browse_crop_color)
        hb_color = QtWidgets.QHBoxLayout()
        hb_color.addWidget(self.editCropColorPath, 1)
        hb_color.addWidget(self.btnBrowseCropColor)
        form.addRow("Color video/dir:", hb_color)

        self.editCropDepthColorPath = QtWidgets.QLineEdit()
        self.btnBrowseCropDepthColor = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropDepthColor.clicked.connect(self._browse_crop_depth_color)
        hb_depth_color = QtWidgets.QHBoxLayout()
        hb_depth_color.addWidget(self.editCropDepthColorPath, 1)
        hb_depth_color.addWidget(self.btnBrowseCropDepthColor)
        form.addRow("Depth color video/dir:", hb_depth_color)

        self.editCropOutDir = QtWidgets.QLineEdit(DEFAULT_CROP_BASE)
        self.btnBrowseCropOut = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropOut.clicked.connect(self._browse_crop_saved_out)
        hb_out = QtWidgets.QHBoxLayout()
        hb_out.addWidget(self.editCropOutDir, 1)
        hb_out.addWidget(self.btnBrowseCropOut)
        form.addRow("Output base:", hb_out)

        frames_widget = QtWidgets.QWidget()
        frames_layout = QtWidgets.QGridLayout(frames_widget)
        frames_layout.setContentsMargins(0, 0, 0, 0)

        frames_layout.addWidget(QtWidgets.QLabel("Start"), 0, 0)
        self.spinCropFrameStart = QtWidgets.QSpinBox()
        self.spinCropFrameStart.setRange(0, 1_000_000)
        frames_layout.addWidget(self.spinCropFrameStart, 0, 1)

        frames_layout.addWidget(QtWidgets.QLabel("End"), 0, 2)
        self.spinCropFrameEnd = QtWidgets.QSpinBox()
        self.spinCropFrameEnd.setRange(-1, 1_000_000)
        self.spinCropFrameEnd.setSpecialValueText("All")
        self.spinCropFrameEnd.setValue(-1)
        frames_layout.addWidget(self.spinCropFrameEnd, 0, 3)

        frames_layout.addWidget(QtWidgets.QLabel("Step"), 1, 0)
        self.spinCropFrameStep = QtWidgets.QSpinBox()
        self.spinCropFrameStep.setRange(1, 1_000_000)
        self.spinCropFrameStep.setValue(1)
        frames_layout.addWidget(self.spinCropFrameStep, 1, 1)

        frames_layout.setColumnStretch(1, 1)
        frames_layout.setColumnStretch(3, 1)

        form.addRow("Frames:", frames_widget)

        layout.addLayout(form)

        self.btnRunCropSaved = QtWidgets.QPushButton("Crop saved files")
        self.btnRunCropSaved.clicked.connect(self._on_run_crop_saved)
        layout.addWidget(self.btnRunCropSaved)
        layout.addStretch(1)

        return widget

    def _create_crop_bag_tab(self) -> QtWidgets.QWidget:
        """bag 파일에서 크롭하는 탭 UI 구성."""

        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()

        self.editCropBagPath = QtWidgets.QLineEdit()
        self.editCropBagPath.editingFinished.connect(self._update_bag_info)
        self.btnBrowseCropBag = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropBag.clicked.connect(self._browse_crop_bag)
        hb_bag = QtWidgets.QHBoxLayout()
        hb_bag.addWidget(self.editCropBagPath, 1)
        hb_bag.addWidget(self.btnBrowseCropBag)
        form.addRow(".bag file:", hb_bag)

        self.lblCropBagInfo = QtWidgets.QLabel("Bag: -")
        self.lblCropBagInfo.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        form.addRow("Info:", self.lblCropBagInfo)

        self.editCropBagOutDir = QtWidgets.QLineEdit(DEFAULT_CROP_BASE)
        self.btnBrowseCropBagOut = QtWidgets.QPushButton("Browse...")
        self.btnBrowseCropBagOut.clicked.connect(self._browse_crop_bag_out)
        hb_out = QtWidgets.QHBoxLayout()
        hb_out.addWidget(self.editCropBagOutDir, 1)
        hb_out.addWidget(self.btnBrowseCropBagOut)
        form.addRow("Output base:", hb_out)

        frames_widget = QtWidgets.QWidget()
        frames_layout = QtWidgets.QGridLayout(frames_widget)
        frames_layout.setContentsMargins(0, 0, 0, 0)

        frames_layout.addWidget(QtWidgets.QLabel("Start"), 0, 0)
        self.spinBagFrameStart = QtWidgets.QSpinBox()
        self.spinBagFrameStart.setRange(0, 1_000_000)
        frames_layout.addWidget(self.spinBagFrameStart, 0, 1)

        frames_layout.addWidget(QtWidgets.QLabel("End"), 0, 2)
        self.spinBagFrameEnd = QtWidgets.QSpinBox()
        self.spinBagFrameEnd.setRange(-1, 1_000_000)
        self.spinBagFrameEnd.setSpecialValueText("All")
        self.spinBagFrameEnd.setValue(-1)
        frames_layout.addWidget(self.spinBagFrameEnd, 0, 3)

        frames_layout.addWidget(QtWidgets.QLabel("Step"), 1, 0)
        self.spinBagFrameStep = QtWidgets.QSpinBox()
        self.spinBagFrameStep.setRange(1, 1_000_000)
        self.spinBagFrameStep.setValue(1)
        frames_layout.addWidget(self.spinBagFrameStep, 1, 1)

        frames_layout.setColumnStretch(1, 1)
        frames_layout.setColumnStretch(3, 1)

        form.addRow("Frames:", frames_widget)

        layout.addLayout(form)

        self.btnRunCropBag = QtWidgets.QPushButton("Crop .bag")
        self.btnRunCropBag.clicked.connect(self._on_run_crop_bag)
        layout.addWidget(self.btnRunCropBag)
        layout.addStretch(1)

        return widget

    # ---------------------------- Crop helpers ---------------------------- #
    def _configure_frame_range_spinboxes(
        self,
        frames: int,
        spin_start: QtWidgets.QSpinBox,
        spin_end: QtWidgets.QSpinBox,
        spin_step: QtWidgets.QSpinBox,
    ) -> None:
        """프레임 범위 선택 스핀박스의 범위를 총 프레임 수에 맞게 조정."""

        frames = max(0, int(frames))

        if frames == 0:
            spin_start.setRange(0, 0)
            spin_start.setValue(0)
            spin_end.setRange(-1, 0)
            spin_end.setValue(-1)
            spin_step.setRange(1, 1_000_000)
            spin_step.setValue(1)
            return

        spin_start.setRange(0, max(0, frames - 1))
        spin_end.setRange(-1, frames)
        spin_step.setRange(1, max(1, frames))

        if spin_start.value() >= frames:
            spin_start.setValue(0)

        end_val = spin_end.value()
        if end_val not in (-1,) and end_val > frames:
            spin_end.setValue(frames)

        spin_step.setValue(max(1, min(spin_step.value(), frames)))

    def _update_saved_depth_info(self) -> None:
        """선택된 depth NPY 정보(프레임 수/크기)를 UI에 반영."""

        if not hasattr(self, "editCropDepthPath"):
            return

        path = self.editCropDepthPath.text().strip()
        if not path:
            self.lblCropSavedInfo.setText("Depth: -")
            self._saved_depth_shape = None
            return

        try:
            info = probe_npy_info(path)
            frames = int(info.get("frames", 0))
            height = int(info.get("height", 0))
            width = int(info.get("width", 0))
            dtype = info.get("dtype", "?")
            prev_shape = self._saved_depth_shape
            self._saved_depth_shape = (frames, height, width)
            self.lblCropSavedInfo.setText(
                f"Frames={frames}  Size={width}x{height}  dtype={dtype}"
            )
            self._configure_frame_range_spinboxes(
                frames,
                self.spinCropFrameStart,
                self.spinCropFrameEnd,
                self.spinCropFrameStep,
            )
            if prev_shape != self._saved_depth_shape:
                self.spinCropFrameStart.setValue(0)
                self.spinCropFrameEnd.setValue(-1)
                self.spinCropFrameStep.setValue(1)
        except Exception as exc:  # noqa: BLE001
            self.lblCropSavedInfo.setText(f"Error: {exc}")
            self._saved_depth_shape = None

    def _update_bag_info(self) -> None:
        """선택된 bag 파일의 해상도 정보를 UI에 반영."""

        if not hasattr(self, "editCropBagPath"):
            return

        path = self.editCropBagPath.text().strip()
        if not path:
            self.lblCropBagInfo.setText("Bag: -")
            self._bag_info = None
            return

        try:
            info = probe_bag_info(path)
            prev_info = self._bag_info
            self._bag_info = info
            depth = info.get("depth") or {}
            width = int(depth.get("width", 0))
            height = int(depth.get("height", 0))
            fps_val = depth.get("fps")
            fps_txt = f"{fps_val}fps" if fps_val else "?"
            color = info.get("color") or {}
            if color:
                c_w = color.get("width")
                c_h = color.get("height")
                color_txt = f" | Color: {c_w}x{c_h}"
            else:
                color_txt = " | Color: N/A"
            scale = info.get("depth_scale_m_per_unit")
            scale_txt = f" | scale={scale}" if scale is not None else ""
            frames_estimate = info.get("frames_estimate")
            if isinstance(frames_estimate, int) and frames_estimate > 0:
                frames_txt = f" | ~{frames_estimate} frames"
            else:
                frames_txt = ""
            self.lblCropBagInfo.setText(
                f"Depth: {width}x{height} @ {fps_txt}{color_txt}{scale_txt}{frames_txt}"
            )

            estimate = (
                frames_estimate if isinstance(frames_estimate, int) and frames_estimate > 0 else 0
            )
            self._configure_frame_range_spinboxes(
                estimate,
                self.spinBagFrameStart,
                self.spinBagFrameEnd,
                self.spinBagFrameStep,
            )
            reset_range = True
            if isinstance(prev_info, dict):
                prev_path = prev_info.get("path")
                prev_estimate = prev_info.get("frames_estimate")
                reset_range = (
                    prev_path != info.get("path")
                    or prev_estimate != frames_estimate
                )
            if reset_range:
                self.spinBagFrameStart.setValue(0)
                self.spinBagFrameEnd.setValue(-1)
                self.spinBagFrameStep.setValue(1)
        except Exception as exc:  # noqa: BLE001
            self.lblCropBagInfo.setText(f"Error: {exc}")
            self._bag_info = None

    def _append_crop_log(self, msg: str) -> None:
        """크롭 로그 텍스트 박스에 메시지를 추가."""

        if hasattr(self, "txtCropLog"):
            self.txtCropLog.append(msg)

    def _set_crop_running(self, running: bool) -> None:
        """크롭 실행 중 UI 활성 상태 제어."""

        self._crop_running = running

        widgets = [
            getattr(self, "btnRunCropSaved", None),
            getattr(self, "btnRunCropBag", None),
            getattr(self, "editCropDepthPath", None),
            getattr(self, "btnBrowseCropDepth", None),
            getattr(self, "editCropColorPath", None),
            getattr(self, "btnBrowseCropColor", None),
            getattr(self, "editCropDepthColorPath", None),
            getattr(self, "btnBrowseCropDepthColor", None),
            getattr(self, "editCropOutDir", None),
            getattr(self, "btnBrowseCropOut", None),
            getattr(self, "spinCropFrameStart", None),
            getattr(self, "spinCropFrameEnd", None),
            getattr(self, "spinCropFrameStep", None),
            getattr(self, "editCropBagPath", None),
            getattr(self, "btnBrowseCropBag", None),
            getattr(self, "editCropBagOutDir", None),
            getattr(self, "btnBrowseCropBagOut", None),
            getattr(self, "spinBagFrameStart", None),
            getattr(self, "spinBagFrameEnd", None),
            getattr(self, "spinBagFrameStep", None),
        ]

        for widget in widgets:
            if widget is not None:
                widget.setEnabled(not running)

        if hasattr(self, "btnCropOpenOut"):
            self.btnCropOpenOut.setEnabled(bool(self._last_crop_out) and not running)

    def _open_crop_out(self) -> None:
        """크롭 결과 폴더 열기."""

        if self._last_crop_out:
            self._open_path(self._last_crop_out)

    # ---------------------------- Crop actions ---------------------------- #
    def _browse_crop_depth(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select depth_raw_frames.npy", "", "NumPy NPY (*.npy)"
        )
        if path:
            self.editCropDepthPath.setText(path)
            self._update_saved_depth_info()

    def _browse_crop_color(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select color video",
            "",
            "Videos/Images (*.mp4 *.avi *.mkv *.mov *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.editCropColorPath.setText(path)

    def _browse_crop_depth_color(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select depth colorized video",
            "",
            "Videos/Images (*.mp4 *.avi *.mkv *.mov *.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.editCropDepthColorPath.setText(path)

    def _browse_crop_saved_out(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select crop output base", ""
        )
        if directory:
            self.editCropOutDir.setText(directory)

    def _browse_crop_bag(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select .bag file", "", "RealSense bag (*.bag)"
        )
        if path:
            self.editCropBagPath.setText(path)
            self._update_bag_info()

    def _browse_crop_bag_out(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select crop output base", ""
        )
        if directory:
            self.editCropBagOutDir.setText(directory)

    def _on_run_crop_saved(self) -> None:
        """저장된 산출물 크롭 실행."""

        if self._crop_running:
            QtWidgets.QMessageBox.information(
                self, "Crop", "이미 크롭 작업이 진행 중입니다."
            )
            return

        depth_path = self.editCropDepthPath.text().strip()
        if not depth_path:
            QtWidgets.QMessageBox.information(
                self, "Crop", "depth_raw_frames.npy 경로를 선택하세요."
            )
            return

        out_base = self.editCropOutDir.text().strip() or DEFAULT_CROP_BASE
        frame_start = int(self.spinCropFrameStart.value())
        frame_end_val = int(self.spinCropFrameEnd.value())
        frame_end = None if frame_end_val < 0 else frame_end_val
        frame_step = int(self.spinCropFrameStep.value())
        color_path = self.editCropColorPath.text().strip() or None
        depth_color_path = self.editCropDepthColorPath.text().strip() or None

        frame_end_txt = "end" if frame_end is None else str(frame_end)
        self._append_crop_log(
            f"[Saved] Start depth={depth_path} frames={frame_start}:{frame_end_txt}:{frame_step} "
            f"color={color_path} depth_color={depth_color_path}"
        )
        self._set_crop_running(True)
        self._crop_context = "saved"

        self._crop_thread = FuncThread(
            crop_saved_outputs,
            depth_path,
            out_base,
            frame_start,
            frame_end,
            frame_step,
            color_source=color_path,
            depth_color_source=depth_color_path,
        )
        self._crop_thread.sig_done.connect(self._on_crop_done)
        self._crop_thread.sig_error.connect(self._on_crop_error)
        self._crop_thread.start()

    def _on_run_crop_bag(self) -> None:
        """bag 파일 크롭 실행."""

        if self._crop_running:
            QtWidgets.QMessageBox.information(
                self, "Crop", "이미 크롭 작업이 진행 중입니다."
            )
            return

        bag_path = self.editCropBagPath.text().strip()
        if not bag_path:
            QtWidgets.QMessageBox.information(
                self, "Crop", ".bag 파일을 선택하세요."
            )
            return

        out_base = self.editCropBagOutDir.text().strip() or DEFAULT_CROP_BASE
        frame_start = int(self.spinBagFrameStart.value())
        frame_end_val = int(self.spinBagFrameEnd.value())
        frame_end = None if frame_end_val < 0 else frame_end_val
        frame_step = int(self.spinBagFrameStep.value())

        frame_end_txt = "end" if frame_end is None else str(frame_end)
        self._append_crop_log(
            f"[Bag] Start bag={bag_path} frames={frame_start}:{frame_end_txt}:{frame_step}"
        )
        self._set_crop_running(True)
        self._crop_context = "bag"

        self._crop_thread = FuncThread(
            crop_bag_frames,
            bag_path,
            out_base,
            frame_start,
            frame_end,
            frame_step,
        )
        self._crop_thread.sig_done.connect(self._on_crop_done)
        self._crop_thread.sig_error.connect(self._on_crop_error)
        self._crop_thread.start()

    def _on_crop_done(self, summary: Dict[str, Any]) -> None:
        """크롭 완료 시 처리."""

        self._set_crop_running(False)

        out_dir = summary.get("out_dir")
        if out_dir:
            self._last_crop_out = str(out_dir)
            self.lblCropOut.setText(str(out_dir))
            self.btnCropOpenOut.setEnabled(True)

        self._append_crop_log("[CROP SUMMARY]")
        for key, value in summary.items():
            self._append_crop_log(f"  {key}: {value}")

        if self.statusBar():
            self.statusBar().showMessage("Crop finished", 5000)

        self._crop_context = None
        verify_widget = getattr(self, "_crop_verify_widget", None)
        if isinstance(verify_widget, CropVerifyWidget) and out_dir:
            try:
                verify_widget.load_directory(str(out_dir))
            except Exception:
                pass
        if self._crop_thread is not None:
            self._crop_thread.deleteLater()
        self._crop_thread = None

    def _on_crop_error(self, msg: str) -> None:
        """크롭 오류 처리."""

        self._set_crop_running(False)
        self._append_crop_log(f"[ERROR] {msg}")
        QtWidgets.QMessageBox.critical(self, "Crop Error", msg)
        self._crop_context = None
        if self._crop_thread is not None:
            self._crop_thread.deleteLater()
        self._crop_thread = None

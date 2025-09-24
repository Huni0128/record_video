# -*- coding: utf-8 -*-
"""메인 윈도우 로직과 분석 실행 스레드를 포함한 PyQt5 GUI 모듈."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets, uic

from ..analysis import DEFAULT_ANALYSIS_BASE, analyze_npy as run_analyze_npy
from ..core import DEFAULT_BASE_DIR, RecordConfig, ensure_out_dir
from ..recording import RecordThread
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
            self.editOutDir.setText("save/farm_record")
        if hasattr(self, "comboRes"):
            self.comboRes.setCurrentText("1280x720")
        if hasattr(self, "comboFps"):
            self.comboFps.setCurrentText("15")
        if hasattr(self, "chkSaveDepthRaw"):
            self.chkSaveDepthRaw.setText("Save Depth (.npy)")
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

        # 분석 도크
        dock = QtWidgets.QDockWidget("Analysis (NPY)", self)
        dock.setObjectName("dockAnalysisNpy")
        dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

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

        dock.setWidget(widget)

    # ----------------------- Helpers for analysis dock -------------------- #
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

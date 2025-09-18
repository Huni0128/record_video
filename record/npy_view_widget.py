# -*- coding: utf-8 -*-
"""
NPY(depth) 프레임 뷰어 위젯.

- depth_raw_frames.npy: (N, H, W) uint16
- 슬라이더/키보드로 프레임 이동(←/→ 또는 A/D)
- Auto CLim(2~98%) 지원
- 히스토그램(H)
- 캔버스 휠은 부모 스크롤로 전달

Docstring 스타일: Google Style
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FigureCanvasNoWheel(FigureCanvas):
    """마우스 휠 이벤트를 부모로 넘겨 페이지 스크롤이 되도록 합니다."""

    def wheelEvent(self, event) -> None:  # noqa: N802  (Qt 메서드명 유지)
        event.ignore()  # 부모(QScrollArea)가 처리


class NpyViewerWidget(QtWidgets.QWidget):
    """NPY(depth) 파일을 PyQt 내에서 인터랙티브로 표시하는 위젯."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.depth: Optional[np.memmap] = None
        self.scale: float = 1e-3
        self.shape: Tuple[int, int, int] = (0, 0, 0)  # (N, H, W)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(420)

        # 상단 경로 표시
        self.lblPath = QtWidgets.QLabel("-")
        self.lblPath.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Figure/Canvas
        self.fig = Figure(figsize=(9, 6), dpi=100, tight_layout=False)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasNoWheel(self.fig)  # 휠 이벤트를 부모로 넘김
        self.canvas.setFocusPolicy(QtCore.Qt.NoFocus)  # 포커스 가져가서 휠 잡는 것 방지
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(360)
        self.im = None

        # Controls
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)
        self.slider.valueChanged.connect(self._on_slider)

        self.lblIdx = QtWidgets.QLabel("0/0")
        # 고정폭 폰트 힌트
        font = self.lblIdx.font()
        font.setStyleHint(font.Monospace)
        self.lblIdx.setFont(font)

        self.chkAuto = QtWidgets.QCheckBox("Auto CLim")
        self.chkAuto.setChecked(True)
        self.chkAuto.toggled.connect(self._refresh_clim)

        self.btnHist = QtWidgets.QPushButton("Histogram (H)")
        self.btnHist.clicked.connect(self._show_hist)

        self.lblStats = QtWidgets.QLabel(
            "nz=0.000  min=nan  p50=nan  p95=nan  max=nan  mean=nan (m)"
        )
        self.lblStats.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # Layout
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("File:"))
        top.addWidget(self.lblPath, 1)

        ctr = QtWidgets.QHBoxLayout()
        ctr.addWidget(self.slider, 1)
        ctr.addWidget(self.lblIdx)
        ctr.addSpacing(8)
        ctr.addWidget(self.chkAuto)
        ctr.addSpacing(8)
        ctr.addWidget(self.btnHist)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addLayout(top)
        vbox.addWidget(self.canvas, 1)
        vbox.addLayout(ctr)
        vbox.addWidget(self.lblStats)

        # Canvas가 세로 공간을 가장 많이 먹게
        vbox.setStretch(0, 0)   # path
        vbox.setStretch(1, 10)  # canvas
        vbox.setStretch(2, 0)   # controls
        vbox.setStretch(3, 0)   # stats

        # 여백 최소화 & 축/눈금 제거로 깔끔하게
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        self._style_axes()

        # 리사이즈/키 이벤트
        # mypy/pyright 혼동 방지를 위해 ignore
        self.canvas.resizeEvent = self._on_canvas_resize  # type: ignore[assignment]
        self.canvas.mpl_connect("key_press_event", self._on_key)

    # ------------------------------ Public ------------------------------ #
    def load_file(self, npy_path: str, meta_path: Optional[str] = None) -> None:
        """NPY 파일을 로드하고 초기 프레임을 표시합니다.

        Args:
            npy_path: depth_raw_frames.npy 경로.
            meta_path: (옵션) 메타 JSON 경로. 미지정 시 같은 폴더의
                device_stream_info.json을 사용합니다.
        """
        npy_path = os.path.abspath(npy_path)
        if meta_path is None:
            meta_path = os.path.join(os.path.dirname(npy_path), "device_stream_info.json")

        depth = np.load(npy_path, mmap_mode="r")
        if depth.ndim != 3:
            raise ValueError(
                f"depth 배열 차원 오류: expected 3D (N, H, W), got {depth.shape}"
            )

        scale = 1e-3
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            s = float(meta.get("depth_scale_m_per_unit", 1e-3))
            if np.isfinite(s):
                scale = s
        except Exception:
            # 메타가 없거나 파싱 실패 시 기본 스케일 사용
            pass

        self.depth = depth
        self.scale = float(scale)
        self.shape = tuple(depth.shape)  # (N, H, W)
        self.lblPath.setText(npy_path)

        num_frames, _, _ = self.shape
        self.slider.setMaximum(max(0, num_frames - 1))
        self.slider.setValue(0)
        self._show_frame(0)

    # ----------------------------- Internals ---------------------------- #
    def _style_axes(self) -> None:
        """이미지 전용 축 스타일(눈금/테두리 제거)."""
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_facecolor("#000")  # 레터박스 영역 어둡게

    def _on_slider(self, value: int) -> None:
        self._show_frame(int(value))

    def _frame_to_m(self, idx: int) -> np.ndarray:
        """프레임을 미터 단위 float32로 변환."""
        assert self.depth is not None
        frame = self.depth[idx].astype(np.float32)
        return frame * self.scale

    def _valid_mask(self, fm: np.ndarray) -> np.ndarray:
        """유효 깊이(>0, finite) 마스크."""
        return (fm > 0) & np.isfinite(fm)

    def _stats(self, fm: np.ndarray) -> Dict[str, float]:
        """프레임 통계값 계산."""
        m = self._valid_mask(fm)
        if m.sum() == 0:
            return {
                "nz": 0.0,
                "vmin": float("nan"),
                "p50": float("nan"),
                "p95": float("nan"),
                "vmax": float("nan"),
                "mean": float("nan"),
            }
        vals = fm[m]
        return {
            "nz": float(m.mean()),
            "vmin": float(vals.min()),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
            "vmax": float(vals.max()),
            "mean": float(vals.mean()),
        }

    def _auto_clim(self, fm: np.ndarray, ql: float = 2, qh: float = 98) -> Tuple[float, float]:
        """자동 대비 범위(백분위) 계산."""
        m = self._valid_mask(fm)
        if m.sum() == 0:
            return 0.0, 1.0
        vals = fm[m]
        vmin, vmax = np.percentile(vals, [ql, qh])
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)

    def _refresh_clim(self) -> None:
        """현재 프레임의 CLim을 재설정."""
        if self.im is None:
            return
        i = self.slider.value()
        fm = self._frame_to_m(i)
        if self.chkAuto.isChecked():
            vmin, vmax = self._auto_clim(fm)
            self.im.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.draw_idle()

    def _show_frame(self, i: int) -> None:
        """프레임 인덱스 i를 표시."""
        if self.depth is None:
            return

        n, h, w = self.shape
        i = max(0, min(i, n - 1))
        fm = self._frame_to_m(i)
        st = self._stats(fm)

        if self.im is None:
            self.ax.clear()
            self._style_axes()
            # 데이터 종횡비 유지 + 보간 없음으로 또렷하게
            self.im = self.ax.imshow(
                fm,
                origin="upper",
                interpolation="nearest",
                aspect="equal",
                cmap="viridis",
            )
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
        else:
            self.im.set_data(fm)
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)

        if self.chkAuto.isChecked():
            vmin, vmax = self._auto_clim(fm)
            self.im.set_clim(vmin=vmin, vmax=vmax)

        self.lblIdx.setText(f"{i}/{n-1}")
        self.lblStats.setText(
            f"nz={st['nz']:.3f}  min={st['vmin']:.3f}  p50={st['p50']:.3f}  "
            f"p95={st['p95']:.3f}  max={st['vmax']:.3f}  mean={st['mean']:.3f}  (m)"
        )

        # 축 박스를 캔버스에 꽉 차게
        self._update_axes_box()
        self.canvas.draw_idle()

    def _update_axes_box(self) -> None:
        """캔버스 픽셀 크기와 데이터 AR을 맞춰 축 위치/크기를 배치."""
        if self.depth is None:
            return

        _, h, w = self.shape
        data_ar = float(w) / float(h) if h > 0 else 1.0

        # 현재 캔버스 크기(px) → figure(inch)
        cw = max(1, self.canvas.width())
        ch = max(1, self.canvas.height())
        fig_w_in = cw / self.fig.dpi
        fig_h_in = ch / self.fig.dpi

        # figure 안에서 데이터 AR 유지하며 최대 크기 배치
        target_w_in = min(fig_w_in, fig_h_in * data_ar)
        target_h_in = target_w_in / data_ar

        left = (fig_w_in - target_w_in) / 2.0 / fig_w_in
        bottom = (fig_h_in - target_h_in) / 2.0 / fig_h_in
        width = target_w_in / fig_w_in
        height = target_h_in / fig_h_in

        # 축 위치 설정(0~1 정규화)
        self.ax.set_position([left, bottom, width, height])

    def _on_canvas_resize(self, ev) -> None:
        """Qt가 캔버스를 리사이즈할 때마다 축 박스를 재배치."""
        self._update_axes_box()
        # 기본 처리 유지(FigureCanvas의 resizeEvent 호출)
        return FigureCanvas.resizeEvent(self.canvas, ev)

    def _show_hist(self) -> None:
        """현재 프레임의 깊이 히스토그램을 다이얼로그로 표시."""
        if self.depth is None:
            return

        i = self.slider.value()
        fm = self._frame_to_m(i)
        m = self._valid_mask(fm)
        vals = fm[m]

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Depth Histogram (frame {i})")

        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        if vals.size > 0:
            ax.hist(vals, bins=200)
        ax.set_xlabel("Depth (m)")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvas(fig)
        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.addWidget(canvas)

        dlg.resize(760, 480)
        dlg.show()

    def _on_key(self, event) -> None:
        """키보드 이벤트 처리(A/D 또는 ←/→, H)."""
        if not event.key:
            return
        k = event.key.lower()
        if k in ("h",):
            self._show_hist()
        elif k in ("left", "a"):
            self.slider.setValue(max(0, self.slider.value() - 1))
        elif k in ("right", "d"):
            self.slider.setValue(min(self.slider.maximum(), self.slider.value() + 1))

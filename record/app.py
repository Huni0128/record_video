# -*- coding: utf-8 -*-
"""
PyQt5 기반 GUI 실행 진입점 모듈.

MainWindow를 초기화하고 QApplication을 실행합니다.

Docstring 스타일: Google Style
"""
from __future__ import annotations

import sys
from PyQt5 import QtWidgets

from .main_window import MainWindow


def main() -> None:
    """애플리케이션 실행 진입점.

    QApplication을 생성하고 MainWindow를 띄운 후 이벤트 루프를 실행합니다.
    """
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 1000)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

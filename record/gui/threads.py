"""Reusable Qt thread helpers used by the GUI layer."""
from __future__ import annotations

from typing import Any, Callable, Dict

from PyQt5 import QtCore

__all__ = ["FuncThread"]


class FuncThread(QtCore.QThread):
    """Run a blocking callable on a background thread and emit results."""

    sig_done = QtCore.pyqtSignal(object)
    sig_error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        func: Callable[..., Dict[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:  # noqa: D401 - inherited documentation suffices
        try:
            result = self.func(*self.args, **self.kwargs)
            func_name = getattr(self.func, "__name__", str(self.func))
            if result is None:
                self.sig_error.emit(f"{func_name} returned None (no summary).")
                return
            if not isinstance(result, dict):
                msg = f"{func_name} returned {type(result).__name__}, expected dict."
                self.sig_error.emit(msg)
                return
            self.sig_done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.sig_error.emit(str(exc))

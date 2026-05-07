from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStatusBar,
    QWidget,
)

from hmi_demo.config import Config


class HMIWindow(QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("HMI Manipulation Demo (MVP)")

        central = QWidget()
        layout = QHBoxLayout(central)

        self.sim_label = QLabel("(sim view loading…)")
        self.sim_label.setFixedSize(*cfg.ui.sim_view_size)
        self.sim_label.setStyleSheet("background:#000;color:#888;")
        self.sim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cam_label = QLabel("(cam view loading…)")
        self.cam_label.setFixedSize(*cfg.ui.cam_view_size)
        self.cam_label.setStyleSheet("background:#000;color:#888;")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.sim_label)
        layout.addWidget(self.cam_label)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._set_status("RUNNING")

    def _set_status(self, state: str) -> None:
        color_map = {
            "RUNNING": "#1b8a3a",
            "PAUSED (gesture)": "#c08400",
            "RETURN_HOME": "#1f6391",
            "E-STOP": "#a02020",
            "NO CAMERA": "#444",
        }
        bg = color_map.get(state, "#222")
        self.status.setStyleSheet(f"QStatusBar{{background:{bg};color:#fff;font-weight:600;}}")
        self.status.showMessage(state)

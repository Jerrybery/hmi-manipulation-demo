from __future__ import annotations

from enum import Enum, auto

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QWidget,
)

from hmi_demo.config import Config
from hmi_demo.sim.ik import diff_ik_dls
from hmi_demo.sim.trajectory import CircleTrajectory
from hmi_demo.sim.world import World
from hmi_demo.ui.render import MujocoRenderer


class MotionState(Enum):
    RUNNING = auto()
    FROZEN = auto()
    RETURN_HOME = auto()


class HMIWindow(QMainWindow):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("HMI Manipulation Demo (MVP)")

        central = QWidget()
        layout = QHBoxLayout(central)
        self.sim_label = QLabel()
        self.sim_label.setFixedSize(*cfg.ui.sim_view_size)
        self.sim_label.setStyleSheet("background:#000;")
        self.sim_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label = QLabel("(no camera)")
        self.cam_label.setFixedSize(*cfg.ui.cam_view_size)
        self.cam_label.setStyleSheet("background:#000;color:#888;")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.sim_label)
        layout.addWidget(self.cam_label)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        try:
            self.world = World(cfg.sim)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.critical(self, "Simulation init failed", str(e))
            raise

        sim_w, sim_h = cfg.ui.sim_view_size
        self.renderer = MujocoRenderer(self.world.model, width=sim_w, height=sim_h)
        self.trajectory = CircleTrajectory(
            center=cfg.trajectory.center,
            radius=cfg.trajectory.radius,
            omega=cfg.trajectory.omega,
        )

        self.state = MotionState.RUNNING
        self.gesture_frozen = False
        self.estop_frozen = False
        self._return_progress = 0.0
        self._q_at_release = self.world.q_home.copy()
        self._dt_outer = 1.0 / cfg.sim.control_hz

        self.timer = QTimer(self)
        self.timer.setInterval(int(1000 / cfg.sim.control_hz))
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._update_status()

    def _effective_frozen(self) -> bool:
        return self.gesture_frozen or self.estop_frozen

    def _tick(self) -> None:
        prev_state = self.state
        frozen = self._effective_frozen()

        if self.state == MotionState.RUNNING and frozen:
            self.state = MotionState.FROZEN
        elif self.state == MotionState.FROZEN and not frozen:
            if self.cfg.recovery.mode == "resume_in_place":
                self.state = MotionState.RUNNING
            else:
                self._q_at_release = np.array(
                    self.world.data.qpos[: self.world.model.nq], copy=True
                )
                self._return_progress = 0.0
                self.state = MotionState.RETURN_HOME

        if self.state == MotionState.RUNNING:
            x_target = self.trajectory.tick(self._dt_outer)
            q_target = diff_ik_dls(
                self.world.model, self.world.data, self.world.ee_site_id, x_target,
                damping=self.cfg.ik.damping, kp=self.cfg.ik.kp,
            )
        elif self.state == MotionState.FROZEN:
            x_target = self.trajectory.tick(0.0)
            q_target = diff_ik_dls(
                self.world.model, self.world.data, self.world.ee_site_id, x_target,
                damping=self.cfg.ik.damping, kp=self.cfg.ik.kp,
            )
        else:  # RETURN_HOME
            self._return_progress = min(
                1.0,
                self._return_progress + self._dt_outer / self.cfg.recovery.return_duration_s,
            )
            s = self._return_progress
            q_target = (1.0 - s) * self._q_at_release + s * self.world.q_home
            if self._return_progress >= 1.0:
                self.trajectory.reset()
                self.state = MotionState.RUNNING

        self.world.data.ctrl[: self.world.model.nu] = q_target
        self.world.step()

        try:
            qimg = self.renderer.grab(self.world.data)
            self.sim_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            self.timer.stop()
            QMessageBox.critical(self, "Render error", str(e))
            return

        if self.state != prev_state:
            self._update_status()

    def _update_status(self) -> None:
        if self.estop_frozen:
            label = "E-STOP"
        elif self.state == MotionState.RUNNING:
            label = "RUNNING"
        elif self.state == MotionState.FROZEN:
            label = "PAUSED (gesture)"
        else:
            label = "RETURN_HOME"
        color_map = {
            "RUNNING": "#1b8a3a",
            "PAUSED (gesture)": "#c08400",
            "RETURN_HOME": "#1f6391",
            "E-STOP": "#a02020",
        }
        bg = color_map.get(label, "#222")
        self.status.setStyleSheet(f"QStatusBar{{background:{bg};color:#fff;font-weight:600;}}")
        self.status.showMessage(label)

    def closeEvent(self, event):
        self.timer.stop()
        self.renderer.close()
        super().closeEvent(event)

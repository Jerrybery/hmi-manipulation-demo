from __future__ import annotations

import logging
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

_LOG = logging.getLogger(__name__)


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

        from hmi_demo.vision.hand_thread import VisionThread
        self.vision_thread = VisionThread(cfg.camera, cfg.gesture, parent=self)
        self.vision_thread.gestureUpdated.connect(self._on_gesture)
        self.vision_thread.cameraError.connect(self._on_camera_error)
        self.vision_thread.start()

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
        else:
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
        except RuntimeError as exc:
            _LOG.error("MuJoCo renderer failed: %s", exc, exc_info=True)
            self.timer.stop()
            # Defer the dialog out of the timer callback to avoid event-loop re-entrancy.
            QTimer.singleShot(
                0,
                lambda: QMessageBox.critical(
                    self,
                    "Render error",
                    "The simulation renderer encountered an error and has been stopped.\n\n"
                    f"Detail: {exc}\n\nPlease restart the application.",
                ),
            )
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

    def _on_gesture(self, raised: bool, frame):
        self.gesture_frozen = raised
        self.cam_label.setPixmap(QPixmap.fromImage(frame))

    def _on_camera_error(self, msg: str):
        self.cam_label.setText(f"NO CAMERA\n{msg}")
        self.status.showMessage(f"NO CAMERA — {msg}")
        self.status.setStyleSheet("QStatusBar{background:#444;color:#ddd;}")
        # gesture_frozen stays False permanently — sim keeps running

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.estop_frozen = not self.estop_frozen
            self._update_status()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "vision_thread"):
            self.vision_thread.request_stop()
            finished = self.vision_thread.wait(2000)
            if not finished:
                _LOG.warning("VisionThread did not stop within 2s; camera may not be released cleanly")
        if hasattr(self, "renderer"):
            self.renderer.close()
        super().closeEvent(event)

from __future__ import annotations

import logging
import math
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
from hmi_demo.ui.render import MujocoRenderer, SphereGeom

_LOG = logging.getLogger(__name__)


class MotionState(Enum):
    RUNNING = auto()         # Circle tracing
    UNLOADING = auto()       # Arm lerping to unload_pose, gripper opening
    RETURNING_HOME = auto()  # Arm lerping from unload_pose to q_home (arm portion of home keyframe)
    IDLE = auto()            # Arm at home, gripper open, awaiting closed_fist resume


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
        self.estop_frozen = False
        self._latest_gesture: str = ""
        self._phase_progress: float = 0.0
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

    def _tick(self) -> None:
        if self.estop_frozen:
            try:
                qimg = self.renderer.grab(self.world.data)
                self.sim_label.setPixmap(QPixmap.fromImage(qimg))
            except RuntimeError as exc:
                _LOG.error("MuJoCo renderer failed: %s", exc, exc_info=True)
                self.timer.stop()
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

        prev_state = self.state
        cfg_r = self.cfg.recovery

        if self.state == MotionState.RUNNING:
            if self._latest_gesture == "open_palm":
                self._q_at_release = self.world.data.qpos[:6].copy()
                self._phase_progress = 0.0
                self.state = MotionState.UNLOADING

        if self.state == MotionState.RUNNING:
            x_target = self.trajectory.tick(self._dt_outer)
            q_target_arm = diff_ik_dls(
                self.world.model, self.world.data, self.world.ee_site_id, x_target,
                damping=self.cfg.ik.damping, kp=self.cfg.ik.kp,
            )[:6]
            gripper_ctrl = float(self.cfg.gripper.close_ctrl)

        elif self.state == MotionState.UNLOADING:
            self._phase_progress = min(
                1.0,
                self._phase_progress + self._dt_outer / cfg_r.unload_duration_s,
            )
            arm_frac = self._phase_progress
            gripper_frac = min(
                1.0,
                self._phase_progress * cfg_r.unload_duration_s / cfg_r.gripper_release_duration_s,
            )
            unload_pose = np.array(cfg_r.unload_pose, dtype=float)
            q_target_arm = (1.0 - arm_frac) * self._q_at_release + arm_frac * unload_pose
            gripper_ctrl = (
                (1.0 - gripper_frac) * self.cfg.gripper.close_ctrl
                + gripper_frac * self.cfg.gripper.open_ctrl
            )
            if self._phase_progress >= 1.0:
                self._phase_progress = 0.0
                self.state = MotionState.RETURNING_HOME

        elif self.state == MotionState.RETURNING_HOME:
            self._phase_progress = min(
                1.0,
                self._phase_progress + self._dt_outer / cfg_r.return_duration_s,
            )
            s = self._phase_progress
            unload_pose = np.array(cfg_r.unload_pose, dtype=float)
            q_home_arm = self.world.q_home[:6]
            q_target_arm = (1.0 - s) * unload_pose + s * q_home_arm
            gripper_ctrl = float(self.cfg.gripper.open_ctrl)
            if self._phase_progress >= 1.0:
                self.trajectory.reset()
                self.state = MotionState.IDLE

        else:  # IDLE
            q_target_arm = self.world.q_home[:6].copy()
            gripper_ctrl = float(self.cfg.gripper.open_ctrl)
            if (
                self._latest_gesture == "closed_fist"
                and self.cfg.gesture.enable_fist_resume
            ):
                self.state = MotionState.RUNNING

        self.world.data.ctrl[:6] = q_target_arm
        self.world.data.ctrl[6] = gripper_ctrl
        self.world.step()

        preview_geoms: list[SphereGeom] = []
        if self.state == MotionState.RUNNING:
            cfg_p = self.cfg.trajectory_preview
            dt_preview = cfg_p.horizon_s / cfg_p.n_samples
            color_near = np.array([*cfg_p.color_near, cfg_p.alpha])
            color_far = np.array([*cfg_p.color_far, cfg_p.alpha])
            for i in range(1, cfg_p.n_samples + 1):
                t_future = self.trajectory.t + i * dt_preview
                a = self.cfg.trajectory.omega * t_future
                pos = np.array([
                    self.trajectory.center[0] + self.cfg.trajectory.radius * math.cos(a),
                    self.trajectory.center[1] + self.cfg.trajectory.radius * math.sin(a),
                    self.trajectory.center[2],
                ])
                t_frac = (i - 1) / max(1, cfg_p.n_samples - 1)
                rgba = color_near * (1 - t_frac) + color_far * t_frac
                preview_geoms.append(SphereGeom(pos=pos, rgba=rgba, radius=cfg_p.sphere_radius))

        try:
            qimg = self.renderer.grab(self.world.data, extra_geoms=preview_geoms or None)
            self.sim_label.setPixmap(QPixmap.fromImage(qimg))
        except RuntimeError as exc:
            _LOG.error("MuJoCo renderer failed: %s", exc, exc_info=True)
            self.timer.stop()
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
        elif self.state == MotionState.UNLOADING:
            label = "UNLOADING"
        elif self.state == MotionState.RETURNING_HOME:
            label = "RETURNING HOME"
        else:
            label = "IDLE — Show closed fist to resume"
        color_map = {
            "RUNNING": "#1b8a3a",
            "UNLOADING": "#c66a14",
            "RETURNING HOME": "#1f6391",
            "IDLE — Show closed fist to resume": "#6a3c8a",
            "E-STOP": "#a02020",
        }
        bg = color_map.get(label, "#222")
        self.status.setStyleSheet(f"QStatusBar{{background:{bg};color:#fff;font-weight:600;}}")
        self.status.showMessage(label)

    def _on_gesture(self, gesture_name: str, frame) -> None:
        self._latest_gesture = gesture_name
        self.cam_label.setPixmap(QPixmap.fromImage(frame))

    def _on_camera_error(self, msg: str):
        self.cam_label.setText(f"NO CAMERA\n{msg}")
        self.status.showMessage(f"NO CAMERA — {msg}")
        self.status.setStyleSheet("QStatusBar{background:#444;color:#ddd;}")

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

from __future__ import annotations

import mujoco
import numpy as np
from PyQt6.QtGui import QImage


class MujocoRenderer:
    """Offscreen MuJoCo renderer that produces QImages."""

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        camera: str | int = -1,
    ):
        self.width = width
        self.height = height
        self.camera = camera
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def grab(self, data: mujoco.MjData) -> QImage:
        self._renderer.update_scene(data, camera=self.camera)
        rgb = self._renderer.render()  # shape (H, W, 3), uint8
        buf = np.ascontiguousarray(rgb)
        h, w, _ = buf.shape
        return QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    def close(self) -> None:
        self._renderer.close()

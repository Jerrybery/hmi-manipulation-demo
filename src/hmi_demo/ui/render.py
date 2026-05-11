from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PyQt6.QtGui import QImage


@dataclass(frozen=True)
class SphereGeom:
    pos: np.ndarray   # shape (3,), world coordinates
    rgba: np.ndarray  # shape (4,), values in [0, 1]
    radius: float


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

    def grab(self, data: mujoco.MjData, extra_geoms: list[SphereGeom] | None = None) -> QImage:
        self._renderer.update_scene(data, camera=self.camera)
        if extra_geoms:
            scn = self._renderer.scene  # MjvScene
            for spec in extra_geoms:
                if scn.ngeom >= scn.maxgeom:
                    break
                g = scn.geoms[scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([spec.radius, 0.0, 0.0]),
                    pos=np.asarray(spec.pos, dtype=float),
                    mat=np.eye(3).flatten(),
                    rgba=np.asarray(spec.rgba, dtype=float),
                )
                scn.ngeom += 1
        rgb = self._renderer.render()  # shape (H, W, 3), uint8
        buf = np.ascontiguousarray(rgb)
        h, w, _ = buf.shape
        return QImage(buf.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()

    def close(self) -> None:
        self._renderer.close()

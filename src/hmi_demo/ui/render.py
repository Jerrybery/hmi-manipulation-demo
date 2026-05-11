from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PyQt6.QtGui import QImage


@dataclass(eq=False)
class SphereGeom:
    """Render-time sphere spec. Not frozen / hashable: numpy fields break dataclass-derived __hash__."""
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
        camera: str | int | mujoco.MjvCamera | None = None,
    ):
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        if camera is None:
            # MuJoCo's auto free-camera frames the model's geometric center, which for
            # UR5e + gripper sits low and clips the wrist/gripper out of a 640x480 view.
            # Pull back to 2.2 m with a 25° downward gaze focused at the table-top
            # workspace (~0.5 m forward of the base) so the whole arm + gripper + the
            # trajectory preview spheres fit in frame.
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.3, 0.0, 0.4]
            cam.distance = 2.2
            cam.azimuth = 135.0
            cam.elevation = -25.0
            self.camera = cam
        else:
            self.camera = camera

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

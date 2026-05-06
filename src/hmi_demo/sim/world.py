from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hmi_demo.config import SimConfig


class World:
    def __init__(self, sim_cfg: SimConfig):
        mjcf_path = Path(sim_cfg.mjcf)
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

        self.cfg = sim_cfg
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, sim_cfg.ee_site
        )
        if self.ee_site_id < 0:
            available = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
                for i in range(self.model.nsite)
            ]
            raise ValueError(
                f"ee_site {sim_cfg.ee_site!r} not in model. Available sites: {available}"
            )

        keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, sim_cfg.home_keyframe
        )
        if keyframe_id < 0:
            raise ValueError(
                f"home_keyframe {sim_cfg.home_keyframe!r} not found in model keyframes"
            )
        self._home_keyframe_id = keyframe_id
        self.q_home = np.array(
            self.model.key_qpos[keyframe_id, : self.model.nq], copy=True
        )

        self.reset_to_home()

    def reset_to_home(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_keyframe_id)
        mujoco.mj_forward(self.model, self.data)

    def step(self) -> None:
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)

    def ee_position(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site_id], copy=True)

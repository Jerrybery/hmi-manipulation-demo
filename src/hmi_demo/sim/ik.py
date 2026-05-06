from __future__ import annotations

import mujoco
import numpy as np


def diff_ik_dls(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    x_target: np.ndarray,
    damping: float = 0.05,
    kp: float = 2.0,
) -> np.ndarray:
    """One-step differential IK with damped least squares.

    Returns absolute joint position command (length nq), clipped to joint limits.
    Only the 3-DOF position error is used; orientation is not constrained.
    """
    nq = model.nq
    nv = model.nv

    x_current = np.asarray(data.site_xpos[site_id], dtype=float)
    err = np.asarray(x_target, dtype=float) - x_current
    v_des = kp * err  # 3-vector

    Jp = np.zeros((3, nv))
    Jr = np.zeros((3, nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)

    # DLS: dq = J^T (J J^T + λ²I)^-1 v
    JJt = Jp @ Jp.T + (damping ** 2) * np.eye(3)
    dq_v = Jp.T @ np.linalg.solve(JJt, v_des)  # length nv

    q_current = np.asarray(data.qpos[:nq], dtype=float)
    # For UR5e (all hinge joints), nq == nv and qpos directly maps to qvel index.
    q_target = q_current + dq_v[:nq]

    lo = model.jnt_range[:nq, 0]
    hi = model.jnt_range[:nq, 1]
    return np.clip(q_target, lo, hi)

import numpy as np
import pytest

from hmi_demo.config import load_config
from hmi_demo.sim.ik import diff_ik_dls
from hmi_demo.sim.world import World


@pytest.fixture
def world(demo_yaml_path):
    cfg = load_config(demo_yaml_path)
    return World(cfg.sim), cfg


def test_ik_moves_ee_toward_target(world):
    w, cfg = world
    w.reset_to_home()
    x_curr = w.ee_position()
    x_target = x_curr + np.array([0.02, 0.0, 0.0])  # 2cm in +x

    for _ in range(50):
        q = diff_ik_dls(
            w.model, w.data, w.ee_site_id, x_target,
            damping=cfg.ik.damping, kp=cfg.ik.kp,
        )
        w.data.qpos[: w.model.nq] = q
        import mujoco
        mujoco.mj_forward(w.model, w.data)

    err = np.linalg.norm(w.ee_position() - x_target)
    assert err < 1e-3, f"IK failed to converge: {err:.4e}"


def test_ik_respects_joint_limits(world):
    w, cfg = world
    w.reset_to_home()
    x_target = w.ee_position() + np.array([5.0, 0.0, 0.0])  # unreachable
    q = diff_ik_dls(
        w.model, w.data, w.ee_site_id, x_target,
        damping=cfg.ik.damping, kp=cfg.ik.kp,
    )
    lo = w.model.jnt_range[: w.model.nq, 0]
    hi = w.model.jnt_range[: w.model.nq, 1]
    assert np.all(q >= lo - 1e-9)
    assert np.all(q <= hi + 1e-9)

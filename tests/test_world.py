import numpy as np

from hmi_demo.config import load_config
from hmi_demo.sim.world import World


def test_world_loads_and_steps_without_nan(demo_yaml_path):
    cfg = load_config(demo_yaml_path)
    world = World(cfg.sim)
    assert world.model.nq == 6
    assert world.model.nu == 6
    assert world.ee_site_id >= 0
    assert world.q_home.shape == (6,)
    world.reset_to_home()
    for _ in range(100):
        world.step()
    assert not np.isnan(world.data.qpos).any()


def test_world_unknown_site_raises(demo_yaml_path, tmp_path):
    cfg = load_config(demo_yaml_path)
    bad_sim = type(cfg.sim)(
        mjcf=cfg.sim.mjcf,
        ee_site="does_not_exist",
        home_keyframe=cfg.sim.home_keyframe,
        control_hz=cfg.sim.control_hz,
        substeps=cfg.sim.substeps,
    )
    import pytest
    with pytest.raises(ValueError, match="ee_site"):
        World(bad_sim)

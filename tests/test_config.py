from hmi_demo.config import Config, load_config


def test_load_demo_yaml(demo_yaml_path):
    cfg = load_config(demo_yaml_path)
    assert isinstance(cfg, Config)
    assert cfg.sim.ee_site == "attachment_site"
    assert cfg.sim.home_keyframe == "home"
    assert cfg.trajectory.radius == 0.15
    assert tuple(cfg.trajectory.center) == (0.5, 0.0, 0.3)
    assert cfg.ik.damping == 0.05
    assert cfg.gesture.hold_frames == 5
    assert cfg.recovery.mode in ("return_home", "resume_in_place")
    # New fields
    assert cfg.gripper.open_ctrl == 0
    assert cfg.gripper.close_ctrl == 255
    assert cfg.gesture.enable_fist_resume is True
    assert len(cfg.recovery.unload_pose) == 6
    assert cfg.recovery.unload_pose[1] == -1.5708
    assert cfg.recovery.unload_duration_s == 1.5
    assert cfg.recovery.gripper_release_duration_s == 0.5
    assert cfg.trajectory_preview.horizon_s == 2.0
    assert cfg.trajectory_preview.n_samples == 20
    assert tuple(cfg.trajectory_preview.color_near) == (0.1, 0.9, 0.2)


def test_recovery_mode_invalid_raises(demo_yaml_path, tmp_path):
    import yaml
    cfg_raw = yaml.safe_load(demo_yaml_path.read_text())
    cfg_raw['recovery']['mode'] = 'nonsense'
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.dump(cfg_raw))
    import pytest
    with pytest.raises(ValueError):
        load_config(bad)


def test_load_config_rejects_wrong_unload_pose_length(demo_yaml_path, tmp_path):
    import yaml
    cfg_raw = yaml.safe_load(demo_yaml_path.read_text())
    cfg_raw['recovery']['unload_pose'] = [0.0, 0.0, 0.0]  # wrong length
    bad = tmp_path / 'bad.yaml'
    bad.write_text(yaml.dump(cfg_raw))
    import pytest
    with pytest.raises(ValueError, match="unload_pose"):
        from hmi_demo.config import load_config
        load_config(bad)

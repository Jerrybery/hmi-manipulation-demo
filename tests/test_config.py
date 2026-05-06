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


def test_recovery_mode_invalid_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "sim:\n  mjcf: x\n  ee_site: x\n  home_keyframe: home\n  control_hz: 100\n  substeps: 5\n"
        "trajectory:\n  center: [0,0,0]\n  radius: 0.1\n  omega: 0.5\n"
        "ik:\n  damping: 0.05\n  kp: 2.0\n"
        "camera:\n  device: 0\n  width: 640\n  height: 480\n  fps: 30\n"
        "gesture:\n  hold_frames: 5\n  enable_palm_check: true\n"
        "recovery:\n  mode: nonsense\n  return_duration_s: 1.0\n"
        "ui:\n  sim_view_size: [640,480]\n  cam_view_size: [320,240]\n"
    )
    import pytest
    with pytest.raises(ValueError):
        load_config(bad)

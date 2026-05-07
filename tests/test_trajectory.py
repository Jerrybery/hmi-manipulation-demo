import math

import numpy as np

from hmi_demo.sim.trajectory import CircleTrajectory


def test_circle_starts_at_phase_zero():
    tr = CircleTrajectory(center=(0.5, 0.0, 0.3), radius=0.15, omega=0.6)
    p = tr.position()
    np.testing.assert_allclose(p, [0.65, 0.0, 0.3], atol=1e-9)


def test_circle_returns_to_start_after_full_period():
    tr = CircleTrajectory(center=(0.5, 0.0, 0.3), radius=0.15, omega=0.6)
    period = 2 * math.pi / 0.6
    dt = 0.001
    n = int(period / dt)
    for _ in range(n):
        tr.tick(dt)
    np.testing.assert_allclose(tr.position(), [0.65, 0.0, 0.3], atol=1e-3)


def test_tick_zero_dt_does_not_advance():
    tr = CircleTrajectory(center=(0.5, 0.0, 0.3), radius=0.15, omega=0.6)
    tr.tick(0.05)
    p_before = tr.position().copy()
    for _ in range(100):
        tr.tick(0.0)
    np.testing.assert_allclose(tr.position(), p_before)


def test_resume_advance_after_zero_dt():
    tr = CircleTrajectory(center=(0.5, 0.0, 0.3), radius=0.15, omega=0.6)
    tr.tick(0.1)
    p_at_pause = tr.position().copy()
    for _ in range(50):
        tr.tick(0.0)
    tr.tick(0.1)
    assert not np.allclose(tr.position(), p_at_pause)


def test_reset_returns_to_phase_zero():
    tr = CircleTrajectory(center=(0.5, 0.0, 0.3), radius=0.15, omega=0.6)
    tr.tick(1.234)
    tr.reset()
    np.testing.assert_allclose(tr.position(), [0.65, 0.0, 0.3], atol=1e-9)

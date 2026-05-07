import numpy as np

from hmi_demo.utils.gesture import (
    HysteresisFilter,
    is_open_hand,
    is_palm_facing_camera,
)


def _open_palm_facing_camera_landmarks() -> np.ndarray:
    """21-landmark array for an open right hand with palm facing the camera, in the XY plane."""
    lm = np.zeros((21, 3))
    lm[0] = [0.5, 0.7, 0.0]   # wrist
    lm[1] = [0.45, 0.65, 0.0]  # thumb_cmc
    lm[2] = [0.43, 0.60, 0.0]  # thumb_mcp
    lm[3] = [0.41, 0.55, 0.0]  # thumb_ip
    lm[4] = [0.39, 0.50, 0.0]  # thumb_tip (far from wrist)
    lm[5] = [0.55, 0.55, 0.0]  # index_mcp
    lm[6] = [0.55, 0.50, 0.0]
    lm[7] = [0.55, 0.45, 0.0]
    lm[8] = [0.55, 0.30, 0.0]  # index_tip
    lm[9] = [0.60, 0.55, 0.0]  # middle_mcp
    lm[10] = [0.60, 0.50, 0.0]
    lm[11] = [0.60, 0.45, 0.0]
    lm[12] = [0.60, 0.28, 0.0]  # middle_tip
    lm[13] = [0.65, 0.55, 0.0]  # ring_mcp
    lm[14] = [0.65, 0.50, 0.0]
    lm[15] = [0.65, 0.45, 0.0]
    lm[16] = [0.65, 0.30, 0.0]  # ring_tip
    lm[17] = [0.70, 0.55, 0.0]  # pinky_mcp
    lm[18] = [0.70, 0.50, 0.0]
    lm[19] = [0.70, 0.45, 0.0]
    lm[20] = [0.70, 0.32, 0.0]  # pinky_tip
    return lm


def test_palm_facing_camera_open_hand_right_returns_true():
    lm = _open_palm_facing_camera_landmarks()
    assert is_palm_facing_camera(lm, handedness="Right") is True


def test_palm_facing_camera_left_hand_mirrored_returns_true():
    lm = _open_palm_facing_camera_landmarks()
    lm_left = lm.copy()
    lm_left[:, 0] = 1.0 - lm_left[:, 0]
    assert is_palm_facing_camera(lm_left, handedness="Left") is True


def test_palm_facing_back_of_hand_returns_false():
    lm = _open_palm_facing_camera_landmarks()
    lm_flipped = lm.copy()
    pinky_indices = list(range(17, 21))
    index_indices = list(range(5, 9))
    lm_flipped[pinky_indices], lm_flipped[index_indices] = (
        lm[index_indices].copy(),
        lm[pinky_indices].copy(),
    )
    assert is_palm_facing_camera(lm_flipped, handedness="Right") is False


def test_open_hand_true_when_fingertips_far():
    lm = _open_palm_facing_camera_landmarks()
    assert is_open_hand(lm) is True


def test_open_hand_false_for_fist():
    lm = _open_palm_facing_camera_landmarks()
    for tip_idx in (8, 12, 16, 20):
        lm[tip_idx] = lm[0] + 0.01 * (lm[tip_idx] - lm[0])
    assert is_open_hand(lm) is False


def test_hysteresis_requires_5_consecutive_true_to_latch_on():
    f = HysteresisFilter(hold_frames=5)
    for _ in range(4):
        assert f.update(True) is False
    assert f.update(True) is True


def test_hysteresis_releases_after_5_consecutive_false():
    f = HysteresisFilter(hold_frames=5)
    for _ in range(5):
        f.update(True)
    assert f.state is True
    for _ in range(4):
        assert f.update(False) is True
    assert f.update(False) is False


def test_hysteresis_resets_streak_on_inconsistent_input():
    f = HysteresisFilter(hold_frames=5)
    f.update(True); f.update(True); f.update(True)
    f.update(False)
    for _ in range(4):
        assert f.update(True) is False
    assert f.update(True) is True

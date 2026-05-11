from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
FINGERTIPS = (8, 12, 16, 20)
FINGER_MCPS = (5, 9, 13, 17)


def is_palm_facing_camera(landmarks: np.ndarray, handedness: str) -> bool:
    """True iff the palm normal points toward the camera.

    Computes the palm-plane normal from wrist, index_mcp, pinky_mcp and flips
    sign per handedness so both hands' palms-facing-camera give the same result.
    Caller must pass the user's real-world handedness (post-flip if applicable).
    """
    wrist = landmarks[WRIST]
    v_index = landmarks[INDEX_MCP] - wrist
    v_pinky = landmarks[PINKY_MCP] - wrist
    normal = np.cross(v_index, v_pinky)
    sign = -1.0 if handedness == "Left" else 1.0
    return float(sign * normal[2]) > 0.0


def is_open_hand(landmarks: np.ndarray) -> bool:
    """True iff each fingertip is farther from the wrist than its corresponding MCP.

    Excludes thumb (index 4) because the thumb's MCP relationship is geometrically
    different and can be ambiguous when the hand rotates.
    """
    wrist = landmarks[WRIST]
    for tip, mcp in zip(FINGERTIPS, FINGER_MCPS):
        d_tip = np.linalg.norm(landmarks[tip] - wrist)
        d_mcp = np.linalg.norm(landmarks[mcp] - wrist)
        if d_tip <= d_mcp:
            return False
    return True


def is_closed_fist(landmarks: np.ndarray) -> bool:
    """True iff every (non-thumb) fingertip is closer to the wrist than its MCP.

    Mirrors `is_open_hand`: fingers curled inward → tip moves closer to wrist than MCP.
    A small slack factor (0.9x) is required so a relaxed/neutral hand pose does not
    trigger the fist detector — a hand only counts as a fist when fingertips are
    decisively pulled in toward the palm.
    """
    wrist = landmarks[WRIST]
    for tip, mcp in zip(FINGERTIPS, FINGER_MCPS):
        d_tip = np.linalg.norm(landmarks[tip] - wrist)
        d_mcp = np.linalg.norm(landmarks[mcp] - wrist)
        if d_tip >= 0.9 * d_mcp:
            return False
    return True


class HysteresisFilter:
    def __init__(self, hold_frames: int = 5):
        if hold_frames < 1:
            raise ValueError("hold_frames must be >= 1")
        self.hold_frames = hold_frames
        self.state = False
        self._streak = 0
        self._streak_value = False

    def update(self, raw: bool) -> bool:
        if raw == self._streak_value:
            self._streak += 1
        else:
            self._streak_value = raw
            self._streak = 1
        if self._streak_value != self.state and self._streak >= self.hold_frames:
            self.state = self._streak_value
        return self.state

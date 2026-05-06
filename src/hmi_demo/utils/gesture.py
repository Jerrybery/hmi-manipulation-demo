from __future__ import annotations

import numpy as np

WRIST = 0
INDEX_MCP = 5
PINKY_MCP = 17
FINGERTIPS = (8, 12, 16, 20)
FINGER_MCPS = (5, 9, 13, 17)


def is_palm_facing_camera(landmarks: np.ndarray, handedness: str) -> bool:
    """True iff the palm normal points toward the camera (-z in MediaPipe convention).

    Computes the palm-plane normal from wrist, index_mcp, pinky_mcp and flips
    sign per handedness so both hands' palms-facing-camera give the same result.
    """
    if landmarks.shape[0] < 21:
        return False
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
    if landmarks.shape[0] < 21:
        return False
    wrist = landmarks[WRIST]
    for tip, mcp in zip(FINGERTIPS, FINGER_MCPS):
        d_tip = np.linalg.norm(landmarks[tip] - wrist)
        d_mcp = np.linalg.norm(landmarks[mcp] - wrist)
        if d_tip <= d_mcp:
            return False
    return True


class HysteresisFilter:
    """N-consecutive-frames latch. Output toggles only after `hold_frames` of agreement."""

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

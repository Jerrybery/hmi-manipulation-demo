from __future__ import annotations

import math
from typing import Tuple

import numpy as np


class CircleTrajectory:
    """Parametric circle in the XY plane at fixed Z, downward EE pose assumed externally.

    Position at time t: center + (radius * cos(omega * t), radius * sin(omega * t), 0).
    Phase 0 puts the EE at center + [+radius, 0, 0].
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        radius: float,
        omega: float,
    ) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.omega = float(omega)
        self.t = 0.0
        self.frozen = False

    def tick(self, dt: float) -> np.ndarray:
        if not self.frozen:
            self.t += dt
        return self.position()

    def position(self) -> np.ndarray:
        a = self.omega * self.t
        return self.center + np.array(
            [self.radius * math.cos(a), self.radius * math.sin(a), 0.0]
        )

    def reset(self) -> None:
        self.t = 0.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import yaml

VALID_RECOVERY_MODES = ("return_home", "resume_in_place")


@dataclass(frozen=True)
class SimConfig:
    mjcf: str
    ee_site: str
    home_keyframe: str
    control_hz: int
    substeps: int


@dataclass(frozen=True)
class TrajectoryConfig:
    center: Tuple[float, float, float]
    radius: float
    omega: float


@dataclass(frozen=True)
class IKConfig:
    damping: float
    kp: float


@dataclass(frozen=True)
class CameraConfig:
    device: int
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class GestureConfig:
    hold_frames: int
    enable_palm_check: bool


@dataclass(frozen=True)
class RecoveryConfig:
    mode: str
    return_duration_s: float


@dataclass(frozen=True)
class UIConfig:
    sim_view_size: Tuple[int, int]
    cam_view_size: Tuple[int, int]


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    trajectory: TrajectoryConfig
    ik: IKConfig
    camera: CameraConfig
    gesture: GestureConfig
    recovery: RecoveryConfig
    ui: UIConfig


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    recovery_mode = raw["recovery"]["mode"]
    if recovery_mode not in VALID_RECOVERY_MODES:
        raise ValueError(
            f"recovery.mode must be one of {VALID_RECOVERY_MODES}, got {recovery_mode!r}"
        )
    return Config(
        sim=SimConfig(**raw["sim"]),
        trajectory=TrajectoryConfig(
            center=tuple(raw["trajectory"]["center"]),
            radius=float(raw["trajectory"]["radius"]),
            omega=float(raw["trajectory"]["omega"]),
        ),
        ik=IKConfig(**raw["ik"]),
        camera=CameraConfig(**raw["camera"]),
        gesture=GestureConfig(**raw["gesture"]),
        recovery=RecoveryConfig(
            mode=recovery_mode,
            return_duration_s=float(raw["recovery"]["return_duration_s"]),
        ),
        ui=UIConfig(
            sim_view_size=tuple(raw["ui"]["sim_view_size"]),
            cam_view_size=tuple(raw["ui"]["cam_view_size"]),
        ),
    )

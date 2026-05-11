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
class TrajectoryPreviewConfig:
    horizon_s: float
    n_samples: int
    sphere_radius: float
    alpha: float
    color_near: Tuple[float, float, float]
    color_far: Tuple[float, float, float]


@dataclass(frozen=True)
class IKConfig:
    damping: float
    kp: float


@dataclass(frozen=True)
class GripperConfig:
    open_ctrl: float
    close_ctrl: float


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
    enable_fist_resume: bool


@dataclass(frozen=True)
class RecoveryConfig:
    mode: str
    return_duration_s: float
    unload_pose: Tuple[float, float, float, float, float, float]
    unload_duration_s: float
    gripper_release_duration_s: float


@dataclass(frozen=True)
class UIConfig:
    sim_view_size: Tuple[int, int]
    cam_view_size: Tuple[int, int]


@dataclass(frozen=True)
class Config:
    sim: SimConfig
    trajectory: TrajectoryConfig
    trajectory_preview: TrajectoryPreviewConfig
    ik: IKConfig
    gripper: GripperConfig
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

    unload_pose = tuple(float(v) for v in raw["recovery"]["unload_pose"])
    if len(unload_pose) != 6:
        raise ValueError(
            f"recovery.unload_pose must have exactly 6 elements, got {len(unload_pose)}"
        )

    return Config(
        sim=SimConfig(**raw["sim"]),
        trajectory=TrajectoryConfig(
            center=tuple(float(v) for v in raw["trajectory"]["center"]),
            radius=float(raw["trajectory"]["radius"]),
            omega=float(raw["trajectory"]["omega"]),
        ),
        trajectory_preview=TrajectoryPreviewConfig(
            horizon_s=float(raw["trajectory_preview"]["horizon_s"]),
            n_samples=int(raw["trajectory_preview"]["n_samples"]),
            sphere_radius=float(raw["trajectory_preview"]["sphere_radius"]),
            alpha=float(raw["trajectory_preview"]["alpha"]),
            color_near=tuple(float(v) for v in raw["trajectory_preview"]["color_near"]),
            color_far=tuple(float(v) for v in raw["trajectory_preview"]["color_far"]),
        ),
        ik=IKConfig(**raw["ik"]),
        gripper=GripperConfig(
            open_ctrl=float(raw["gripper"]["open_ctrl"]),
            close_ctrl=float(raw["gripper"]["close_ctrl"]),
        ),
        camera=CameraConfig(**raw["camera"]),
        gesture=GestureConfig(**raw["gesture"]),
        recovery=RecoveryConfig(
            mode=recovery_mode,
            return_duration_s=float(raw["recovery"]["return_duration_s"]),
            unload_pose=unload_pose,
            unload_duration_s=float(raw["recovery"]["unload_duration_s"]),
            gripper_release_duration_s=float(raw["recovery"]["gripper_release_duration_s"]),
        ),
        ui=UIConfig(
            sim_view_size=tuple(raw["ui"]["sim_view_size"]),
            cam_view_size=tuple(raw["ui"]["cam_view_size"]),
        ),
    )

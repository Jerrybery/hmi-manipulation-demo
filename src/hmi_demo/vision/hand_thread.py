from __future__ import annotations

import logging
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from hmi_demo.config import CameraConfig, GestureConfig
from hmi_demo.utils.gesture import (
    HysteresisFilter,
    is_closed_fist,
    is_open_hand,
    is_palm_facing_camera,
)

_LOG = logging.getLogger(__name__)

_MP_FAILURE_THRESHOLD = 10
_CAM_SOFT_THRESHOLD = 5
_CAM_HARD_THRESHOLD = 30

_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
_HAND_LANDMARKER_FILENAME = "hand_landmarker.task"

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


def _ensure_hand_landmarker_model() -> Path:
    cache_dir = Path.home() / ".cache" / "hmi-demo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / _HAND_LANDMARKER_FILENAME
    if target.exists():
        return target
    _LOG.info("Downloading hand_landmarker.task to %s", target)
    urllib.request.urlretrieve(_HAND_LANDMARKER_URL, target)
    return target


class VisionThread(QThread):
    # Emits the currently-latched gesture name ("open_palm" / "closed_fist" / "") and the annotated frame.
    # Empty string means no stable gesture.
    gestureUpdated = pyqtSignal(str, QImage)
    cameraError = pyqtSignal(str)

    def __init__(self, cam_cfg: CameraConfig, gesture_cfg: GestureConfig, parent=None):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.gesture_cfg = gesture_cfg
        self._stop = False
        self._palm_filter = HysteresisFilter(hold_frames=gesture_cfg.hold_frames)
        self._fist_filter = HysteresisFilter(hold_frames=gesture_cfg.hold_frames)
        self._mp_failures = 0

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            model_path = _ensure_hand_landmarker_model()
        except (OSError, urllib.error.URLError) as exc:
            self.cameraError.emit(f"Failed to fetch hand-landmarker model: {exc}")
            return

        cap = cv2.VideoCapture(self.cam_cfg.device)
        if not cap.isOpened():
            self.cameraError.emit(f"Cannot open camera device {self.cam_cfg.device}")
            return
        ok_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.width)
        ok_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.height)
        ok_f = cap.set(cv2.CAP_PROP_FPS, self.cam_cfg.fps)
        if not (ok_w and ok_h and ok_f):
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            _LOG.warning(
                "Camera device %d: requested %dx%d@%dfps but got %dx%d@%.1ffps",
                self.cam_cfg.device,
                self.cam_cfg.width, self.cam_cfg.height, self.cam_cfg.fps,
                actual_w, actual_h, actual_fps,
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=mp_vision.RunningMode.VIDEO,
        )

        consecutive_failures = 0
        try:
            with mp_vision.HandLandmarker.create_from_options(options) as detector:
                start_time = time.monotonic()
                while not self._stop:
                    ok, frame = cap.read()
                    if not ok:
                        consecutive_failures += 1
                        if consecutive_failures == 1:
                            _LOG.warning(
                                "Camera device %d: cap.read() returned False",
                                self.cam_cfg.device,
                            )
                        if consecutive_failures == _CAM_SOFT_THRESHOLD:
                            self.cameraError.emit(
                                f"Camera device {self.cam_cfg.device}: dropped "
                                f"{_CAM_SOFT_THRESHOLD} consecutive frames (may be transient)"
                            )
                        if consecutive_failures >= _CAM_HARD_THRESHOLD:
                            self.cameraError.emit(
                                f"Camera device {self.cam_cfg.device}: read failed "
                                f"{consecutive_failures} frames in a row — check USB connection "
                                f"and try restarting"
                            )
                            return
                        continue
                    consecutive_failures = 0
                    # Mirror horizontally for natural selfie display. MediaPipe's handedness
                    # label is image-relative; we invert it in _evaluate to recover real-world handedness.
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int((time.monotonic() - start_time) * 1000)

                    try:
                        result = detector.detect_for_video(mp_image, timestamp_ms)
                        self._mp_failures = 0
                    except RuntimeError as exc:
                        self._mp_failures += 1
                        _LOG.warning(
                            "MediaPipe detect_for_video() failed (consecutive=%d): %s",
                            self._mp_failures, exc,
                        )
                        if self._mp_failures >= _MP_FAILURE_THRESHOLD:
                            self.cameraError.emit(
                                f"MediaPipe failed {_MP_FAILURE_THRESHOLD} frames in a row: {exc}"
                            )
                            return
                        continue

                    raw = self._evaluate(result)
                    # Each gesture has its own filter so transitioning between gestures
                    # doesn't bleed across hysteresis windows.
                    palm_stable = self._palm_filter.update(raw == "open_palm")
                    fist_stable = self._fist_filter.update(raw == "closed_fist")

                    if palm_stable:
                        stable_name = "open_palm"
                    elif fist_stable:
                        stable_name = "closed_fist"
                    else:
                        stable_name = ""

                    annotated = self._annotate(rgb, result)
                    qimg = self._to_qimage(annotated)
                    self.gestureUpdated.emit(stable_name, qimg)
        finally:
            cap.release()

    def _evaluate(self, result) -> str:
        if not result.hand_landmarks:
            return ""
        landmarks_list = result.hand_landmarks[0]
        lm = np.array([[p.x, p.y, p.z] for p in landmarks_list])

        # Closed fist takes priority if enabled — a strongly curled hand should not
        # be ambiguously read as "almost open" by is_open_hand if both happened to be False.
        if self.gesture_cfg.enable_fist_resume and is_closed_fist(lm):
            return "closed_fist"

        if is_open_hand(lm):
            if self.gesture_cfg.enable_palm_check:
                handedness_label = "Right"
                if result.handedness:
                    raw = result.handedness[0][0].category_name
                    handedness_label = "Left" if raw == "Right" else "Right"
                if not is_palm_facing_camera(lm, handedness_label):
                    return ""
            return "open_palm"

        return ""

    def _annotate(self, rgb: np.ndarray, result) -> np.ndarray:
        annotated = rgb.copy()
        if not result.hand_landmarks:
            return annotated
        h, w, _ = annotated.shape
        for hand_lms in result.hand_landmarks:
            for a, b in _HAND_CONNECTIONS:
                x0, y0 = int(hand_lms[a].x * w), int(hand_lms[a].y * h)
                x1, y1 = int(hand_lms[b].x * w), int(hand_lms[b].y * h)
                cv2.line(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for p in hand_lms:
                x, y = int(p.x * w), int(p.y * h)
                cv2.circle(annotated, (x, y), 4, (255, 0, 0), -1)
        return annotated

    @staticmethod
    def _to_qimage(rgb: np.ndarray) -> QImage:
        h, w, _ = rgb.shape
        return QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()

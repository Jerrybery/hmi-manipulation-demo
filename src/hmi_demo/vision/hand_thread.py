from __future__ import annotations

import logging

import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage

from hmi_demo.config import CameraConfig, GestureConfig
from hmi_demo.utils.gesture import (
    HysteresisFilter,
    is_open_hand,
    is_palm_facing_camera,
)

_LOG = logging.getLogger(__name__)

_MP_FAILURE_THRESHOLD = 10
_CAM_SOFT_THRESHOLD = 5
_CAM_HARD_THRESHOLD = 30


class VisionThread(QThread):
    gestureUpdated = pyqtSignal(bool, QImage)
    cameraError = pyqtSignal(str)

    def __init__(self, cam_cfg: CameraConfig, gesture_cfg: GestureConfig, parent=None):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.gesture_cfg = gesture_cfg
        self._stop = False
        self._filter = HysteresisFilter(hold_frames=gesture_cfg.hold_frames)
        self._mp_failures = 0

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
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

        consecutive_failures = 0
        try:
            with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            ) as hands:
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

                    try:
                        result = hands.process(rgb)
                        self._mp_failures = 0
                    except RuntimeError as exc:
                        self._mp_failures += 1
                        _LOG.warning(
                            "MediaPipe hands.process() failed (consecutive=%d): %s",
                            self._mp_failures, exc,
                        )
                        if self._mp_failures >= _MP_FAILURE_THRESHOLD:
                            self.cameraError.emit(
                                f"MediaPipe failed {_MP_FAILURE_THRESHOLD} frames in a row: {exc}"
                            )
                            return
                        continue

                    raw_raised = self._evaluate(result)
                    stable = self._filter.update(raw_raised)
                    annotated = self._annotate(rgb, result)
                    qimg = self._to_qimage(annotated)

                    self.gestureUpdated.emit(stable, qimg)
        finally:
            cap.release()

    def _evaluate(self, result) -> bool:
        if not result.multi_hand_landmarks:
            return False
        landmarks_norm = result.multi_hand_landmarks[0].landmark
        lm = np.array([[p.x, p.y, p.z] for p in landmarks_norm])
        if not is_open_hand(lm):
            return False
        if self.gesture_cfg.enable_palm_check:
            # MediaPipe's handedness is based on the image. Frame is horizontally
            # flipped above for natural display, so MP's "Right" label maps to the
            # user's physical left hand and vice versa. Invert here so callers see
            # real-world handedness.
            handedness_label = "Right"  # default if MP didn't classify yet
            if result.multi_handedness:
                raw = result.multi_handedness[0].classification[0].label
                handedness_label = "Left" if raw == "Right" else "Right"
            if not is_palm_facing_camera(lm, handedness_label):
                return False
        return True

    def _annotate(self, rgb: np.ndarray, result) -> np.ndarray:
        annotated = rgb.copy()
        if result.multi_hand_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles
            for hand in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
        return annotated

    @staticmethod
    def _to_qimage(rgb: np.ndarray) -> QImage:
        h, w, _ = rgb.shape
        return QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888
        ).copy()

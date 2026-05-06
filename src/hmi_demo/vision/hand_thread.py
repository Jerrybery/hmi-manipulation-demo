from __future__ import annotations

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


class VisionThread(QThread):
    gestureUpdated = pyqtSignal(bool, QImage)
    cameraError = pyqtSignal(str)

    def __init__(self, cam_cfg: CameraConfig, gesture_cfg: GestureConfig, parent=None):
        super().__init__(parent)
        self.cam_cfg = cam_cfg
        self.gesture_cfg = gesture_cfg
        self._stop = False
        self._filter = HysteresisFilter(hold_frames=gesture_cfg.hold_frames)

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cam_cfg.device)
        if not cap.isOpened():
            self.cameraError.emit(f"Cannot open camera device {self.cam_cfg.device}")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_cfg.height)
        cap.set(cv2.CAP_PROP_FPS, self.cam_cfg.fps)

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
                        if consecutive_failures >= 30:
                            self.cameraError.emit("Camera read failed 30 frames in a row")
                            return
                        continue
                    consecutive_failures = 0
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    try:
                        result = hands.process(rgb)
                    except Exception:
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
            handedness_label = "Right"
            if result.multi_handedness:
                handedness_label = result.multi_handedness[0].classification[0].label
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

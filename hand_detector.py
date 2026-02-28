"""
Hand detection using the MediaPipe Tasks API (mediapipe >= 0.10.14).

The model file 'hand_landmarker.task' is downloaded automatically on first run.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Connections between the 21 hand landmarks for drawing
_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[HandDetector] Downloading model from MediaPipe CDN …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[HandDetector] Model saved to '{MODEL_PATH}'.")


class HandDetector:
    """Detects and annotates hands using the MediaPipe Tasks Hand Landmarker."""

    def __init__(self, max_hands: int = 2):
        _ensure_model()

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._start_ms = int(time.time() * 1000)

    def detect(self, frame):
        """
        Process a BGR frame and return (hands_data, annotated_frame).

        Each element of hands_data is a dict:
            landmarks  : list of (x, y) pixel coords for all 21 landmarks
            handedness : 'Left' or 'Right'
            wrist      : (x, y) of landmark 0
        """
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000) - self._start_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        hands_data = []
        for lm_list, handedness_list in zip(
            result.hand_landmarks, result.handedness
        ):
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
            hands_data.append(
                {
                    "landmarks": landmarks,
                    "handedness": handedness_list[0].category_name,
                    "wrist": landmarks[0],
                }
            )
            self._draw_hand(frame, landmarks)

        return hands_data, frame

    @staticmethod
    def _draw_hand(frame, landmarks: list):
        for start, end in _CONNECTIONS:
            cv2.line(frame, landmarks[start], landmarks[end], (0, 220, 0), 2)
        for pt in landmarks:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1)
        # Highlight wrist and fingertips
        for tip in [0, 4, 8, 12, 16, 20]:
            cv2.circle(frame, landmarks[tip], 6, (0, 180, 255), -1)

    def release(self):
        self._landmarker.close()

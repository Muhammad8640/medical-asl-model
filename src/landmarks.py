"""Dataset-independent MediaPipe hand and upper-body pose extraction."""

from __future__ import annotations

from typing import Any

import cv2
import mediapipe as mp
import numpy as np


POSE_LANDMARK_INDICES = (
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value,
    mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
)


def _landmark_array(landmark_list: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((count, 3), dtype=np.float32)
    present = np.zeros(count, dtype=bool)
    if landmark_list is None:
        return values, present
    for index, landmark in enumerate(landmark_list.landmark[:count]):
        values[index] = (landmark.x, landmark.y, landmark.z)
        present[index] = True
    return values, present


class MediaPipeLandmarkExtractor:
    """Stateful extractor so tracking state can be reused across video frames."""

    def __init__(self) -> None:
        self._holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract_landmarks(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Extract anatomical left/right hands and the six required pose joints."""
        if frame is None or frame.ndim != 3:
            raise ValueError("Expected a non-empty BGR image")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._holistic.process(rgb)

        left_hand, left_present = _landmark_array(result.left_hand_landmarks, 21)
        right_hand, right_present = _landmark_array(result.right_hand_landmarks, 21)

        pose = np.zeros((6, 3), dtype=np.float32)
        pose_present = np.zeros(6, dtype=bool)
        if result.pose_landmarks is not None:
            all_pose = result.pose_landmarks.landmark
            for output_index, source_index in enumerate(POSE_LANDMARK_INDICES):
                landmark = all_pose[source_index]
                pose[output_index] = (landmark.x, landmark.y, landmark.z)
                pose_present[output_index] = True

        return {
            "left_hand": left_hand,
            "left_hand_present": left_present,
            "right_hand": right_hand,
            "right_hand_present": right_present,
            "pose": pose,
            "pose_present": pose_present,
        }

    def close(self) -> None:
        self._holistic.close()

    def __enter__(self) -> "MediaPipeLandmarkExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def extract_landmarks(
    frame: np.ndarray, extractor: MediaPipeLandmarkExtractor | None = None
) -> dict[str, np.ndarray]:
    """Convenience wrapper; reuse an extractor for multi-frame sequences."""
    if extractor is not None:
        return extractor.extract_landmarks(frame)
    with MediaPipeLandmarkExtractor() as temporary:
        return temporary.extract_landmarks(frame)

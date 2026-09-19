"""Reusable normalization and temporal preprocessing for Medical ASL V1."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from landmarks import MediaPipeLandmarkExtractor


SEQUENCE_LENGTH = 30
HAND_LANDMARKS = 21
POSE_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
FEATURES_PER_FRAME = (HAND_LANDMARKS * 2 + len(POSE_NAMES)) * 3
EPSILON = 1e-6


def feature_layout() -> list[str]:
    """Return the exact, stable per-frame feature ordering."""
    layout: list[str] = []
    for hand in ("left_hand", "right_hand"):
        for index in range(HAND_LANDMARKS):
            layout.extend(f"{hand}_{index}_{axis}" for axis in ("x", "y", "z"))
    for name in POSE_NAMES:
        layout.extend(f"{name}_{axis}" for axis in ("x", "y", "z"))
    return layout


def _fallback_origin_scale(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    """Return a finite origin/scale when both shoulders are not usable.

    The fallback uses all available required pose points. If none exist, it uses
    all available hand points. Their centroid is the origin and the maximum
    Euclidean distance from it is the scale. With fewer than two distinct
    points, the scale is 1.0.
    """
    pose = raw["pose"]
    pose_present = raw["pose_present"].astype(bool)
    points = pose[pose_present]
    if points.size == 0:
        hand_points = []
        for side in ("left_hand", "right_hand"):
            present = raw[f"{side}_present"].astype(bool)
            if present.any():
                hand_points.append(raw[side][present])
        points = np.concatenate(hand_points, axis=0) if hand_points else np.empty((0, 3))

    if points.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    origin = points.mean(axis=0, dtype=np.float64).astype(np.float32)
    distances = np.linalg.norm(points - origin, axis=1)
    scale = float(distances.max(initial=0.0))
    return origin, scale if np.isfinite(scale) and scale > EPSILON else 1.0


def normalize_landmarks(raw: dict[str, np.ndarray]) -> np.ndarray:
    """Normalize one structured landmark result into exactly 144 float values.

    Detected points are translated by a body-relative origin and divided by a
    body-relative scale. Missing landmark slots remain exactly zero.
    """
    pose = np.asarray(raw["pose"], dtype=np.float32)
    pose_present = np.asarray(raw["pose_present"], dtype=bool)

    if pose_present[0] and pose_present[1]:
        left_shoulder, right_shoulder = pose[0], pose[1]
        shoulder_distance = float(np.linalg.norm(left_shoulder - right_shoulder))
    else:
        shoulder_distance = 0.0

    if np.isfinite(shoulder_distance) and shoulder_distance > EPSILON:
        origin = (pose[0] + pose[1]) / 2.0
        scale = shoulder_distance
    else:
        origin, scale = _fallback_origin_scale(raw)

    normalized_parts: list[np.ndarray] = []
    for name in ("left_hand", "right_hand", "pose"):
        values = np.asarray(raw[name], dtype=np.float32)
        present = np.asarray(raw[f"{name}_present"], dtype=bool)
        normalized = np.zeros_like(values, dtype=np.float32)
        normalized[present] = (values[present] - origin) / scale
        normalized_parts.append(normalized.reshape(-1))

    vector = np.concatenate(normalized_parts).astype(np.float32, copy=False)
    if vector.shape != (FEATURES_PER_FRAME,):
        raise ValueError(f"Expected {FEATURES_PER_FRAME} features, got {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("Normalization produced NaN or infinite values")
    return vector


def resample_sequence(sequence: np.ndarray, length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Deterministically convert an N x F sequence to ``length`` timesteps."""
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("A non-empty 2D landmark sequence is required")
    if values.shape[0] >= length:
        indices = np.linspace(0, values.shape[0] - 1, length, dtype=np.int64)
        return values[indices]
    if values.shape[0] == 1:
        return np.repeat(values, length, axis=0)

    source_times = np.linspace(0.0, 1.0, values.shape[0])
    target_times = np.linspace(0.0, 1.0, length)
    result = np.empty((length, values.shape[1]), dtype=np.float32)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(target_times, source_times, values[:, column])
    return result


def preprocess_sequence(
    frames: Sequence[np.ndarray] | Iterable[np.ndarray],
    extractor: "MediaPipeLandmarkExtractor | None" = None,
) -> np.ndarray:
    """Convert isolated-sign BGR frames to a fixed 30 x 144 sequence.

    This interface is source-agnostic and can later receive camera frames. For
    long inputs, frames are sampled before MediaPipe work; short inputs are
    landmark-extracted first and linearly interpolated afterward.
    """
    from landmarks import MediaPipeLandmarkExtractor

    frame_list = list(frames)
    if not frame_list:
        raise ValueError("Cannot preprocess an empty frame sequence")

    if len(frame_list) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(frame_list) - 1, SEQUENCE_LENGTH, dtype=np.int64)
        selected = [frame_list[int(index)] for index in indices]
    else:
        selected = frame_list

    owns_extractor = extractor is None
    active_extractor = extractor or MediaPipeLandmarkExtractor()
    try:
        values = np.stack(
            [normalize_landmarks(active_extractor.extract_landmarks(frame)) for frame in selected]
        )
    finally:
        if owns_extractor:
            active_extractor.close()
    return resample_sequence(values, SEQUENCE_LENGTH)


def flatten_sequence(sequence: np.ndarray) -> np.ndarray:
    """Flatten in timestep-major, feature-layout-minor C order."""
    values = np.asarray(sequence, dtype=np.float32)
    expected = (SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if values.shape != expected:
        raise ValueError(f"Expected sequence shape {expected}, got {values.shape}")
    return values.reshape(-1, order="C")

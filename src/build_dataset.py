"""Build fixed-size Medical-MS-ASL NumPy datasets with official splits."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from download_videos import find_video
from landmarks import MediaPipeLandmarkExtractor
from preprocessing import FEATURES_PER_FRAME, SEQUENCE_LENGTH, preprocess_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_USABLE_SAMPLES_PER_CLASS = 5
VALID_SPLITS = ("train", "val", "test")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() and default is not None:
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def read_annotated_frames(path: Path, instance: dict[str, Any]) -> tuple[list[np.ndarray], str | None]:
    """Read a clip already cut to MS-ASL start_time/end_time at acquisition."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not open the video")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError("no decodable frames in the selected sign interval")
    return frames, None


def _empty_x() -> np.ndarray:
    return np.empty((0, SEQUENCE_LENGTH, FEATURES_PER_FRAME), dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "data/medical_subset.json")
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data/videos")
    parser.add_argument("--processed", type=Path, default=PROJECT_ROOT / "processed")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "logs/dataset_report.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.processed.mkdir(parents=True, exist_ok=True)

    subset = _read_json(args.subset)
    samples: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    failures: list[dict[str, str]] = []
    boundary_fallbacks: list[dict[str, str]] = []
    attempted_with_video = 0

    with MediaPipeLandmarkExtractor() as extractor:
        for entry in subset:
            for instance in entry.get("instances", []):
                gloss = str(instance["gloss"])
                video_id = str(instance["video_id"])
                split = str(instance.get("split", ""))
                if split not in VALID_SPLITS:
                    reason = f"unsupported split: {split!r}"
                    logging.error("MediaPipe skipped %s (%s): %s", video_id, gloss, reason)
                    failures.append({"video_id": video_id, "gloss": gloss, "reason": reason})
                    continue
                video_path = find_video(args.videos, video_id)
                if video_path is None:
                    continue  # already accounted for by acquisition reporting
                attempted_with_video += 1
                try:
                    frames, fallback = read_annotated_frames(video_path, instance)
                    if fallback:
                        logging.warning("Boundary fallback for %s: %s", video_id, fallback)
                        boundary_fallbacks.append({"video_id": video_id, "reason": fallback})
                    sequence = preprocess_sequence(frames, extractor)
                    samples[gloss].append((split, sequence))
                    logging.info("Processed %s (%s, %s)", video_id, gloss, split)
                except Exception as error:  # one corrupt/detection failure must not stop the run
                    logging.error("MediaPipe failed for %s (%s): %s", video_id, gloss, error)
                    failures.append({"video_id": video_id, "gloss": gloss, "reason": str(error)})

    usable_counts = {entry["gloss"]: len(samples.get(entry["gloss"], [])) for entry in subset}
    included_classes = sorted(
        gloss for gloss, count in usable_counts.items() if count >= MIN_USABLE_SAMPLES_PER_CLASS
    )
    excluded = []
    for gloss, count in usable_counts.items():
        if count < MIN_USABLE_SAMPLES_PER_CLASS:
            item = {
                "class": gloss,
                "reason": "insufficient usable samples",
                "usable_samples": count,
                "minimum_required": MIN_USABLE_SAMPLES_PER_CLASS,
            }
            excluded.append(item)
            logging.warning(
                "Excluded class: %s; usable samples: %d; minimum: %d",
                gloss, count, MIN_USABLE_SAMPLES_PER_CLASS,
            )

    by_split: dict[str, list[tuple[np.ndarray, str]]] = {name: [] for name in VALID_SPLITS}
    for gloss in included_classes:
        for split, sequence in samples[gloss]:
            by_split[split].append((sequence, gloss))

    for split in VALID_SPLITS:
        split_samples = by_split[split]
        x = np.stack([item[0] for item in split_samples]) if split_samples else _empty_x()
        y = np.asarray([item[1] for item in split_samples], dtype=str)
        np.save(args.processed / f"X_{split}.npy", x, allow_pickle=False)
        np.save(args.processed / f"y_{split}.npy", y, allow_pickle=False)
        logging.info("Saved %s split: %d samples", split, len(y))
    _write_json(args.processed / "classes.json", included_classes)

    report = _read_json(args.report, {})
    report.update(
        {
            "mediapipe_success": sum(usable_counts.values()),
            "mediapipe_failed": len(failures),
            "mediapipe_attempted": attempted_with_video,
            "mediapipe_failures": failures,
            "boundary_fallbacks": boundary_fallbacks,
            "usable_samples_by_class": usable_counts,
            "excluded_classes": excluded,
            "included_classes": included_classes,
            "split_samples": {split: len(by_split[split]) for split in VALID_SPLITS},
        }
    )
    _write_json(args.report, report)
    logging.info("Included classes: %s", included_classes or "none")


if __name__ == "__main__":
    main()

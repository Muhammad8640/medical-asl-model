"""Select exact medical glosses from the official MS-ASL split metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import urllib.parse
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIRED_MEDICAL_GLOSSES = [
    "help", "yes", "no", "pain", "hurt", "sick", "medicine", "doctor",
    "hospital", "blood", "head", "face", "eye", "ear", "mouth", "neck",
    "chest", "heart", "stomach", "back", "arm", "hand", "finger", "leg",
    "knee", "foot", "breathe", "eat", "drink", "sleep", "where", "when",
]
SPLITS = ("train", "val", "test")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _source_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if "youtu.be" in parsed.netloc:
        candidate = parsed.path.strip("/").split("/")[0]
    elif "youtube.com" in parsed.netloc:
        candidate = urllib.parse.parse_qs(parsed.query).get("v", [""])[0]
    else:
        candidate = ""
    if re.fullmatch(r"[A-Za-z0-9_-]{6,20}", candidate):
        return candidate
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def select_subset(metadata_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    desired = set(DESIRED_MEDICAL_GLOSSES)
    by_gloss: dict[str, list[dict[str, Any]]] = {gloss: [] for gloss in DESIRED_MEDICAL_GLOSSES}

    for split in SPLITS:
        records = _read_json(metadata_dir / f"MSASL_{split}.json")
        for source_index, record in enumerate(records):
            gloss = str(record.get("clean_text", record.get("text", ""))).strip().lower()
            if gloss not in desired:
                continue
            url = str(record["url"])
            by_gloss[gloss].append({
                "gloss": gloss,
                "video_id": f"msasl_{split}_{source_index:06d}",
                "source_id": _source_id(url),
                "source_file": record.get("file"),
                "url": url,
                "signer_id": record.get("signer_id"),
                "split": split,
                "start_time": record.get("start_time"),
                "end_time": record.get("end_time"),
                "frame_start": record.get("start"),
                "frame_end": record.get("end"),
                "bbox": record.get("box"),
                "fps": record.get("fps"),
                "width": record.get("width"),
                "height": record.get("height"),
                "msasl_label": record.get("label"),
                "original_text": record.get("org_text"),
            })

    subset = [
        {"gloss": gloss, "instances": by_gloss[gloss]}
        for gloss in DESIRED_MEDICAL_GLOSSES
        if by_gloss[gloss]
    ]
    missing = [gloss for gloss in DESIRED_MEDICAL_GLOSSES if not by_gloss[gloss]]
    for entry in subset:
        split_counts = {
            split: sum(item["split"] == split for item in entry["instances"])
            for split in SPLITS
        }
        logging.info("Available gloss %-12s %d instances %s", entry["gloss"], len(entry["instances"]), split_counts)
    for gloss in missing:
        logging.warning("Missing requested MS-ASL gloss: %s", gloss)
    return subset, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data/msasl")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/medical_subset.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "logs/dataset_report.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    subset, missing = select_subset(args.metadata_dir)
    _write_json(args.output, subset)
    available = [entry["gloss"] for entry in subset]
    instance_counts = {entry["gloss"]: len(entry["instances"]) for entry in subset}
    split_counts = {
        split: sum(item["split"] == split for entry in subset for item in entry["instances"])
        for split in SPLITS
    }
    unique_sources = {item["source_id"] for entry in subset for item in entry["instances"]}
    report = {
        "dataset": "MS-ASL",
        "requested_classes": DESIRED_MEDICAL_GLOSSES,
        "available_classes": available,
        "missing_classes": missing,
        "metadata_instances_by_class": instance_counts,
        "metadata_instances_by_split": split_counts,
        "excluded_classes": [],
        "videos_requested": sum(instance_counts.values()),
        "unique_sources_referenced": len(unique_sources),
        "videos_downloaded": 0,
        "videos_unavailable": 0,
        "mediapipe_success": 0,
        "mediapipe_failed": 0,
        "usable_samples_by_class": {},
    }
    _write_json(args.report, report)
    logging.info("Saved %d MS-ASL classes and %d instances to %s", len(subset), sum(instance_counts.values()), args.output)


if __name__ == "__main__":
    main()


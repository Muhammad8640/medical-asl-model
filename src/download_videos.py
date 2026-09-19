"""Download and temporally clip every selected MS-ASL instance."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".avi")


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


def find_video(directory: Path, video_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = directory / f"{video_id}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def validate_video(path: Path) -> tuple[bool, str]:
    """Reject HTML/error bodies and files OpenCV cannot decode."""
    try:
        with path.open("rb") as handle:
            prefix = handle.read(256).lstrip().lower()
        if prefix.startswith((b"<!doctype html", b"<html")):
            return False, "server returned HTML instead of video"
    except OSError as error:
        return False, f"could not inspect file: {error}"
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    ok, frame = capture.read() if opened else (False, None)
    capture.release()
    if not opened or not ok or frame is None:
        return False, "OpenCV could not decode the downloaded clip"
    return True, ""


def _time_range(instance: dict[str, Any]) -> tuple[float, float]:
    try:
        start = float(instance["start_time"])
        end = float(instance["end_time"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("missing or invalid MS-ASL time boundaries") from error
    if start < 0 or end <= start:
        raise ValueError(f"invalid MS-ASL time interval: {start}..{end}")
    return start, end


def _ffmpeg_executable(directory: Path) -> Path:
    """Expose imageio's versioned binary under the name yt-dlp expects."""
    bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    tool_dir = directory / ".ffmpeg"
    tool_dir.mkdir(parents=True, exist_ok=True)
    target = tool_dir / ("ffmpeg.exe" if bundled.suffix.lower() == ".exe" else "ffmpeg")
    if not target.exists() or target.stat().st_size != bundled.stat().st_size:
        shutil.copy2(bundled, target)
    return target.resolve()


def download_instance(instance: dict[str, Any], directory: Path) -> Path:
    video_id = str(instance["video_id"])
    start, end = _time_range(instance)
    ffmpeg_executable = _ffmpeg_executable(directory)
    options = {
        "outtmpl": str(directory / f"{video_id}.%(ext)s"),
        "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": str(ffmpeg_executable),
        "external_downloader": "ffmpeg",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
    }
    # yt-dlp's partial-download availability check searches PATH even when
    # ffmpeg_location is provided. Expose the bundled binary for that check.
    ffmpeg_directory = str(ffmpeg_executable.parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_directory not in path_entries:
        os.environ["PATH"] = os.pathsep.join((ffmpeg_directory, *path_entries))
    with YoutubeDL(options) as downloader:
        downloader.extract_info(str(instance["url"]), download=True)
    result = find_video(directory, video_id)
    if result is None:
        raise RuntimeError("yt-dlp returned without creating a clip")
    valid, reason = validate_video(result)
    if not valid:
        result.unlink(missing_ok=True)
        raise RuntimeError(reason)
    return result


def download_source(instances: list[dict[str, Any]], directory: Path) -> tuple[list[Path], list[dict[str, str]]]:
    """Download every requested interval from one source with one extraction request."""
    if not instances:
        return [], []
    source_id = str(instances[0]["source_id"])
    url = str(instances[0]["url"])
    staging = directory / ".sections"
    staging.mkdir(parents=True, exist_ok=True)
    ffmpeg_executable = _ffmpeg_executable(directory)
    ffmpeg_directory = str(ffmpeg_executable.parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if ffmpeg_directory not in path_entries:
        os.environ["PATH"] = os.pathsep.join((ffmpeg_directory, *path_entries))
    ranges = [_time_range(instance) for instance in instances]
    options = {
        "outtmpl": str(staging / f"{source_id}_%(section_start)s-%(section_end)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]/bestvideo/best[ext=mp4]/best",
        "download_ranges": download_range_func(None, ranges),
        "force_keyframes_at_cuts": True,
        "ffmpeg_location": str(ffmpeg_executable),
        "external_downloader": "ffmpeg",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 2,
        "fragment_retries": 2,
        "sleep_interval_requests": 0.75,
    }
    extraction_error = ""
    try:
        # YouTube documents a guest-session rate of roughly 300 videos/hour.
        # One grouped request represents one source video, regardless of how
        # many annotated sign intervals it contains.
        time.sleep(6)
        with YoutubeDL(options) as downloader:
            downloader.extract_info(url, download=True)
    except Exception as error:
        extraction_error = str(error)

    downloaded: list[Path] = []
    unavailable: list[dict[str, str]] = []
    for instance, (start, end) in zip(instances, ranges):
        prefix = f"{source_id}_{start}-{end}."
        section = next((path for path in staging.glob(f"{source_id}_*") if path.name.startswith(prefix)), None)
        reason = extraction_error or "yt-dlp returned without creating the annotated section"
        if section is not None:
            valid, validation_reason = validate_video(section)
            if valid:
                target = directory / f"{instance['video_id']}{section.suffix}"
                section.replace(target)
                downloaded.append(target)
                continue
            section.unlink(missing_ok=True)
            reason = validation_reason
        unavailable.append({
            "video_id": str(instance["video_id"]),
            "source_id": source_id,
            "gloss": str(instance["gloss"]),
            "split": str(instance["split"]),
            "reason": reason,
        })

    # A bad interval can make ffmpeg abort the remaining intervals in a grouped
    # request. Retry only that narrow failure mode one interval at a time.
    if "ffmpeg exited" in extraction_error.lower() and unavailable:
        instances_by_id = {str(instance["video_id"]): instance for instance in instances}
        for failed in list(unavailable):
            try:
                time.sleep(2)
                recovered = download_instance(instances_by_id[failed["video_id"]], directory)
            except Exception as error:
                failed["reason"] = str(error)
            else:
                downloaded.append(recovered)
                unavailable.remove(failed)
    return downloaded, unavailable


def recover_unambiguous_staging(instances: list[dict[str, Any]], directory: Path) -> list[Path]:
    """Recover single-section files whose filename timestamps were rounded by yt-dlp."""
    staging = directory / ".sections"
    if not staging.exists():
        return []
    instances_by_source: dict[str, list[dict[str, Any]]] = {}
    for instance in instances:
        if find_video(directory, str(instance["video_id"])) is None:
            instances_by_source.setdefault(str(instance["source_id"]), []).append(instance)
    recovered: list[Path] = []
    for source_id, source_instances in instances_by_source.items():
        candidates = list(staging.glob(f"{source_id}_*"))
        if len(source_instances) != 1 or len(candidates) != 1:
            continue
        candidate = candidates[0]
        valid, _ = validate_video(candidate)
        if not valid:
            continue
        target = directory / f"{source_instances[0]['video_id']}{candidate.suffix}"
        candidate.replace(target)
        recovered.append(target)
    return recovered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=PROJECT_ROOT / "data/medical_subset.json")
    parser.add_argument("--videos", type=Path, default=PROJECT_ROOT / "data/videos")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "logs/dataset_report.json")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.videos.mkdir(parents=True, exist_ok=True)

    subset = _read_json(args.subset)
    instances = [item for entry in subset for item in entry.get("instances", [])]
    recovered = recover_unambiguous_staging(instances, args.videos)
    if recovered:
        logging.info("Recovered %d unambiguous staged clips", len(recovered))
    if args.report_only:
        report = _read_json(args.report, {})
        unavailable = [
            entry for entry in report.get("unavailable_videos", [])
            if find_video(args.videos, str(entry["video_id"])) is None
        ]
        available = sum(find_video(args.videos, str(item["video_id"])) is not None for item in instances)
        report.update({
            "videos_requested": len(instances),
            "videos_downloaded": available,
            "videos_unavailable": len(instances) - available,
            "unavailable_videos": unavailable,
        })
        _write_json(args.report, report)
        logging.info("Clip availability: %d/%d; unavailable: %d", available, len(instances), len(instances) - available)
        return
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    ffmpeg_directory = str(_ffmpeg_executable(args.videos).parent)
    if ffmpeg_directory not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join((ffmpeg_directory, os.environ.get("PATH", "")))

    unavailable: list[dict[str, str]] = []
    newly_downloaded = 0
    already_present = 0
    pending_by_source: dict[str, list[dict[str, Any]]] = {}
    for number, instance in enumerate(instances, start=1):
        video_id = str(instance["video_id"])
        existing = find_video(args.videos, video_id)
        if existing is not None:
            valid, reason = validate_video(existing)
            if valid:
                already_present += 1
                logging.info("[%d/%d] Already present: %s (%s)", number, len(instances), video_id, instance["gloss"])
                continue
            logging.warning("Removing invalid existing clip %s: %s", existing.name, reason)
            existing.unlink(missing_ok=True)
        pending_by_source.setdefault(str(instance["source_id"]), []).append(instance)

    total_sources = len(pending_by_source)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_source, source_instances, args.videos): (source_number, source_id, source_instances)
            for source_number, (source_id, source_instances) in enumerate(pending_by_source.items(), start=1)
        }
        for future in as_completed(futures):
            source_number, source_id, source_instances = futures[future]
            downloaded, failed = future.result()
            newly_downloaded += len(downloaded)
            unavailable.extend(failed)
            logging.info(
                "[source %d/%d] %s: %d downloaded, %d unavailable (%d intervals)",
                source_number,
                total_sources,
                source_id,
                len(downloaded),
                len(failed),
                len(source_instances),
            )

    available = sum(find_video(args.videos, str(item["video_id"])) is not None for item in instances)
    report = _read_json(args.report, {})
    report.update({
        "videos_requested": len(instances),
        "videos_downloaded": available,
        "videos_newly_downloaded": newly_downloaded,
        "videos_already_present": already_present,
        "videos_unavailable": len(unavailable),
        "unavailable_videos": unavailable,
    })
    _write_json(args.report, report)
    logging.info("Clip availability: %d/%d; unavailable: %d", available, len(instances), len(unavailable))


if __name__ == "__main__":
    main()

"""I/O pipeline for building a reviewable tennis-ball candidate dataset."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2

from tennis_vision.config import CameraConfig, TrackNetConfig
from tennis_vision.dataset.ball import (
    MISS_REVIEW,
    SPLIT_NAMES,
    TRACKNET_PRELABEL,
    CandidateFrame,
    DatasetRecord,
    SplitSegment,
    VideoMetadata,
    build_candidate_box,
    build_coco_document,
    build_split_segments,
    records_for_split,
    segment_for_frame,
    select_diverse_frames,
)
from tennis_vision.media import open_source
from tennis_vision.vision.tracknet import TrackNetBallDetector

VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4"}


@dataclass(frozen=True)
class DatasetBuildConfig:
    videos: Path
    output: Path
    target_frames: int = 300
    minimum_frame_gap: int = 8
    minimum_time_gap: float = 0.08
    miss_review_fraction: float = 0.10
    boundary_guard_seconds: float = 1.0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    spatial_columns: int = 4
    spatial_rows: int = 3
    box_scale: float = 1.5
    minimum_box_size: float = 12.0
    maximum_box_size: float = 48.0
    image_format: str = "jpg"
    jpeg_quality: int = 95
    seed: int = 7
    checkpoint_path: str = "models/tracknet/tracknet_v1_tennis.pth"
    device: str | None = None


def discover_videos(location: Path) -> tuple[Path, ...]:
    """Return supported source videos without modifying them."""
    location = location.expanduser().resolve()
    if location.is_file():
        if location.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {location.suffix}")
        return (location,)
    if not location.is_dir():
        raise ValueError(f"Video source does not exist: {location}")
    return tuple(
        sorted(
            path
            for path in location.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
    )


def probe_video(path: Path, *, source_video: str | None = None) -> VideoMetadata:
    """Read container metadata needed for splitting and provenance."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video file: {path}")
    try:
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if width < 1 or height < 1 or not math.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"Video has unusable dimensions or FPS metadata: {path}")
    if frame_count < 1:
        raise RuntimeError(f"Video has unusable frame-count metadata: {path}")
    duration_seconds, duration_source = _probe_duration(
        path, fallback=frame_count / fps
    )
    return VideoMetadata(
        source_video=source_video or path.name,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        duration_source=duration_source,
    )


def inspect_sources(
    videos_location: Path,
    *,
    boundary_guard_seconds: float = 1.0,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[tuple[Path, VideoMetadata, tuple[SplitSegment, ...]], ...]:
    """Probe sources and calculate split-first temporal plans."""
    paths = discover_videos(videos_location)
    if not paths:
        raise RuntimeError(f"No supported videos found in: {videos_location}")
    root = videos_location.resolve() if videos_location.is_dir() else paths[0].parent
    inspected = []
    for path in paths:
        source_video = path.relative_to(root).as_posix()
        metadata = probe_video(path, source_video=source_video)
        segments = build_split_segments(
            metadata,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            boundary_guard_seconds=boundary_guard_seconds,
        )
        inspected.append((path, metadata, segments))
    return tuple(inspected)


def scan_tracknet_candidates(
    inspected_sources: Sequence[
        tuple[Path, VideoMetadata, tuple[SplitSegment, ...]]
    ],
    detector: TrackNetBallDetector,
    config: DatasetBuildConfig,
) -> tuple[CandidateFrame, ...]:
    """Run TrackNet on every source frame and retain split-eligible observations."""
    candidates: list[CandidateFrame] = []
    for path, video, segments in inspected_sources:
        detector.reset()
        source = open_source(CameraConfig(), str(path))
        previous_detection: tuple[int, float, float, float] | None = None
        started = time.perf_counter()
        decoded_frames = 0
        try:
            while True:
                source_frame = source.read()
                if source_frame is None:
                    break
                decoded_frames += 1
                frame = source_frame.image
                metadata = source_frame.metadata
                detections = detector.detect(frame, metadata)
                detection = detections[0] if detections else None
                segment = segment_for_frame(segments, metadata.frame_index)

                motion_proxy = None
                if detection is not None and previous_detection is not None:
                    previous_frame, previous_time, previous_x, previous_y = (
                        previous_detection
                    )
                    elapsed = metadata.timestamp_seconds - previous_time
                    if metadata.frame_index == previous_frame + 1 and elapsed > 0.0:
                        diagonal = math.hypot(frame.shape[1], frame.shape[0])
                        motion_proxy = (
                            math.dist(
                                detection.center, (previous_x, previous_y)
                            )
                            / diagonal
                            / elapsed
                        )
                previous_detection = (
                    None
                    if detection is None
                    else (
                        metadata.frame_index,
                        metadata.timestamp_seconds,
                        detection.center_x,
                        detection.center_y,
                    )
                )

                # The first frames only warm the temporal model and are not
                # meaningful miss/negative candidates.
                if (
                    segment is None
                    or metadata.frame_index < detector.config.temporal_frames - 1
                ):
                    continue
                if detection is None:
                    candidates.append(
                        CandidateFrame(
                            source_video=video.source_video,
                            frame_index=metadata.frame_index,
                            timestamp_seconds=metadata.timestamp_seconds,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            split=segment.split,
                            group=segment.group,
                            candidate_type=MISS_REVIEW,
                            tracknet_x=None,
                            tracknet_y=None,
                            tracknet_confidence=None,
                            tracknet_box=None,
                            candidate_box=None,
                        )
                    )
                else:
                    candidates.append(
                        CandidateFrame(
                            source_video=video.source_video,
                            frame_index=metadata.frame_index,
                            timestamp_seconds=metadata.timestamp_seconds,
                            width=frame.shape[1],
                            height=frame.shape[0],
                            split=segment.split,
                            group=segment.group,
                            candidate_type=TRACKNET_PRELABEL,
                            tracknet_x=detection.center_x,
                            tracknet_y=detection.center_y,
                            tracknet_confidence=detection.confidence,
                            tracknet_box=detection.bounding_box,
                            candidate_box=build_candidate_box(
                                detection,
                                image_width=frame.shape[1],
                                image_height=frame.shape[0],
                                scale=config.box_scale,
                                minimum_size=config.minimum_box_size,
                                maximum_size=config.maximum_box_size,
                            ),
                            motion_proxy_per_second=motion_proxy,
                        )
                    )
                if decoded_frames % 250 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"  {video.source_video}: {decoded_frames}/"
                        f"{video.frame_count} frames ({elapsed:.1f}s)"
                    )
        finally:
            source.release()
        print(
            f"  {video.source_video}: scanned {decoded_frames} frames in "
            f"{time.perf_counter() - started:.1f}s"
        )
    return tuple(candidates)


def build_dataset(config: DatasetBuildConfig) -> dict[str, object]:
    """Build a new candidate dataset and return its measured summary."""
    _validate_build_config(config)
    if config.output.exists():
        raise RuntimeError(
            f"Output already exists; choose a new dataset directory: {config.output}"
        )
    inspected = inspect_sources(
        config.videos,
        boundary_guard_seconds=config.boundary_guard_seconds,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    detector = TrackNetBallDetector(
        TrackNetConfig(
            checkpoint_path=config.checkpoint_path,
            device=config.device,
        )
    )
    candidates = scan_tracknet_candidates(inspected, detector, config)
    selected = select_diverse_frames(
        candidates,
        target_frames=config.target_frames,
        minimum_frame_gap=config.minimum_frame_gap,
        minimum_time_gap=config.minimum_time_gap,
        miss_review_fraction=config.miss_review_fraction,
        split_fractions={
            "train": config.train_fraction,
            "validation": config.validation_fraction,
            "test": config.test_fraction,
        },
        spatial_columns=config.spatial_columns,
        spatial_rows=config.spatial_rows,
        seed=config.seed,
    )
    _create_workspace(config.output)
    records = _extract_selected_images(inspected, selected, config)
    _write_dataset_files(config, inspected, records)
    summary = summarize_records(records, source_video_count=len(inspected))
    _write_json(config.output / "metadata" / "summary.json", summary)
    return summary


def describe_inspection(
    inspected: Sequence[tuple[Path, VideoMetadata, tuple[SplitSegment, ...]]]
) -> str:
    """Return a human-readable dry-run report."""
    lines = [f"Source videos: {len(inspected)}"]
    for _, video, segments in inspected:
        lines.append(
            f"- {video.source_video}: {video.width}x{video.height}, "
            f"{video.fps:.3f} FPS, {video.frame_count} frames, "
            f"{video.duration_seconds:.3f}s"
        )
        for segment in segments:
            lines.append(
                f"  {segment.split}: frames {segment.start_frame}.."
                f"{segment.end_frame - 1} ({segment.end_frame - segment.start_frame})"
            )
    return "\n".join(lines)


def summarize_records(
    records: Sequence[DatasetRecord], *, source_video_count: int
) -> dict[str, object]:
    split_counts = Counter(record.split for record in records)
    prelabels = sum(record.candidate_box is not None for record in records)
    miss_reviews = len(records) - prelabels
    return {
        "source_videos": source_video_count,
        "candidate_images": len(records),
        "tracknet_prelabels": prelabels,
        "needs_manual_ball_search_or_negative_review": miss_reviews,
        "confirmed_negatives": 0,
        "splits": {name: split_counts.get(name, 0) for name in SPLIT_NAMES},
        "ground_truth_ready": False,
        "human_review_required": True,
    }


def _extract_selected_images(
    inspected: Sequence[tuple[Path, VideoMetadata, tuple[SplitSegment, ...]]],
    selected: Sequence[CandidateFrame],
    config: DatasetBuildConfig,
) -> tuple[DatasetRecord, ...]:
    selected_by_video: dict[str, dict[int, CandidateFrame]] = defaultdict(dict)
    for candidate in selected:
        selected_by_video[candidate.source_video][candidate.frame_index] = candidate

    image_id_by_key = {
        (candidate.source_video, candidate.frame_index): image_id
        for image_id, candidate in enumerate(selected, start=1)
    }
    records: list[DatasetRecord] = []
    extension = ".jpg" if config.image_format == "jpg" else ".png"
    for path, video, _ in inspected:
        wanted = selected_by_video.get(video.source_video, {})
        if not wanted:
            continue
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not reopen video for extraction: {path}")
        try:
            frame_index = 0
            remaining = set(wanted)
            while remaining:
                success, frame = capture.read()
                if not success:
                    break
                candidate = wanted.get(frame_index)
                if candidate is not None:
                    source_hash = hashlib.sha1(
                        video.source_video.encode("utf-8")
                    ).hexdigest()[:8]
                    stem = _safe_filename(Path(video.source_video).stem)
                    filename = (
                        f"{stem}_{source_hash}_f{frame_index:08d}{extension}"
                    )
                    image_path = config.output / "candidates" / "images" / filename
                    preview_path = (
                        config.output / "candidates" / "previews" / filename
                    )
                    _write_image(image_path, frame, config)
                    preview = _make_preview(frame, candidate)
                    _write_image(preview_path, preview, config)
                    records.append(
                        DatasetRecord(
                            image_id=image_id_by_key[
                                (candidate.source_video, candidate.frame_index)
                            ],
                            image_filename=filename,
                            preview_filename=filename,
                            source_video=candidate.source_video,
                            source_frame_index=candidate.frame_index,
                            source_timestamp_seconds=candidate.timestamp_seconds,
                            original_width=frame.shape[1],
                            original_height=frame.shape[0],
                            tracknet_x=candidate.tracknet_x,
                            tracknet_y=candidate.tracknet_y,
                            tracknet_confidence=candidate.tracknet_confidence,
                            tracknet_box=candidate.tracknet_box,
                            candidate_box=candidate.candidate_box,
                            candidate_type=candidate.candidate_type,
                            split=candidate.split,
                            group=candidate.group,
                        )
                    )
                    remaining.remove(frame_index)
                frame_index += 1
        finally:
            capture.release()
        if remaining:
            missing = sorted(remaining)
            raise RuntimeError(
                f"Could not decode selected frames from {video.source_video}: {missing}"
            )
    return tuple(sorted(records, key=lambda record: record.image_id))


def _write_dataset_files(
    config: DatasetBuildConfig,
    inspected: Sequence[tuple[Path, VideoMetadata, tuple[SplitSegment, ...]]],
    records: Sequence[DatasetRecord],
) -> None:
    metadata_directory = config.output / "metadata"
    labels_directory = config.output / "candidates" / "labels"
    _write_json(
        metadata_directory / "source_videos.json",
        [
            {
                **asdict(video),
                "segments": [asdict(segment) for segment in segments],
            }
            for _, video, segments in inspected
        ],
    )
    _write_json(metadata_directory / "build_config.json", _jsonable_config(config))
    manifest = metadata_directory / "candidates.jsonl"
    with manifest.open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")

    dataset_name = config.output.name
    _write_json(
        labels_directory / "instances_candidates.json",
        build_coco_document(records, dataset_name=dataset_name),
    )
    for split in SPLIT_NAMES:
        split_records = records_for_split(records, split)
        _write_json(
            labels_directory / f"instances_{split}.json",
            build_coco_document(
                split_records, dataset_name=f"{dataset_name} {split}"
            ),
        )
        split_file = config.output / "splits" / f"{split}.txt"
        split_file.write_text(
            "".join(f"{record.image_filename}\n" for record in split_records),
            encoding="utf-8",
        )
    (config.output / "REVIEW.md").write_text(_review_instructions(), encoding="utf-8")


def _create_workspace(output: Path) -> None:
    for relative in (
        "raw",
        "candidates/images",
        "candidates/labels",
        "candidates/previews",
        "reviewed/images",
        "reviewed/labels",
        "metadata",
        "splits",
    ):
        (output / relative).mkdir(parents=True, exist_ok=False)


def _make_preview(frame: object, candidate: CandidateFrame) -> object:
    preview = frame.copy()
    if candidate.candidate_box is not None:
        x1, y1, x2, y2 = (round(value) for value in candidate.candidate_box)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        assert candidate.tracknet_x is not None and candidate.tracknet_y is not None
        center = (round(candidate.tracknet_x), round(candidate.tracknet_y))
        cv2.drawMarker(
            preview,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=14,
            thickness=2,
        )
        confidence = (
            "n/a"
            if candidate.tracknet_confidence is None
            else f"{candidate.tracknet_confidence:.3f}"
        )
        label = f"TrackNet prelabel | confidence {confidence}"
    else:
        label = "TrackNet miss | review for ball or true negative"
    cv2.putText(
        preview,
        f"frame {candidate.frame_index} | {label}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if candidate.candidate_box is None else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return preview


def _write_image(path: Path, frame: object, config: DatasetBuildConfig) -> None:
    options = (
        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        if config.image_format == "jpg"
        else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    )
    if not cv2.imwrite(str(path), frame, options):
        raise RuntimeError(f"Failed to write extracted image: {path}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable_config(config: DatasetBuildConfig) -> dict[str, object]:
    values = asdict(config)
    values["videos"] = str(config.videos)
    values["output"] = str(config.output)
    return values


def _validate_build_config(config: DatasetBuildConfig) -> None:
    if config.target_frames < 1:
        raise ValueError("target_frames must be positive")
    if config.minimum_frame_gap < 0 or config.minimum_time_gap < 0.0:
        raise ValueError("sampling gaps cannot be negative")
    if not 0.0 <= config.miss_review_fraction <= 1.0:
        raise ValueError("miss_review_fraction must be in [0, 1]")
    if config.boundary_guard_seconds < 0.0:
        raise ValueError("boundary_guard_seconds cannot be negative")
    if config.spatial_columns < 1 or config.spatial_rows < 1:
        raise ValueError("spatial grid dimensions must be positive")
    if (
        config.box_scale <= 0.0
        or config.minimum_box_size <= 0.0
        or config.maximum_box_size < config.minimum_box_size
    ):
        raise ValueError("invalid candidate box sizing configuration")
    if config.image_format not in {"jpg", "png"}:
        raise ValueError("image_format must be jpg or png")
    if not 1 <= config.jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if config.output.resolve() in {Path("/").resolve(), Path.home().resolve()}:
        raise ValueError("refusing to use a broad filesystem location as output")


def _safe_filename(value: str) -> str:
    cleaned = "".join(character if character.isalnum() else "_" for character in value)
    return cleaned.strip("_") or "video"


def _review_instructions() -> str:
    return """# Human review required

TrackNet boxes in this dataset are preliminary suggestions, not ground truth.

1. Inspect `candidates/previews/` for a fast visual pass.
2. Import `candidates/images/` and
   `candidates/labels/instances_candidates.json` into a COCO-compatible
   annotation tool such as CVAT.
3. For every image, approve or correct the `tennis_ball` box, reject false
   TrackNet labels, find missed visible balls, and confirm genuinely ball-free
   images as negatives.
4. Export the reviewed COCO annotations to
   `reviewed/labels/instances_reviewed.json`. Keep only reviewed full-resolution
   images in `reviewed/images/` or preserve a documented reference to the
   candidate images.
5. Update review statuses in `metadata/candidates.jsonl` when consolidating the
   reviewed export.

Do not train a detector from `instances_candidates.json`; it contains unreviewed
TrackNet pre-annotations and unconfirmed negative candidates.
"""


def _probe_duration(path: Path, *, fallback: float) -> tuple[float, str]:
    """Use container duration when ffprobe exists, otherwise record an estimate."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        try:
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            duration = float(result.stdout.strip())
            if math.isfinite(duration) and duration > 0.0:
                return duration, "ffprobe_container"
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return fallback, "frame_count_divided_by_fps"

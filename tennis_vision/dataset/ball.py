"""Pure dataset planning, sampling, and annotation conversion logic."""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from tennis_vision.vision.ball import BallDetection

SPLIT_NAMES = ("train", "validation", "test")
TRACKNET_PRELABEL = "TRACKNET_PRELABEL"
MISS_REVIEW = "TRACKNET_MISS_REVIEW"
UNREVIEWED = "UNREVIEWED"


@dataclass(frozen=True)
class VideoMetadata:
    """Properties reported by the source video container."""

    source_video: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    duration_source: str = "frame_count_divided_by_fps"


@dataclass(frozen=True)
class SplitSegment:
    """One contiguous, half-open temporal segment assigned to a split."""

    split: str
    group: str
    start_frame: int
    end_frame: int

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index < self.end_frame


@dataclass(frozen=True)
class CandidateFrame:
    """A source frame considered for extraction after TrackNet inference."""

    source_video: str
    frame_index: int
    timestamp_seconds: float
    width: int
    height: int
    split: str
    group: str
    candidate_type: str
    tracknet_x: float | None
    tracknet_y: float | None
    tracknet_confidence: float | None
    tracknet_box: tuple[float, float, float, float] | None
    candidate_box: tuple[float, float, float, float] | None
    motion_proxy_per_second: float | None = None

    @property
    def has_prelabel(self) -> bool:
        return self.candidate_type == TRACKNET_PRELABEL


@dataclass(frozen=True)
class DatasetRecord:
    """Extracted image metadata stored in the dataset manifest."""

    image_id: int
    image_filename: str
    preview_filename: str
    source_video: str
    source_frame_index: int
    source_timestamp_seconds: float
    original_width: int
    original_height: int
    tracknet_x: float | None
    tracknet_y: float | None
    tracknet_confidence: float | None
    tracknet_box: tuple[float, float, float, float] | None
    candidate_box: tuple[float, float, float, float] | None
    candidate_type: str
    split: str
    group: str
    review_status: str = UNREVIEWED

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_split_segments(
    video: VideoMetadata,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    boundary_guard_seconds: float = 1.0,
) -> tuple[SplitSegment, ...]:
    """Split a video contiguously, excluding a guard band at each boundary.

    Assignment happens before frame sampling. ``boundary_guard_seconds`` is the
    total excluded time centered on each split boundary, not the amount removed
    from each side.
    """
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0.0 for value in fractions):
        raise ValueError("split fractions cannot be negative")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    if boundary_guard_seconds < 0.0:
        raise ValueError("boundary_guard_seconds cannot be negative")
    if video.frame_count < 1:
        return ()

    train_boundary = round(video.frame_count * train_fraction)
    validation_boundary = round(
        video.frame_count * (train_fraction + validation_fraction)
    )
    half_guard = math.ceil(video.fps * boundary_guard_seconds / 2.0)
    ranges = (
        ("train", 0, max(0, train_boundary - half_guard)),
        (
            "validation",
            min(video.frame_count, train_boundary + half_guard),
            max(0, validation_boundary - half_guard),
        ),
        (
            "test",
            min(video.frame_count, validation_boundary + half_guard),
            video.frame_count,
        ),
    )
    segments: list[SplitSegment] = []
    video_key = _safe_identifier(video.source_video)
    for split, start, end in ranges:
        start = min(max(start, 0), video.frame_count)
        end = min(max(end, start), video.frame_count)
        if start < end:
            segments.append(
                SplitSegment(
                    split=split,
                    group=f"{video_key}_{split}_segment_000",
                    start_frame=start,
                    end_frame=end,
                )
            )
    return tuple(segments)


def segment_for_frame(
    segments: Sequence[SplitSegment], frame_index: int
) -> SplitSegment | None:
    return next(
        (segment for segment in segments if segment.contains(frame_index)), None
    )


def build_candidate_box(
    detection: BallDetection,
    *,
    image_width: int,
    image_height: int,
    scale: float = 1.5,
    minimum_size: float = 12.0,
    maximum_size: float = 48.0,
) -> tuple[float, float, float, float]:
    """Create a clipped square prelabel from TrackNet's Hough-circle support.

    The TrackNet box is not an object boundary. Its largest diameter is scaled
    and clamped to conservative configurable limits for human correction.
    """
    if image_width < 1 or image_height < 1:
        raise ValueError("image dimensions must be positive")
    if scale <= 0.0 or minimum_size <= 0.0 or maximum_size < minimum_size:
        raise ValueError("invalid candidate box sizing configuration")
    x1, y1, x2, y2 = detection.bounding_box
    support_diameter = max(x2 - x1, y2 - y1)
    size = min(max(support_diameter * scale, minimum_size), maximum_size)
    half = size / 2.0
    return (
        max(0.0, detection.center_x - half),
        max(0.0, detection.center_y - half),
        min(float(image_width), detection.center_x + half),
        min(float(image_height), detection.center_y + half),
    )


def select_diverse_frames(
    candidates: Sequence[CandidateFrame],
    *,
    target_frames: int,
    minimum_frame_gap: int,
    minimum_time_gap: float,
    miss_review_fraction: float = 0.10,
    split_fractions: Mapping[str, float] | None = None,
    spatial_columns: int = 4,
    spatial_rows: int = 3,
    seed: int = 7,
) -> tuple[CandidateFrame, ...]:
    """Select temporally spaced frames using source, position, and motion bins."""
    if target_frames < 0 or minimum_frame_gap < 0 or minimum_time_gap < 0.0:
        raise ValueError("sampling targets and gaps cannot be negative")
    if not 0.0 <= miss_review_fraction <= 1.0:
        raise ValueError("miss_review_fraction must be in [0, 1]")
    if spatial_columns < 1 or spatial_rows < 1:
        raise ValueError("spatial grid dimensions must be positive")
    fractions = split_fractions or {
        "train": 0.70,
        "validation": 0.15,
        "test": 0.15,
    }
    quotas = _allocate_counts(target_frames, fractions)
    selected: list[CandidateFrame] = []
    for split_index, split in enumerate(SPLIT_NAMES):
        quota = quotas.get(split, 0)
        pool = [candidate for candidate in candidates if candidate.split == split]
        desired_misses = round(quota * miss_review_fraction)
        misses = [item for item in pool if not item.has_prelabel]
        hits = [item for item in pool if item.has_prelabel]

        split_selected: list[CandidateFrame] = []
        split_selected.extend(
            _round_robin_select(
                misses,
                limit=desired_misses,
                already_selected=split_selected,
                minimum_frame_gap=minimum_frame_gap,
                minimum_time_gap=minimum_time_gap,
                spatial_columns=spatial_columns,
                spatial_rows=spatial_rows,
                seed=seed + split_index * 101,
            )
        )
        split_selected.extend(
            _round_robin_select(
                hits,
                limit=max(0, quota - len(split_selected)),
                already_selected=split_selected,
                minimum_frame_gap=minimum_frame_gap,
                minimum_time_gap=minimum_time_gap,
                spatial_columns=spatial_columns,
                spatial_rows=spatial_rows,
                seed=seed + split_index * 101 + 1,
            )
        )
        if len(split_selected) < quota:
            chosen_keys = {_candidate_key(item) for item in split_selected}
            remaining = [
                item for item in pool if _candidate_key(item) not in chosen_keys
            ]
            split_selected.extend(
                _round_robin_select(
                    remaining,
                    limit=quota - len(split_selected),
                    already_selected=split_selected,
                    minimum_frame_gap=minimum_frame_gap,
                    minimum_time_gap=minimum_time_gap,
                    spatial_columns=spatial_columns,
                    spatial_rows=spatial_rows,
                    seed=seed + split_index * 101 + 2,
                )
            )
        selected.extend(split_selected)

    return tuple(
        sorted(
            selected,
            key=lambda item: (item.source_video, item.frame_index),
        )
    )


def build_coco_document(
    records: Sequence[DatasetRecord], *, dataset_name: str
) -> dict[str, object]:
    """Build a valid one-class COCO detection document from prelabels."""
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    annotation_id = 1
    for record in records:
        images.append(
            {
                "id": record.image_id,
                "file_name": record.image_filename,
                "width": record.original_width,
                "height": record.original_height,
                "split": record.split,
                "source_video": record.source_video,
                "source_frame_index": record.source_frame_index,
            }
        )
        if record.candidate_box is None:
            continue
        x1, y1, x2, y2 = record.candidate_box
        width = x2 - x1
        height = y2 - y1
        annotations.append(
            {
                "id": annotation_id,
                "image_id": record.image_id,
                "category_id": 1,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "pre_annotation": True,
                "review_status": record.review_status,
            }
        )
        annotation_id += 1
    return {
        "info": {
            "description": dataset_name,
            "annotation_status": (
                "TrackNet pre-annotations; human review required before training"
            ),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "tennis_ball", "supercategory": "ball"}],
    }


def records_for_split(
    records: Iterable[DatasetRecord], split: str
) -> tuple[DatasetRecord, ...]:
    return tuple(record for record in records if record.split == split)


def _round_robin_select(
    candidates: Sequence[CandidateFrame],
    *,
    limit: int,
    already_selected: Sequence[CandidateFrame],
    minimum_frame_gap: int,
    minimum_time_gap: float,
    spatial_columns: int,
    spatial_rows: int,
    seed: int,
) -> list[CandidateFrame]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    buckets: dict[tuple[object, ...], list[CandidateFrame]] = defaultdict(list)
    for candidate in candidates:
        buckets[
            _diversity_bucket(candidate, spatial_columns, spatial_rows)
        ].append(candidate)
    bucket_keys = sorted(buckets, key=repr)
    rng.shuffle(bucket_keys)
    queues: dict[tuple[object, ...], deque[CandidateFrame]] = {}
    for key in bucket_keys:
        values = sorted(buckets[key], key=lambda item: item.frame_index)
        rng.shuffle(values)
        queues[key] = deque(values)

    chosen = list(already_selected)
    added: list[CandidateFrame] = []
    while len(added) < limit:
        made_progress = False
        for key in bucket_keys:
            queue = queues[key]
            while queue:
                candidate = queue.popleft()
                if _is_temporally_spaced(
                    candidate,
                    chosen,
                    minimum_frame_gap=minimum_frame_gap,
                    minimum_time_gap=minimum_time_gap,
                ):
                    chosen.append(candidate)
                    added.append(candidate)
                    made_progress = True
                    break
            if len(added) >= limit:
                break
        if not made_progress:
            break
    return added


def _diversity_bucket(
    candidate: CandidateFrame, columns: int, rows: int
) -> tuple[object, ...]:
    if candidate.tracknet_x is None or candidate.tracknet_y is None:
        spatial = ("miss",)
    else:
        column = min(columns - 1, int(candidate.tracknet_x / candidate.width * columns))
        row = min(rows - 1, int(candidate.tracknet_y / candidate.height * rows))
        spatial = (column, row)
    return (
        candidate.source_video,
        *spatial,
        _motion_bucket(candidate.motion_proxy_per_second),
        _confidence_bucket(candidate.tracknet_confidence),
    )


def _motion_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.15:
        return "low"
    if value < 0.75:
        return "medium"
    return "high"


def _confidence_bucket(value: float | None) -> str:
    if value is None:
        return "unavailable"
    if value < 0.33:
        return "low"
    if value < 0.67:
        return "medium"
    return "high"


def _is_temporally_spaced(
    candidate: CandidateFrame,
    selected: Sequence[CandidateFrame],
    *,
    minimum_frame_gap: int,
    minimum_time_gap: float,
) -> bool:
    for existing in selected:
        if existing.source_video != candidate.source_video:
            continue
        if abs(existing.frame_index - candidate.frame_index) < minimum_frame_gap:
            return False
        if (
            abs(existing.timestamp_seconds - candidate.timestamp_seconds)
            < minimum_time_gap
        ):
            return False
    return True


def _allocate_counts(
    target: int, fractions: Mapping[str, float]
) -> dict[str, int]:
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("split fractions cannot be negative")
    total = sum(fractions.get(name, 0.0) for name in SPLIT_NAMES)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")
    raw = {name: target * fractions.get(name, 0.0) for name in SPLIT_NAMES}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = target - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _candidate_key(candidate: CandidateFrame) -> tuple[str, int]:
    return candidate.source_video, candidate.frame_index


def _safe_identifier(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)

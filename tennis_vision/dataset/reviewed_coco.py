"""Validation and temporal splitting for a human-reviewed COCO export."""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

TENNIS_BALL_CLASS = "tennis ball"
FRAME_PATTERN = re.compile(r"(?:^|_)f(?P<frame>\d{8})(?:_|\.|$)")


@dataclass(frozen=True)
class TemporalRange:
    """One half-open source-frame range assigned to an RF-DETR split."""

    name: str
    start_frame: int
    end_frame: int

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index < self.end_frame


DEFAULT_TEMPORAL_RANGES = (
    TemporalRange("train", 0, 2247),
    TemporalRange("valid", 2367, 2742),
    TemporalRange("test", 2862, 3296),
)


@dataclass(frozen=True)
class ReviewedDatasetReport:
    """Auditable validation and size summary for a reviewed export."""

    source: str
    annotation_file: str
    image_count: int
    annotation_count: int
    annotated_image_count: int
    zero_annotation_image_count: int
    zero_annotation_filenames: tuple[str, ...]
    multi_annotation_image_count: int
    multi_annotation_filenames: tuple[str, ...]
    category_names: tuple[str, ...]
    annotated_category_names: tuple[str, ...]
    image_widths: tuple[int, ...]
    image_heights: tuple[int, ...]
    frame_min: int | None
    frame_max: int | None
    split_counts: Mapping[str, int]
    box_width_pixels: Mapping[str, float]
    box_height_pixels: Mapping[str, float]
    box_area_pixels: Mapping[str, float]
    box_area_percent: Mapping[str, float]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_source_frame(filename: str) -> int:
    """Extract the zero-based source frame encoded by the dataset generator."""
    match = FRAME_PATTERN.search(Path(filename).name)
    if match is None:
        raise ValueError(f"Filename has no source-frame token: {filename}")
    return int(match.group("frame"))


def locate_coco_annotation(source: Path) -> Path:
    """Find the single Roboflow COCO annotation file in an export directory."""
    source = source.expanduser().resolve()
    candidates = sorted(source.glob("*/_annotations.coco.json"))
    if (source / "_annotations.coco.json").is_file():
        candidates.insert(0, source / "_annotations.coco.json")
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one _annotations.coco.json under {source}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def load_reviewed_coco(source: Path) -> tuple[Path, dict[str, Any]]:
    annotation_path = locate_coco_annotation(source)
    try:
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read COCO annotations: {annotation_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("COCO annotation root must be an object")
    return annotation_path, document


def split_for_frame(
    frame_index: int,
    ranges: Sequence[TemporalRange] = DEFAULT_TEMPORAL_RANGES,
) -> str | None:
    matches = [item.name for item in ranges if item.contains(frame_index)]
    if len(matches) > 1:
        raise ValueError(f"Overlapping temporal ranges for frame {frame_index}")
    return matches[0] if matches else None


def inspect_reviewed_dataset(
    source: Path,
    *,
    ranges: Sequence[TemporalRange] = DEFAULT_TEMPORAL_RANGES,
    verify_pixels: bool = True,
) -> ReviewedDatasetReport:
    """Validate class, boxes, files, dimensions, and temporal provenance."""
    annotation_path, document = load_reviewed_coco(source)
    image_root = annotation_path.parent
    errors: list[str] = []
    warnings: list[str] = []

    images = document.get("images", [])
    annotations = document.get("annotations", [])
    categories = document.get("categories", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO images and annotations must be arrays")
    if not isinstance(categories, list):
        raise ValueError("COCO categories must be an array")

    category_names_by_id: dict[int, str] = {}
    for category in categories:
        try:
            category_names_by_id[int(category["id"])] = str(category["name"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"Invalid category record: {category!r}")
    if TENNIS_BALL_CLASS not in category_names_by_id.values():
        errors.append(f'Required category name "{TENNIS_BALL_CLASS}" is missing')

    images_by_id: dict[int, dict[str, Any]] = {}
    frame_by_image_id: dict[int, int] = {}
    filenames: Counter[str] = Counter()
    frame_counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    widths: set[int] = set()
    heights: set[int] = set()
    for image in images:
        try:
            image_id = int(image["id"])
            filename = str(image["file_name"])
            width = int(image["width"])
            height = int(image["height"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"Invalid image record: {image!r}")
            continue
        if image_id in images_by_id:
            errors.append(f"Duplicate image ID: {image_id}")
            continue
        images_by_id[image_id] = image
        filenames[filename] += 1
        widths.add(width)
        heights.add(height)
        if width < 1 or height < 1:
            errors.append(f"Image {filename} has non-positive dimensions")
        try:
            frame_index = parse_source_frame(filename)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        frame_by_image_id[image_id] = frame_index
        frame_counts[frame_index] += 1
        split = split_for_frame(frame_index, ranges)
        if split is None:
            errors.append(
                f"Reviewed image falls in an excluded guard/out-of-range frame: {filename}"
            )
        else:
            split_counts[split] += 1

        image_path = image_root / filename
        if not image_path.is_file():
            errors.append(f"Missing image file: {image_path}")
        elif verify_pixels:
            pixels = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if pixels is None:
                errors.append(f"Unreadable image file: {image_path}")
            elif pixels.shape[1] != width or pixels.shape[0] != height:
                errors.append(
                    f"Image dimension mismatch for {filename}: COCO={width}x{height}, "
                    f"file={pixels.shape[1]}x{pixels.shape[0]}"
                )

    annotation_counts: Counter[int] = Counter()
    annotated_category_names: set[str] = set()
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []
    box_area_percentages: list[float] = []
    annotation_ids: set[int] = set()
    for annotation in annotations:
        try:
            annotation_id = int(annotation["id"])
            image_id = int(annotation["image_id"])
            category_id = int(annotation["category_id"])
            x, y, width, height = (float(value) for value in annotation["bbox"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"Invalid annotation record: {annotation!r}")
            continue
        if annotation_id in annotation_ids:
            errors.append(f"Duplicate annotation ID: {annotation_id}")
        annotation_ids.add(annotation_id)
        image = images_by_id.get(image_id)
        if image is None:
            errors.append(f"Annotation {annotation_id} references missing image {image_id}")
            continue
        category_name = category_names_by_id.get(category_id)
        if category_name is None:
            errors.append(
                f"Annotation {annotation_id} references missing category {category_id}"
            )
        else:
            annotated_category_names.add(category_name)
            if category_name != TENNIS_BALL_CLASS:
                errors.append(
                    f"Annotation {annotation_id} uses class {category_name!r}; "
                    f"expected {TENNIS_BALL_CLASS!r}"
                )
        image_width = int(image["width"])
        image_height = int(image["height"])
        if not np.all(np.isfinite((x, y, width, height))):
            errors.append(f"Annotation {annotation_id} has non-finite box coordinates")
            continue
        if width <= 0.0 or height <= 0.0:
            errors.append(f"Annotation {annotation_id} has a non-positive box")
            continue
        if x < 0.0 or y < 0.0 or x + width > image_width or y + height > image_height:
            errors.append(f"Annotation {annotation_id} box is outside its image")
        area = width * height
        reported_area = annotation.get("area")
        if reported_area is not None:
            try:
                area_matches = np.isclose(float(reported_area), area)
            except (TypeError, ValueError):
                area_matches = False
            if not area_matches:
                warnings.append(
                    f"Annotation {annotation_id} area differs from bbox-derived area"
                )
        annotation_counts[image_id] += 1
        box_widths.append(width)
        box_heights.append(height)
        box_areas.append(area)
        box_area_percentages.append(100.0 * area / (image_width * image_height))

    zero_annotation_filenames = tuple(
        str(image["file_name"])
        for image_id, image in images_by_id.items()
        if annotation_counts[image_id] == 0
    )
    multi_annotation_filenames = tuple(
        str(image["file_name"])
        for image_id, image in images_by_id.items()
        if annotation_counts[image_id] > 1
    )
    annotated_images = sum(count > 0 for count in annotation_counts.values())
    zero_annotation_images = len(zero_annotation_filenames)
    multi_annotation_images = len(multi_annotation_filenames)
    if zero_annotation_images:
        warnings.append(
            f"{zero_annotation_images} reviewed image(s) contain no tennis-ball annotation"
        )
    if multi_annotation_images:
        warnings.append(
            f"{multi_annotation_images} image(s) contain multiple tennis-ball annotations"
        )
    duplicate_filenames = sorted(name for name, count in filenames.items() if count > 1)
    if duplicate_filenames:
        errors.append(f"Duplicate image filenames: {duplicate_filenames}")
    duplicate_frames = sorted(frame for frame, count in frame_counts.items() if count > 1)
    if duplicate_frames:
        warnings.append(f"Multiple images share source frame(s): {duplicate_frames}")

    frames = sorted(frame_by_image_id.values())
    return ReviewedDatasetReport(
        source=str(Path(source).expanduser().resolve()),
        annotation_file=str(annotation_path),
        image_count=len(images_by_id),
        annotation_count=len(annotations),
        annotated_image_count=annotated_images,
        zero_annotation_image_count=zero_annotation_images,
        zero_annotation_filenames=zero_annotation_filenames,
        multi_annotation_image_count=multi_annotation_images,
        multi_annotation_filenames=multi_annotation_filenames,
        category_names=tuple(category_names_by_id.values()),
        annotated_category_names=tuple(sorted(annotated_category_names)),
        image_widths=tuple(sorted(widths)),
        image_heights=tuple(sorted(heights)),
        frame_min=frames[0] if frames else None,
        frame_max=frames[-1] if frames else None,
        split_counts=dict(split_counts),
        box_width_pixels=_distribution(box_widths),
        box_height_pixels=_distribution(box_heights),
        box_area_pixels=_distribution(box_areas),
        box_area_percent=_distribution(box_area_percentages),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def prepare_temporal_dataset(
    source: Path,
    output: Path,
    *,
    ranges: Sequence[TemporalRange] = DEFAULT_TEMPORAL_RANGES,
) -> ReviewedDatasetReport:
    """Copy a valid reviewed export into RF-DETR's guarded split layout."""
    report = inspect_reviewed_dataset(source, ranges=ranges)
    if not report.is_valid:
        raise ValueError("Reviewed dataset validation failed:\n- " + "\n- ".join(report.errors))
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    annotation_path, document = load_reviewed_coco(source)
    image_root = annotation_path.parent
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output.mkdir(parents=True)
    for split in (item.name for item in ranges):
        split_dir = output / split
        split_dir.mkdir()
        split_images: list[dict[str, Any]] = []
        split_annotations: list[dict[str, Any]] = []
        for image in document["images"]:
            frame_index = parse_source_frame(str(image["file_name"]))
            if split_for_frame(frame_index, ranges) != split:
                continue
            copied_image = dict(image)
            copied_image["source_frame_index"] = frame_index
            split_images.append(copied_image)
            split_annotations.extend(annotations_by_image[int(image["id"])])
            shutil.copy2(image_root / str(image["file_name"]), split_dir / str(image["file_name"]))

        split_document = dict(document)
        split_document["images"] = split_images
        split_document["annotations"] = split_annotations
        info = dict(split_document.get("info", {}))
        info.update(
            {
                "annotation_status": "human-reviewed authoritative ground truth",
                "temporal_split": split,
                "split_method": "contiguous source-frame ranges with guard bands",
            }
        )
        split_document["info"] = info
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps(split_document, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        "source": str(Path(source).expanduser().resolve()),
        "class_name": TENNIS_BALL_CLASS,
        "ranges": [asdict(item) for item in ranges],
        "report": report.to_dict(),
        "limitation": (
            "This dataset originates from one recording and is not sufficient "
            "evidence of cross-session generalization."
        ),
    }
    (output / "dataset_report.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }

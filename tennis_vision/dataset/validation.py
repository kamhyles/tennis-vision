"""Sanity checks for generated tennis-ball candidate datasets."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2

from tennis_vision.dataset.ball import MISS_REVIEW, TRACKNET_PRELABEL


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        return f"errors={len(self.errors)}, warnings={len(self.warnings)}"


def validate_dataset(dataset: Path) -> ValidationReport:
    """Check manifest, COCO data, images, boxes, duplicates, and leakage clues."""
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = dataset / "metadata" / "candidates.jsonl"
    coco_path = dataset / "candidates" / "labels" / "instances_candidates.json"
    config_path = dataset / "metadata" / "build_config.json"
    for path in (manifest_path, coco_path, config_path):
        if not path.is_file():
            errors.append(f"missing required file: {path.relative_to(dataset)}")
    if errors:
        return ValidationReport(tuple(errors), tuple(warnings))

    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if categories != [
        {"id": 1, "name": "tennis_ball", "supercategory": "ball"}
    ]:
        errors.append("COCO categories must contain only class 1 tennis_ball")

    record_by_id = {record["image_id"]: record for record in records}
    if len(record_by_id) != len(records):
        errors.append("duplicate image_id values in metadata")
    coco_image_by_id = {image["id"]: image for image in images}
    if len(coco_image_by_id) != len(images):
        errors.append("duplicate image IDs in COCO document")
    if set(record_by_id) != set(coco_image_by_id):
        errors.append("COCO images and metadata image IDs do not match")

    annotations_by_image: dict[int, list[dict[str, object]]] = defaultdict(list)
    annotation_ids: set[int] = set()
    for annotation in annotations:
        annotation_id = int(annotation.get("id", -1))
        if annotation_id in annotation_ids:
            errors.append(f"duplicate annotation ID: {annotation_id}")
        annotation_ids.add(annotation_id)
        image_id = int(annotation.get("image_id", -1))
        annotations_by_image[image_id].append(annotation)
        if image_id not in record_by_id:
            errors.append(f"annotation {annotation_id} references missing image {image_id}")
        if annotation.get("category_id") != 1:
            errors.append(f"annotation {annotation_id} has invalid class ID")

    image_hashes: dict[str, str] = {}
    source_frames: set[tuple[str, int]] = set()
    groups: dict[str, str] = {}
    records_by_source: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        image_id = int(record["image_id"])
        filename = str(record["image_filename"])
        image_path = dataset / "candidates" / "images" / filename
        source_key = (
            str(record["source_video"]), int(record["source_frame_index"])
        )
        if source_key in source_frames:
            errors.append(f"duplicate extracted source frame: {source_key}")
        source_frames.add(source_key)
        records_by_source[source_key[0]].append(record)

        group = str(record["group"])
        split = str(record["split"])
        if group in groups and groups[group] != split:
            errors.append(f"temporal group {group} appears in multiple splits")
        groups[group] = split

        if not image_path.is_file():
            errors.append(f"missing extracted image: {filename}")
        else:
            digest = _sha256(image_path)
            if digest in image_hashes:
                warnings.append(
                    f"duplicate image content: {filename} and {image_hashes[digest]}"
                )
            image_hashes[digest] = filename
            decoded = cv2.imread(str(image_path))
            if decoded is None:
                errors.append(f"unreadable extracted image: {filename}")
            elif (
                decoded.shape[1] != int(record["original_width"])
                or decoded.shape[0] != int(record["original_height"])
            ):
                errors.append(f"image dimensions do not match metadata: {filename}")

        image_annotations = annotations_by_image.get(image_id, [])
        candidate_type = record["candidate_type"]
        if candidate_type == TRACKNET_PRELABEL and len(image_annotations) != 1:
            errors.append(f"prelabel image must have exactly one annotation: {filename}")
        if candidate_type == MISS_REVIEW and image_annotations:
            errors.append(f"TrackNet-miss review image must not be auto-labeled: {filename}")
        for annotation in image_annotations:
            _check_box(record, annotation, errors, warnings)

    minimum_frame_gap = int(config.get("minimum_frame_gap", 0))
    minimum_time_gap = float(config.get("minimum_time_gap", 0.0))
    for source_video, source_records in records_by_source.items():
        ordered = sorted(
            source_records, key=lambda record: int(record["source_frame_index"])
        )
        for previous, current in zip(ordered, ordered[1:], strict=False):
            frame_gap = int(current["source_frame_index"]) - int(
                previous["source_frame_index"]
            )
            time_gap = float(current["source_timestamp_seconds"]) - float(
                previous["source_timestamp_seconds"]
            )
            if frame_gap < minimum_frame_gap or time_gap < minimum_time_gap:
                errors.append(
                    f"nearly adjacent selections in {source_video}: frames "
                    f"{previous['source_frame_index']} and {current['source_frame_index']}"
                )
            if previous["split"] != current["split"]:
                guard = float(config.get("boundary_guard_seconds", 0.0))
                if time_gap < guard:
                    errors.append(
                        f"split boundary lacks configured temporal guard in "
                        f"{source_video}: {previous['split']} to {current['split']}"
                    )

    return ValidationReport(tuple(errors), tuple(warnings))


def _check_box(
    record: dict[str, object],
    annotation: dict[str, object],
    errors: list[str],
    warnings: list[str],
) -> None:
    filename = str(record["image_filename"])
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        errors.append(f"invalid COCO bbox structure: {filename}")
        return
    x, y, width, height = (float(value) for value in bbox)
    image_width = float(record["original_width"])
    image_height = float(record["original_height"])
    if width <= 0.0 or height <= 0.0:
        errors.append(f"zero-area or negative box: {filename}")
    if x < 0.0 or y < 0.0 or x + width > image_width or y + height > image_height:
        errors.append(f"box outside image bounds: {filename}")
    if width < 4.0 or height < 4.0:
        warnings.append(f"suspiciously small box: {filename}")
    if width * height > image_width * image_height * 0.02:
        warnings.append(f"suspiciously large box: {filename}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

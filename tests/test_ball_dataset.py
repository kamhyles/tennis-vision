"""Tests for tennis-ball dataset planning and annotations."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tennis_vision.dataset.ball import (
    MISS_REVIEW,
    TRACKNET_PRELABEL,
    CandidateFrame,
    DatasetRecord,
    VideoMetadata,
    build_candidate_box,
    build_coco_document,
    build_split_segments,
    segment_for_frame,
    select_diverse_frames,
)
from tennis_vision.dataset.validation import validate_dataset
from tennis_vision.vision.ball import BallDetection


def candidate(
    frame_index: int,
    *,
    x: float | None = 100.0,
    y: float | None = 100.0,
    split: str = "train",
) -> CandidateFrame:
    has_detection = x is not None and y is not None
    box = None if not has_detection else (x - 6, y - 6, x + 6, y + 6)
    return CandidateFrame(
        source_video="clip.mov",
        frame_index=frame_index,
        timestamp_seconds=frame_index / 100.0,
        width=400,
        height=200,
        split=split,
        group=f"clip_{split}",
        candidate_type=TRACKNET_PRELABEL if has_detection else MISS_REVIEW,
        tracknet_x=x,
        tracknet_y=y,
        tracknet_confidence=None,
        tracknet_box=box,
        candidate_box=box,
    )


def record(
    image_id: int,
    *,
    candidate_type: str = TRACKNET_PRELABEL,
    box: tuple[float, float, float, float] | None = (10.0, 20.0, 30.0, 40.0),
) -> DatasetRecord:
    return DatasetRecord(
        image_id=image_id,
        image_filename=f"image_{image_id}.jpg",
        preview_filename=f"image_{image_id}.jpg",
        source_video="clip.mov",
        source_frame_index=image_id * 10,
        source_timestamp_seconds=image_id / 10.0,
        original_width=400,
        original_height=200,
        tracknet_x=None if box is None else 20.0,
        tracknet_y=None if box is None else 30.0,
        tracknet_confidence=None,
        tracknet_box=box,
        candidate_box=box,
        candidate_type=candidate_type,
        split="train",
        group="clip_train",
    )


class SplitPlanningTests(unittest.TestCase):
    def test_segments_are_contiguous_regions_with_boundary_guards(self) -> None:
        video = VideoMetadata("clip.mov", 1920, 1080, 100.0, 1000, 10.0)
        segments = build_split_segments(video, boundary_guard_seconds=1.0)

        self.assertEqual(
            [(item.split, item.start_frame, item.end_frame) for item in segments],
            [
                ("train", 0, 650),
                ("validation", 750, 800),
                ("test", 900, 1000),
            ],
        )
        self.assertIsNone(segment_for_frame(segments, 700))
        self.assertEqual(segment_for_frame(segments, 760).split, "validation")

    def test_temporal_groups_never_span_splits(self) -> None:
        video = VideoMetadata("clip.mov", 640, 360, 30.0, 900, 30.0)
        segments = build_split_segments(video)
        group_to_splits: dict[str, set[str]] = {}
        for segment in segments:
            group_to_splits.setdefault(segment.group, set()).add(segment.split)
        self.assertTrue(all(len(splits) == 1 for splits in group_to_splits.values()))


class SamplingTests(unittest.TestCase):
    def test_enforces_frame_and_timestamp_spacing(self) -> None:
        pool = [candidate(index, x=float(index), y=50.0) for index in range(100)]
        selected = select_diverse_frames(
            pool,
            target_frames=20,
            minimum_frame_gap=10,
            minimum_time_gap=0.1,
            split_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
        )
        indices = sorted(item.frame_index for item in selected)
        self.assertTrue(
            all(current - previous >= 10 for previous, current in zip(indices, indices[1:]))
        )
        self.assertLessEqual(len(selected), 10)

    def test_round_robin_sampling_covers_spatial_regions(self) -> None:
        pool = [
            candidate(0, x=25.0, y=25.0),
            candidate(10, x=375.0, y=25.0),
            candidate(20, x=25.0, y=175.0),
            candidate(30, x=375.0, y=175.0),
        ]
        selected = select_diverse_frames(
            pool,
            target_frames=4,
            minimum_frame_gap=1,
            minimum_time_gap=0.0,
            miss_review_fraction=0.0,
            split_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
            spatial_columns=2,
            spatial_rows=2,
        )
        regions = {
            (int(item.tracknet_x // 200), int(item.tracknet_y // 100))
            for item in selected
            if item.tracknet_x is not None and item.tracknet_y is not None
        }
        self.assertEqual(len(regions), 4)

    def test_tracknet_misses_remain_unconfirmed_review_candidates(self) -> None:
        pool = [candidate(index * 10, x=None, y=None) for index in range(5)]
        selected = select_diverse_frames(
            pool,
            target_frames=3,
            minimum_frame_gap=1,
            minimum_time_gap=0.0,
            miss_review_fraction=1.0,
            split_fractions={"train": 1.0, "validation": 0.0, "test": 0.0},
        )
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(item.candidate_type == MISS_REVIEW for item in selected))
        self.assertTrue(all(item.candidate_box is None for item in selected))


class AnnotationTests(unittest.TestCase):
    def test_candidate_box_uses_source_coordinates_and_clips_to_bounds(self) -> None:
        detection = BallDetection(
            center_x=3.0,
            center_y=4.0,
            bounding_box=(1.0, 2.0, 5.0, 6.0),
            confidence=None,
            frame_index=10,
            timestamp_seconds=0.1,
        )
        box = build_candidate_box(
            detection,
            image_width=1920,
            image_height=1080,
            scale=1.5,
            minimum_size=12.0,
            maximum_size=48.0,
        )
        self.assertEqual(box, (0.0, 0.0, 9.0, 10.0))

    def test_metadata_contains_provenance_and_review_status(self) -> None:
        values = record(1).to_dict()
        self.assertEqual(values["source_video"], "clip.mov")
        self.assertEqual(values["source_frame_index"], 10)
        self.assertEqual(values["split"], "train")
        self.assertEqual(values["review_status"], "UNREVIEWED")

    def test_coco_has_one_class_and_no_auto_annotation_for_miss(self) -> None:
        records = (
            record(1),
            record(2, candidate_type=MISS_REVIEW, box=None),
        )
        coco = build_coco_document(records, dataset_name="test")
        self.assertEqual(
            coco["categories"],
            [{"id": 1, "name": "tennis_ball", "supercategory": "ball"}],
        )
        self.assertEqual(len(coco["images"]), 2)
        self.assertEqual(len(coco["annotations"]), 1)
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 20.0, 20.0, 20.0])


class DatasetValidationTests(unittest.TestCase):
    def test_validates_minimal_dataset_and_catches_invalid_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "metadata").mkdir()
            (root / "candidates" / "labels").mkdir(parents=True)
            (root / "candidates" / "images").mkdir()
            item = record(1)
            (root / "metadata" / "candidates.jsonl").write_text(
                json.dumps(item.to_dict()) + "\n", encoding="utf-8"
            )
            (root / "metadata" / "build_config.json").write_text(
                json.dumps(
                    {
                        "minimum_frame_gap": 8,
                        "minimum_time_gap": 0.08,
                        "boundary_guard_seconds": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            coco = build_coco_document((item,), dataset_name="test")
            labels = root / "candidates" / "labels" / "instances_candidates.json"
            labels.write_text(json.dumps(coco), encoding="utf-8")
            cv2.imwrite(
                str(root / "candidates" / "images" / item.image_filename),
                np.zeros((200, 400, 3), dtype=np.uint8),
            )
            self.assertTrue(validate_dataset(root).is_valid)

            coco["annotations"][0]["category_id"] = 2
            labels.write_text(json.dumps(coco), encoding="utf-8")
            report = validate_dataset(root)
            self.assertFalse(report.is_valid)
            self.assertTrue(any("invalid class ID" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()

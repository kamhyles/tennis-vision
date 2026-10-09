"""Tests for validation and leakage-resistant splitting of reviewed COCO data."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from tennis_vision.dataset.reviewed_coco import (
    inspect_reviewed_dataset,
    parse_source_frame,
    prepare_temporal_dataset,
    split_for_frame,
)


class ReviewedCocoTests(unittest.TestCase):
    def test_frame_provenance_and_guard_ranges(self) -> None:
        self.assertEqual(
            parse_source_frame("IMG_8356_abcd_f00003104_jpg.rf.hash.jpg"), 3104
        )
        self.assertEqual(split_for_frame(2246), "train")
        self.assertIsNone(split_for_frame(2247))
        self.assertEqual(split_for_frame(2367), "valid")
        self.assertIsNone(split_for_frame(2742))
        self.assertEqual(split_for_frame(2862), "test")

    def test_validates_and_splits_exact_space_separated_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "export" / "train"
            source.mkdir(parents=True)
            names = (
                "clip_f00000010_jpg.rf.a.jpg",
                "clip_f00002400_jpg.rf.b.jpg",
                "clip_f00003000_jpg.rf.c.jpg",
            )
            for name in names:
                cv2.imwrite(str(source / name), np.zeros((100, 200, 3), dtype=np.uint8))
            document = {
                "images": [
                    {"id": index, "file_name": name, "width": 200, "height": 100}
                    for index, name in enumerate(names, start=1)
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 20, 8, 6],
                        "area": 48,
                        "iscrowd": 0,
                    },
                    {
                        "id": 2,
                        "image_id": 2,
                        "category_id": 1,
                        "bbox": [20, 30, 12, 10],
                        "area": 120,
                        "iscrowd": 0,
                    },
                ],
                "categories": [
                    {"id": 0, "name": "Tennis-Ball-Detection"},
                    {"id": 1, "name": "tennis ball"},
                ],
            }
            (source / "_annotations.coco.json").write_text(
                json.dumps(document), encoding="utf-8"
            )

            report = inspect_reviewed_dataset(root / "export")
            self.assertTrue(report.is_valid)
            self.assertEqual(report.split_counts, {"train": 1, "valid": 1, "test": 1})
            self.assertEqual(report.zero_annotation_image_count, 1)
            self.assertEqual(report.box_width_pixels["median"], 10.0)

            output = root / "prepared"
            prepare_temporal_dataset(root / "export", output)
            for split in ("train", "valid", "test"):
                split_document = json.loads(
                    (output / split / "_annotations.coco.json").read_text()
                )
                self.assertEqual(len(split_document["images"]), 1)
                self.assertEqual(split_document["categories"][1]["name"], "tennis ball")

    def test_rejects_underscore_class_and_guard_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            name = "clip_f00002300.jpg"
            cv2.imwrite(str(source / name), np.zeros((20, 20, 3), dtype=np.uint8))
            document = {
                "images": [{"id": 1, "file_name": name, "width": 20, "height": 20}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}
                ],
                "categories": [{"id": 1, "name": "tennis_ball"}],
            }
            (source / "_annotations.coco.json").write_text(json.dumps(document))
            report = inspect_reviewed_dataset(source)
            self.assertFalse(report.is_valid)
            self.assertTrue(any("Required category" in error for error in report.errors))
            self.assertTrue(any("guard" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()

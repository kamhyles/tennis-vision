"""Tests for center-localization metrics."""

import unittest

from tennis_vision.evaluation.ball import CenterLocalizationMetrics, box_iou
from tennis_vision.vision.ball import BallDetection


def detection(box: tuple[float, float, float, float]) -> BallDetection:
    return BallDetection(
        center_x=(box[0] + box[2]) / 2,
        center_y=(box[1] + box[3]) / 2,
        bounding_box=box,
        confidence=0.8,
        frame_index=1,
        timestamp_seconds=0.1,
    )


class LocalizationMetricTests(unittest.TestCase):
    def test_exact_box_has_zero_center_error_and_unit_iou(self) -> None:
        metrics = CenterLocalizationMetrics()
        metrics.add_image(
            [(10, 20, 20, 30)],
            [detection((10, 20, 20, 30))],
            image_width=200,
            image_height=100,
        )
        summary = metrics.summary()
        self.assertEqual(summary["center_error_mean_pixels"], 0.0)
        self.assertEqual(summary["iou_mean"], 1.0)
        self.assertEqual(summary["annotated_image_recall"], 1.0)

    def test_misses_negatives_and_extra_detections_are_explicit(self) -> None:
        metrics = CenterLocalizationMetrics()
        metrics.add_image([(0, 0, 4, 4)], [], image_width=10, image_height=10)
        metrics.add_image([], [detection((1, 1, 3, 3))], image_width=10, image_height=10)
        metrics.add_image(
            [(0, 0, 4, 4)],
            [detection((0, 0, 4, 4)), detection((5, 5, 7, 7))],
            image_width=10,
            image_height=10,
        )
        summary = metrics.summary()
        self.assertEqual(summary["missed_annotated_images"], 1)
        self.assertEqual(summary["false_positive_detections_on_negative_images"], 1)
        self.assertEqual(summary["extra_detections_on_annotated_images"], 1)
        self.assertEqual(summary["annotated_image_recall"], 0.5)

    def test_iou_is_zero_for_nonoverlapping_boxes(self) -> None:
        self.assertEqual(box_iou((0, 0, 2, 2), (3, 3, 5, 5)), 0.0)


if __name__ == "__main__":
    unittest.main()

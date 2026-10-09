"""Small-object localization metrics supplementing standard COCO metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from tennis_vision.vision.ball import BallDetection

Box = tuple[float, float, float, float]


@dataclass
class CenterLocalizationMetrics:
    """Accumulate raw single-ball localization outcomes per image."""

    annotated_images: int = 0
    negative_images: int = 0
    detected_annotated_images: int = 0
    missed_annotated_images: int = 0
    false_positive_detections: int = 0
    extra_detections: int = 0
    center_errors_pixels: list[float] = field(default_factory=list)
    center_errors_diagonal_fraction: list[float] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)

    def add_image(
        self,
        ground_truth_boxes: Sequence[Box],
        predictions: Sequence[BallDetection],
        *,
        image_width: int,
        image_height: int,
    ) -> None:
        """Score the highest-confidence raw candidate for a one-ball image."""
        if len(ground_truth_boxes) > 1:
            raise ValueError("Center localization evaluation expects at most one ball")
        if not ground_truth_boxes:
            self.negative_images += 1
            self.false_positive_detections += len(predictions)
            return

        self.annotated_images += 1
        if not predictions:
            self.missed_annotated_images += 1
            return
        self.detected_annotated_images += 1
        self.extra_detections += max(0, len(predictions) - 1)
        truth = ground_truth_boxes[0]
        prediction = predictions[0]
        truth_center = ((truth[0] + truth[2]) / 2.0, (truth[1] + truth[3]) / 2.0)
        error = math.dist(truth_center, prediction.center)
        self.center_errors_pixels.append(error)
        diagonal = math.hypot(image_width, image_height)
        self.center_errors_diagonal_fraction.append(error / diagonal)
        self.ious.append(box_iou(truth, prediction.bounding_box))

    def summary(self) -> dict[str, float | int | None]:
        errors = np.asarray(self.center_errors_pixels, dtype=float)
        normalized = np.asarray(self.center_errors_diagonal_fraction, dtype=float)
        ious = np.asarray(self.ious, dtype=float)
        return {
            "annotated_images": self.annotated_images,
            "negative_images": self.negative_images,
            "detected_annotated_images": self.detected_annotated_images,
            "missed_annotated_images": self.missed_annotated_images,
            "false_positive_detections_on_negative_images": self.false_positive_detections,
            "extra_detections_on_annotated_images": self.extra_detections,
            "annotated_image_recall": _ratio(
                self.detected_annotated_images, self.annotated_images
            ),
            "center_error_mean_pixels": _mean(errors),
            "center_error_median_pixels": _median(errors),
            "center_error_p90_pixels": _percentile(errors, 90),
            "center_error_mean_diagonal_fraction": _mean(normalized),
            "center_within_5_pixels_rate": _threshold_rate(errors, 5.0),
            "center_within_10_pixels_rate": _threshold_rate(errors, 10.0),
            "center_within_20_pixels_rate": _threshold_rate(errors, 20.0),
            "iou_mean": _mean(ious),
            "iou_median": _median(ious),
        }


def coco_bbox_to_xyxy(bbox: Sequence[float]) -> Box:
    x, y, width, height = (float(value) for value in bbox)
    return x, y, x + width, y + height


def box_iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.mean(values))


def _median(values: np.ndarray) -> float | None:
    return None if values.size == 0 else float(np.median(values))


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, percentile))


def _threshold_rate(values: np.ndarray, threshold: float) -> float | None:
    return None if values.size == 0 else float(np.mean(values <= threshold))

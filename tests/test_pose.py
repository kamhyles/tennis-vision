"""Tests for pose data and primary-person selection."""

import unittest
from types import SimpleNamespace

import numpy as np

from tennis_vision.config import PoseConfig
from tennis_vision.vision.pose import (
    PoseDetection,
    PoseFrame,
    PoseTracker,
    select_primary_person,
)


def make_detection(box: tuple[float, float, float, float], score: float) -> PoseDetection:
    return PoseDetection(
        keypoints=np.zeros((17, 2), dtype=np.float32),
        confidences=np.ones(17, dtype=np.float32),
        bounding_box=np.asarray(box, dtype=np.float32),
        detection_confidence=score,
    )


class FakeTensor:
    def __init__(self, values: object) -> None:
        self.values = np.asarray(values)

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakeModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    def predict(self, **_options: object) -> list[object]:
        self.calls += 1
        return [self.result]


class PoseDataTests(unittest.TestCase):
    def test_named_keypoint_exposes_coordinates_and_confidence(self) -> None:
        detection = make_detection((0, 0, 10, 10), 0.8)
        detection.keypoints[9] = (12.5, 20.25)
        detection.confidences[9] = 0.75
        self.assertEqual(detection.keypoint("left_wrist"), (12.5, 20.25, 0.75))

    def test_unknown_keypoint_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            make_detection((0, 0, 10, 10), 0.8).keypoint("tail")

    def test_primary_person_prefers_largest_box(self) -> None:
        detections = (
            make_detection((0, 0, 20, 20), 0.99),
            make_detection((0, 0, 50, 50), 0.70),
        )
        self.assertEqual(select_primary_person(detections), 1)

    def test_primary_person_uses_confidence_as_tiebreaker(self) -> None:
        detections = (
            make_detection((0, 0, 20, 20), 0.70),
            make_detection((5, 5, 25, 25), 0.90),
        )
        self.assertEqual(select_primary_person(detections), 1)

    def test_no_detections_has_no_primary_person(self) -> None:
        frame = PoseFrame((), select_primary_person(()))
        self.assertIsNone(frame.primary)


class PoseTrackerTests(unittest.TestCase):
    def test_process_converts_model_output_to_internal_data(self) -> None:
        coordinates = np.zeros((1, 17, 2), dtype=np.float32)
        coordinates[0, 9] = (123.0, 234.0)
        result = SimpleNamespace(
            keypoints=SimpleNamespace(
                xy=FakeTensor(coordinates),
                conf=FakeTensor(np.full((1, 17), 0.8, dtype=np.float32)),
            ),
            boxes=SimpleNamespace(
                xyxy=FakeTensor([[10.0, 20.0, 300.0, 440.0]]),
                conf=FakeTensor([0.9]),
            ),
        )
        tracker = PoseTracker.__new__(PoseTracker)
        tracker.config = PoseConfig()
        tracker._model = FakeModel(result)

        pose_frame = tracker.process(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(tracker._model.calls, 1)
        self.assertIsNotNone(pose_frame.primary)
        assert pose_frame.primary is not None
        x, y, confidence = pose_frame.primary.keypoint("left_wrist")
        self.assertEqual((x, y), (123.0, 234.0))
        self.assertAlmostEqual(confidence, 0.8)
        self.assertIsInstance(pose_frame.primary.keypoints, np.ndarray)


if __name__ == "__main__":
    unittest.main()

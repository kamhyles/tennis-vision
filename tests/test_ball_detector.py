"""Tests for the detector boundary without loading model weights."""

import unittest
from types import SimpleNamespace

import numpy as np

from tennis_vision.config import BallDetectorConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.ball import BallDetector


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
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.options: dict[str, object] = {}

    def predict(self, **options: object) -> list[object]:
        self.options = options
        return self.results


def make_detector(results: list[object]) -> BallDetector:
    detector = BallDetector.__new__(BallDetector)
    detector.config = BallDetectorConfig()
    detector._model = FakeModel(results)
    return detector


class BallDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.metadata = FrameMetadata(7, 0.25, 30.0, False)

    def test_no_model_results_returns_no_detection(self) -> None:
        detector = make_detector([])
        self.assertEqual(detector.detect(self.frame, self.metadata), ())

    def test_empty_boxes_returns_no_detection(self) -> None:
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=FakeTensor(np.empty((0, 4))),
                conf=FakeTensor(np.empty((0,))),
            )
        )
        detector = make_detector([result])
        self.assertEqual(detector.detect(self.frame, self.metadata), ())

    def test_detection_is_structured_and_timestamped(self) -> None:
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxy=FakeTensor([[10.0, 20.0, 30.0, 40.0]]),
                conf=FakeTensor([0.75]),
            )
        )
        detector = make_detector([result])

        detections = detector.detect(self.frame, self.metadata)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].center, (20.0, 30.0))
        self.assertEqual(detections[0].frame_index, 7)
        self.assertEqual(detections[0].timestamp_seconds, 0.25)
        self.assertEqual(detector._model.options["classes"], [32])


if __name__ == "__main__":
    unittest.main()

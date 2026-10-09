"""RF-DETR adapter tests that do not require the optional package or weights."""

import unittest
from types import SimpleNamespace

import numpy as np

from tennis_vision.config import RFDETRConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.rfdetr import RFDETRBallDetector, detections_from_rfdetr


class FakeRFDETRModel:
    def __init__(self, result: object) -> None:
        self.result = result
        self.image: np.ndarray | None = None
        self.options: dict[str, object] = {}

    def predict(self, image: np.ndarray, **options: object) -> object:
        self.image = image
        self.options = options
        return self.result


class RFDETRAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = FrameMetadata(9, 0.075, 120.0, False)

    def test_conversion_preserves_raw_source_coordinates_and_exact_class(self) -> None:
        result = SimpleNamespace(
            xyxy=np.array([[100, 200, 112, 214], [1, 2, 10, 12]], dtype=float),
            confidence=np.array([0.8, 0.99]),
            class_id=np.array([0, 0]),
            data={"class_name": np.array(["tennis ball", "Tennis-Ball-Detection"])},
        )
        detections = detections_from_rfdetr(result, self.metadata)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bounding_box, (100.0, 200.0, 112.0, 214.0))
        self.assertEqual(detections[0].center, (106.0, 207.0))
        self.assertEqual(detections[0].frame_index, 9)
        self.assertEqual(detections[0].detector_name, "rfdetr")

    def test_detector_converts_bgr_to_rgb_and_passes_resolution(self) -> None:
        result = SimpleNamespace(
            xyxy=np.empty((0, 4)), confidence=np.empty((0,)), class_id=np.empty((0,)), data={}
        )
        detector = RFDETRBallDetector.__new__(RFDETRBallDetector)
        detector.config = RFDETRConfig(input_size=1024)
        detector._model = FakeRFDETRModel(result)
        frame = np.zeros((2, 3, 3), dtype=np.uint8)
        frame[0, 0] = [10, 20, 30]

        self.assertEqual(detector.detect(frame, self.metadata), ())
        np.testing.assert_array_equal(detector._model.image[0, 0], [30, 20, 10])
        self.assertEqual(detector._model.options["shape"], (1024, 1024))
        self.assertEqual(detector._model.options["threshold"], 0.25)
        self.assertFalse(detector._model.options["include_source_image"])

    def test_candidates_are_confidence_sorted_and_none_is_last(self) -> None:
        result = SimpleNamespace(
            xyxy=np.array([[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]]),
            confidence=np.array([0.2, np.nan, 0.9]),
            class_id=np.array([0, 0, 0]),
            data={},
        )
        detections = detections_from_rfdetr(result, self.metadata)
        self.assertEqual([item.confidence for item in detections], [0.9, 0.2, None])

    def test_conversion_defensively_applies_confidence_threshold(self) -> None:
        result = SimpleNamespace(
            xyxy=np.array([[0, 0, 2, 2], [2, 2, 4, 4]]),
            confidence=np.array([0.24, 0.25]),
            class_id=np.array([0, 0]),
            data={},
        )
        detections = detections_from_rfdetr(
            result, self.metadata, minimum_confidence=0.25
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, 0.25)


if __name__ == "__main__":
    unittest.main()

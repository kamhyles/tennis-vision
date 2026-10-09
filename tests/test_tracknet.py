"""Tests for TrackNet temporal input and output conversion."""

import unittest

import cv2
import numpy as np

from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.tracknet import (
    BufferedTrackNetFrame,
    CoordinateTransform,
    TemporalFrameBuffer,
    decode_tracknet_class_map,
    make_temporal_tensor,
    make_tracknet_detection,
    preprocess_frame,
)


def metadata(index: int) -> FrameMetadata:
    return FrameMetadata(index, index / 120.0, 120.0, False)


def buffered(index: int) -> BufferedTrackNetFrame:
    image = np.full((4, 6, 3), index, dtype=np.float32)
    return BufferedTrackNetFrame(image, metadata(index))


class TemporalFrameBufferTests(unittest.TestCase):
    def test_requires_three_consecutive_frames_and_returns_newest_first(self) -> None:
        buffer = TemporalFrameBuffer(3)
        buffer.append(buffered(0))
        buffer.append(buffered(1))
        self.assertFalse(buffer.ready)
        buffer.append(buffered(2))

        frames = buffer.newest_first()

        self.assertTrue(buffer.ready)
        self.assertEqual([item.metadata.frame_index for item in frames], [2, 1, 0])

    def test_nonconsecutive_frame_resets_temporal_context(self) -> None:
        buffer = TemporalFrameBuffer(3)
        buffer.append(buffered(0))
        buffer.append(buffered(1))
        buffer.append(buffered(4))
        self.assertFalse(buffer.ready)
        self.assertEqual(buffer.newest_first(), ())

    def test_clear_discards_context_between_source_videos(self) -> None:
        buffer = TemporalFrameBuffer(3)
        buffer.append(buffered(0))
        buffer.append(buffered(1))
        buffer.append(buffered(2))
        buffer.clear()
        self.assertFalse(buffer.ready)
        self.assertEqual(buffer.newest_first(), ())

    def test_tensor_preserves_newest_first_bgr_channel_groups(self) -> None:
        frames = (buffered(2), buffered(1), buffered(0))
        tensor = make_temporal_tensor(frames)
        self.assertEqual(tuple(tensor.shape), (1, 9, 4, 6))
        self.assertTrue(np.allclose(tensor[0, 0:3].numpy(), 2.0))
        self.assertTrue(np.allclose(tensor[0, 3:6].numpy(), 1.0))
        self.assertTrue(np.allclose(tensor[0, 6:9].numpy(), 0.0))


class TrackNetGeometryTests(unittest.TestCase):
    def test_preprocessing_resizes_copy_and_normalizes(self) -> None:
        source = np.full((12, 20, 3), 255, dtype=np.uint8)
        processed = preprocess_frame(source, width=10, height=6)
        self.assertEqual(processed.shape, (6, 10, 3))
        self.assertEqual(processed.dtype, np.float32)
        self.assertAlmostEqual(float(processed.max()), 1.0)
        self.assertEqual(source.shape, (12, 20, 3))

    def test_coordinates_map_back_to_original_1080p_frame(self) -> None:
        transform = CoordinateTransform(1920, 1080, 640, 360)
        self.assertEqual(transform.model_to_source(320.0, 180.0), (960.0, 540.0))

    def test_decoded_output_converts_to_shared_detection(self) -> None:
        result = make_tracknet_detection(
            (320.0, 180.0, 4.0),
            metadata(12),
            source_width=1920,
            source_height=1080,
            model_width=640,
            model_height=360,
        )
        self.assertEqual(result.center, (960.0, 540.0))
        self.assertEqual(result.frame_index, 12)
        self.assertIsNone(result.confidence)
        self.assertEqual(result.bounding_box, (948.0, 528.0, 972.0, 552.0))

    def test_missing_heatmap_detection_returns_none(self) -> None:
        class_map = np.zeros((360, 640), dtype=np.uint8)
        self.assertIsNone(decode_tracknet_class_map(class_map, threshold=127))

    def test_bright_heatmap_circle_is_decoded(self) -> None:
        class_map = np.zeros((360, 640), dtype=np.uint8)
        cv2.circle(class_map, (100, 80), 4, 255, -1)
        decoded = decode_tracknet_class_map(class_map, threshold=127)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertAlmostEqual(decoded[0], 100.0, delta=2.0)
        self.assertAlmostEqual(decoded[1], 80.0, delta=2.0)


if __name__ == "__main__":
    unittest.main()

"""Tests for source-frame timing without opening a camera or file."""

import unittest

import cv2
import numpy as np

from tennis_vision.media import FrameSource


class FakeCapture:
    def __init__(self, fps: float, timestamps_ms: list[float]) -> None:
        self.fps = fps
        self.timestamps_ms = timestamps_ms
        self.read_count = 0
        self.released = False

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self.fps
        if property_id == cv2.CAP_PROP_POS_MSEC and self.read_count:
            return self.timestamps_ms[self.read_count - 1]
        return 0.0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_count >= len(self.timestamps_ms):
            return False, None
        self.read_count += 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


class FrameSourceTests(unittest.TestCase):
    def test_video_preserves_fps_frame_index_and_source_timestamp(self) -> None:
        capture = FakeCapture(120.0, [0.0, 8.333, 16.667])
        source = FrameSource(
            capture,  # type: ignore[arg-type]
            is_video=True,
            source_name="fast.mp4",
        )

        frames = [source.read(), source.read(), source.read()]

        self.assertEqual(source.source_fps, 120.0)
        self.assertEqual(
            [item.metadata.frame_index for item in frames if item], [0, 1, 2]
        )
        self.assertAlmostEqual(
            frames[1].metadata.timestamp_seconds, 0.008333  # type: ignore[union-attr]
        )
        self.assertFalse(frames[2].metadata.timing_is_estimated)  # type: ignore[union-attr]
        self.assertIsNone(source.read())

    def test_video_falls_back_to_fps_when_position_metadata_stalls(self) -> None:
        capture = FakeCapture(60.0, [0.0, 0.0])
        source = FrameSource(
            capture,  # type: ignore[arg-type]
            is_video=True,
            source_name="clip.mp4",
        )
        source.read()

        second = source.read()

        assert second is not None
        self.assertAlmostEqual(second.metadata.timestamp_seconds, 1.0 / 60.0)
        self.assertFalse(second.metadata.timing_is_estimated)


if __name__ == "__main__":
    unittest.main()

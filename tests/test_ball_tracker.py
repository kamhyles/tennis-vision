"""Synthetic temporal tests for single-ball tracking."""

import unittest

from tennis_vision.config import BallTrackerConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.tracking.ball import BallTracker, TrackState
from tennis_vision.vision.ball import BallDetection


def metadata(frame_index: int, fps: float = 30.0) -> FrameMetadata:
    return FrameMetadata(frame_index, frame_index / fps, fps, False)


def detection(frame_index: int, x: float, y: float = 50.0) -> BallDetection:
    return BallDetection(
        center_x=x,
        center_y=y,
        bounding_box=(x - 3.0, y - 3.0, x + 3.0, y + 3.0),
        confidence=0.9,
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
    )


class BallTrackerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker = BallTracker(
            BallTrackerConfig(
                maximum_missed_detections=2,
                base_jump_tolerance_pixels=20.0,
                maximum_speed_pixels_per_second=600.0,
                velocity_smoothing=1.0,
            )
        )

    def test_normal_continuous_motion(self) -> None:
        for index, x in enumerate((100.0, 110.0, 120.0, 130.0)):
            track = self.tracker.update([detection(index, x)], metadata(index))
            self.assertIs(track.state, TrackState.DETECTED)
        self.assertEqual(track.position, (130.0, 50.0))
        self.assertEqual(len(track.history), 4)
        self.assertTrue(all(point.observed for point in track.history))

    def test_temporary_missed_detection_predicts(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))
        self.tracker.update([detection(1, 110.0)], metadata(1))

        track = self.tracker.update([], metadata(2))

        self.assertIs(track.state, TrackState.PREDICTED)
        self.assertEqual(track.missed_detections, 1)
        self.assertAlmostEqual(track.position[0], 120.0)  # type: ignore[index]
        self.assertFalse(track.history[-1].observed)

    def test_track_recovers_after_missed_detection(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))
        self.tracker.update([detection(1, 110.0)], metadata(1))
        self.tracker.update([], metadata(2))

        track = self.tracker.update([detection(3, 130.0)], metadata(3))

        self.assertIs(track.state, TrackState.DETECTED)
        self.assertEqual(track.missed_detections, 0)
        self.assertEqual(track.position, (130.0, 50.0))

    def test_implausible_position_jump_is_rejected(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))

        track = self.tracker.update([detection(1, 900.0)], metadata(1))

        self.assertIs(track.state, TrackState.PREDICTED)
        self.assertNotEqual(track.position, (900.0, 50.0))
        self.assertEqual(track.missed_detections, 1)

    def test_prolonged_disappearance_becomes_lost(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))
        self.tracker.update([], metadata(1))
        self.tracker.update([], metadata(2))

        track = self.tracker.update([], metadata(3))

        self.assertIs(track.state, TrackState.LOST)
        self.assertIsNone(track.position)
        self.assertEqual(track.history, ())

    def test_new_track_starts_after_previous_track_is_lost(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))
        self.tracker.update([], metadata(1))
        self.tracker.update([], metadata(2))
        self.tracker.update([], metadata(3))

        track = self.tracker.update([detection(4, 700.0)], metadata(4))

        self.assertIs(track.state, TrackState.DETECTED)
        self.assertEqual(track.position, (700.0, 50.0))
        self.assertEqual(len(track.history), 1)

    def test_scheduled_detector_skip_does_not_count_as_a_miss(self) -> None:
        self.tracker.update([detection(0, 100.0)], metadata(0))
        track = self.tracker.update(
            [], metadata(1), detection_attempted=False
        )
        self.assertIs(track.state, TrackState.PREDICTED)
        self.assertEqual(track.missed_detections, 0)


if __name__ == "__main__":
    unittest.main()

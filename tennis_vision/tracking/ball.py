"""Explainable single-ball temporal tracking in image coordinates."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from tennis_vision.config import BallTrackerConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.ball import BallDetection


class TrackState(str, Enum):
    DETECTED = "DETECTED"
    PREDICTED = "PREDICTED"
    LOST = "LOST"


@dataclass(frozen=True)
class TrackPoint:
    x: float
    y: float
    frame_index: int
    timestamp_seconds: float
    observed: bool

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class BallTrack:
    state: TrackState
    position: tuple[float, float] | None
    history: tuple[TrackPoint, ...]
    missed_detections: int
    detection: BallDetection | None = None


class BallTracker:
    """Track one ball with constant-velocity prediction and motion gating."""

    def __init__(self, config: BallTrackerConfig | None = None) -> None:
        self.config = config or BallTrackerConfig()
        if self.config.maximum_missed_detections < 0:
            raise ValueError("maximum_missed_detections cannot be negative")
        if self.config.maximum_history < 1:
            raise ValueError("maximum_history must be positive")
        if not 0.0 <= self.config.velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1]")
        self._history: deque[TrackPoint] = deque(
            maxlen=self.config.maximum_history
        )
        self._state = TrackState.LOST
        self._position: tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._last_metadata: FrameMetadata | None = None
        self._missed_detections = 0

    def update(
        self,
        detections: Sequence[BallDetection],
        metadata: FrameMetadata,
        *,
        detection_attempted: bool = True,
    ) -> BallTrack:
        """Advance the track for one source frame.

        Scheduled detector skips can set ``detection_attempted=False`` so they
        produce a prediction without being counted as failed observations.
        """
        if (
            self._last_metadata
            and metadata.frame_index <= self._last_metadata.frame_index
        ):
            raise ValueError("frame indices must increase monotonically")

        if self._state is TrackState.LOST or self._position is None:
            if not detections:
                self._last_metadata = metadata
                return self._snapshot()
            detection = max(
                detections,
                key=lambda item: (
                    -1.0 if item.confidence is None else item.confidence
                ),
            )
            self._start_track(detection, metadata)
            return self._snapshot(detection)

        elapsed = self._elapsed_seconds(metadata)
        predicted = (
            self._position[0] + self._velocity[0] * elapsed,
            self._position[1] + self._velocity[1] * elapsed,
        )
        detection = self._nearest_plausible_detection(detections, predicted, elapsed)

        if detection is not None:
            measured_velocity = (
                (detection.center_x - self._position[0]) / elapsed,
                (detection.center_y - self._position[1]) / elapsed,
            )
            weight = self.config.velocity_smoothing
            self._velocity = (
                weight * measured_velocity[0] + (1.0 - weight) * self._velocity[0],
                weight * measured_velocity[1] + (1.0 - weight) * self._velocity[1],
            )
            self._position = detection.center
            self._state = TrackState.DETECTED
            self._missed_detections = 0
            self._append_point(metadata, observed=True)
            self._last_metadata = metadata
            return self._snapshot(detection)

        if detection_attempted:
            self._missed_detections += 1
        if self._missed_detections > self.config.maximum_missed_detections:
            self.reset(metadata)
            return self._snapshot()

        self._position = predicted
        self._state = TrackState.PREDICTED
        self._append_point(metadata, observed=False)
        self._last_metadata = metadata
        return self._snapshot()

    def reset(self, metadata: FrameMetadata | None = None) -> None:
        """Drop all trajectory state after a prolonged absence."""
        self._history.clear()
        self._state = TrackState.LOST
        self._position = None
        self._velocity = (0.0, 0.0)
        self._last_metadata = metadata
        self._missed_detections = 0

    def _start_track(
        self, detection: BallDetection, metadata: FrameMetadata
    ) -> None:
        self._history.clear()
        self._position = detection.center
        self._velocity = (0.0, 0.0)
        self._state = TrackState.DETECTED
        self._missed_detections = 0
        self._last_metadata = metadata
        self._append_point(metadata, observed=True)

    def _elapsed_seconds(self, metadata: FrameMetadata) -> float:
        assert self._last_metadata is not None
        elapsed = metadata.timestamp_seconds - self._last_metadata.timestamp_seconds
        if elapsed > 0.0:
            return elapsed
        if metadata.source_fps and metadata.source_fps > 0.0:
            frame_delta = metadata.frame_index - self._last_metadata.frame_index
            return frame_delta / metadata.source_fps
        raise ValueError("timestamps must increase when source FPS is unavailable")

    def _nearest_plausible_detection(
        self,
        detections: Sequence[BallDetection],
        predicted: tuple[float, float],
        elapsed: float,
    ) -> BallDetection | None:
        if not detections:
            return None
        candidate = min(
            detections,
            key=lambda item: math.dist(item.center, predicted),
        )
        maximum_distance = (
            self.config.base_jump_tolerance_pixels
            + self.config.maximum_speed_pixels_per_second * elapsed
        )
        if math.dist(candidate.center, predicted) > maximum_distance:
            return None
        return candidate

    def _append_point(self, metadata: FrameMetadata, *, observed: bool) -> None:
        assert self._position is not None
        self._history.append(
            TrackPoint(
                x=self._position[0],
                y=self._position[1],
                frame_index=metadata.frame_index,
                timestamp_seconds=metadata.timestamp_seconds,
                observed=observed,
            )
        )

    def _snapshot(self, detection: BallDetection | None = None) -> BallTrack:
        return BallTrack(
            state=self._state,
            position=self._position,
            history=tuple(self._history),
            missed_detections=self._missed_detections,
            detection=detection,
        )

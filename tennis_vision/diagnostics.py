"""Operational ball-tracking metrics and optional per-frame CSV output."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TextIO

from tennis_vision.frames import FrameMetadata
from tennis_vision.tracking.ball import BallTrack, TrackState
from tennis_vision.vision.ball import BallDetection


class BallRunDiagnostics:
    """Count operational states without presenting them as accuracy metrics."""

    def __init__(self) -> None:
        self.total_frames = 0
        self.direct_localizations = 0
        self.predicted_frames = 0
        self.lost_frames = 0
        self.track_interruptions = 0
        self.longest_continuous_track = 0
        self._current_continuous_track = 0
        self._previous_state = TrackState.LOST

    def observe(self, track: BallTrack, *, had_direct_localization: bool) -> None:
        self.total_frames += 1
        if had_direct_localization:
            self.direct_localizations += 1
        if track.state is TrackState.PREDICTED:
            self.predicted_frames += 1
        elif track.state is TrackState.LOST:
            self.lost_frames += 1

        if track.state is TrackState.LOST:
            if self._previous_state is not TrackState.LOST:
                self.track_interruptions += 1
            self._current_continuous_track = 0
        else:
            self._current_continuous_track += 1
            self.longest_continuous_track = max(
                self.longest_continuous_track,
                self._current_continuous_track,
            )
        self._previous_state = track.state

    def summary(self) -> str:
        percentage = (
            0.0
            if not self.total_frames
            else 100.0 * self.direct_localizations / self.total_frames
        )
        return (
            f"frames={self.total_frames}, direct={self.direct_localizations} "
            f"({percentage:.1f}%), predicted={self.predicted_frames}, "
            f"lost={self.lost_frames}, interruptions={self.track_interruptions}, "
            f"longest_track={self.longest_continuous_track} frames"
        )


class DetectionCsvWriter:
    """Write detector and downstream tracking output for each source frame."""

    def __init__(self, output_path: str | None) -> None:
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        if output_path:
            path = Path(output_path).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", newline="", encoding="utf-8")
            fields = (
                "frame_index",
                "timestamp_seconds",
                "detector",
                "detection_valid",
                "x",
                "y",
                "confidence",
                "tracking_state",
            )
            self._writer = csv.DictWriter(self._file, fieldnames=fields)
            self._writer.writeheader()

    def write(
        self,
        metadata: FrameMetadata,
        detector_name: str,
        track: BallTrack,
        direct_detection: BallDetection | None,
    ) -> None:
        if self._writer is None:
            return
        position = (
            direct_detection.center
            if direct_detection is not None
            else track.position
        )
        self._writer.writerow(
            {
                "frame_index": metadata.frame_index,
                "timestamp_seconds": f"{metadata.timestamp_seconds:.9f}",
                "detector": detector_name,
                "detection_valid": int(direct_detection is not None),
                "x": "" if position is None else f"{position[0]:.3f}",
                "y": "" if position is None else f"{position[1]:.3f}",
                "confidence": (
                    ""
                    if direct_detection is None
                    or direct_detection.confidence is None
                    else f"{direct_detection.confidence:.6f}"
                ),
                "tracking_state": track.state.value,
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

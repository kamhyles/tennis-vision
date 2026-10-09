"""OpenCV camera/video input with preserved source timing."""

from __future__ import annotations

import math
import platform
import time
from pathlib import Path

import cv2

from tennis_vision.config import CameraConfig
from tennis_vision.frames import FrameMetadata, SourceFrame


def _valid_fps(value: float) -> float | None:
    return value if math.isfinite(value) and value > 0.0 else None


class FrameSource:
    """Read timestamped frames from either a camera or a video file."""

    def __init__(
        self,
        capture: cv2.VideoCapture,
        *,
        is_video: bool,
        source_name: str,
    ) -> None:
        self._capture = capture
        self.is_video = is_video
        self.source_name = source_name
        self.source_fps = _valid_fps(capture.get(cv2.CAP_PROP_FPS))
        self._frame_index = 0
        self._started_at = time.perf_counter()
        self._last_timestamp = -1.0

    def read(self) -> SourceFrame | None:
        success, image = self._capture.read()
        if not success:
            return None

        timestamp, estimated = self._timestamp_for_current_frame()
        metadata = FrameMetadata(
            frame_index=self._frame_index,
            timestamp_seconds=timestamp,
            source_fps=self.source_fps,
            timing_is_estimated=estimated,
        )
        self._frame_index += 1
        self._last_timestamp = timestamp
        return SourceFrame(image=image, metadata=metadata)

    def _timestamp_for_current_frame(self) -> tuple[float, bool]:
        if not self.is_video:
            return time.perf_counter() - self._started_at, False

        position_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
        position_seconds = position_ms / 1000.0
        if (
            math.isfinite(position_seconds)
            and position_seconds >= 0.0
            and (self._frame_index == 0 or position_seconds > self._last_timestamp)
        ):
            return position_seconds, False
        if self.source_fps is not None:
            return self._frame_index / self.source_fps, False
        # Preserve that this is processing time rather than true source time.
        return time.perf_counter() - self._started_at, True

    def release(self) -> None:
        self._capture.release()


def open_source(
    camera_config: CameraConfig, video_path: str | None = None
) -> FrameSource:
    """Open a video file or a camera with useful, specific errors."""
    if video_path:
        path = Path(video_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"Video file does not exist: {path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open video file: {path}")
        return FrameSource(capture, is_video=True, source_name=str(path))

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_config.index, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open camera index {camera_config.index}. Check camera "
            "permissions and whether another application is using it."
        )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config.height)
    return FrameSource(
        capture, is_video=False, source_name=f"camera:{camera_config.index}"
    )

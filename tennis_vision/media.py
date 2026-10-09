"""OpenCV camera/video input with preserved source timing."""

from __future__ import annotations

import math
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from tennis_vision.config import CameraConfig
from tennis_vision.frames import FrameMetadata, SourceFrame


def _valid_fps(value: float) -> float | None:
    return value if math.isfinite(value) and value > 0.0 else None


class TimestampFrameSampler:
    """Select constant-rate output frames without changing source-time duration."""

    def __init__(self, output_fps: float) -> None:
        if _valid_fps(output_fps) is None:
            raise ValueError("output FPS must be a finite positive number")
        self.output_fps = output_fps
        self._next_output_index = 0

    def copies_for(self, timestamp_seconds: float) -> int:
        """Return how many copies are due at this source timestamp.

        High-frame-rate inputs are sampled down. If source timestamps contain a
        gap, frames are duplicated to preserve elapsed source time rather than
        shortening the exported video.
        """
        if not math.isfinite(timestamp_seconds) or timestamp_seconds < 0.0:
            raise ValueError("frame timestamp must be finite and non-negative")
        due_index = int(math.floor(timestamp_seconds * self.output_fps + 1e-6))
        if due_index < self._next_output_index:
            return 0
        copies = due_index - self._next_output_index + 1
        self._next_output_index = due_index + 1
        return copies


class AnnotatedVideoWriter:
    """Stream timestamp-sampled BGR frames to an H.264 MP4 using FFmpeg."""

    def __init__(
        self,
        output_path: str,
        *,
        width: int,
        height: int,
        output_fps: float,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("output video dimensions must be positive")

        path = Path(output_path).expanduser()
        if path.suffix.lower() != ".mp4":
            raise ValueError("--output-video must use an .mp4 filename")
        if path.exists():
            raise FileExistsError(
                f"Output video already exists: {path}. Choose a new path or "
                "remove it before exporting again."
            )

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "FFmpeg is required for --output-video but was not found on PATH"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_fd, temporary_name = tempfile.mkstemp(
            prefix=f".{path.stem}.", suffix=".mp4", dir=path.parent
        )
        os.close(temporary_fd)

        self.output_path = path
        self.width = width
        self.height = height
        self.output_fps = output_fps
        self._temporary_path = Path(temporary_name)
        self._sampler = TimestampFrameSampler(output_fps)
        self._closed = False
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{output_fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._temporary_path),
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError:
            self._temporary_path.unlink(missing_ok=True)
            raise

    def write(
        self, frame: npt.NDArray[np.uint8], timestamp_seconds: float
    ) -> None:
        """Write this annotated frame when required by the output timeline."""
        if self._closed:
            raise RuntimeError("cannot write to a closed output video")
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError(
                "output frames must be uint8 BGR images matching the source size"
            )
        copies = self._sampler.copies_for(timestamp_seconds)
        if copies == 0:
            return
        assert self._process.stdin is not None
        encoded_frame = frame.tobytes()
        try:
            for _ in range(copies):
                self._process.stdin.write(encoded_frame)
        except BrokenPipeError as exc:
            message = self._read_encoder_error()
            self.abort()
            raise RuntimeError(f"FFmpeg stopped during video export: {message}") from exc

    def close(self) -> None:
        """Finish encoding and atomically publish the completed MP4."""
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        return_code = self._process.wait()
        message = self._read_encoder_error()
        if return_code != 0:
            self._temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FFmpeg could not finish the output video: {message or return_code}"
            )
        if self.output_path.exists():
            self._temporary_path.unlink(missing_ok=True)
            raise FileExistsError(
                f"Output video appeared while exporting: {self.output_path}"
            )
        self._temporary_path.replace(self.output_path)

    def abort(self) -> None:
        """Discard a partial export after an upstream processing failure."""
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        self._temporary_path.unlink(missing_ok=True)

    def _read_encoder_error(self) -> str:
        if self._process.stderr is None:
            return ""
        return self._process.stderr.read().decode("utf-8", errors="replace").strip()


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

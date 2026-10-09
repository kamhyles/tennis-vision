"""Temporal TrackNet tennis-ball detector backend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import torch

from tennis_vision.config import TrackNetConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.ball import BallDetection
from tennis_vision.vision.tracknet_model import TrackNetV1


@dataclass(frozen=True)
class CoordinateTransform:
    """Explicit direct-resize transform between model and source coordinates."""

    source_width: int
    source_height: int
    model_width: int
    model_height: int

    def model_to_source(self, x: float, y: float) -> tuple[float, float]:
        return (
            x * self.source_width / self.model_width,
            y * self.source_height / self.model_height,
        )


@dataclass(frozen=True)
class BufferedTrackNetFrame:
    image: npt.NDArray[np.float32]
    metadata: FrameMetadata


class TemporalFrameBuffer:
    """Keep exactly the consecutive preprocessed frames TrackNet expects."""

    def __init__(self, length: int) -> None:
        if length < 1:
            raise ValueError("temporal frame-buffer length must be positive")
        self.length = length
        self._frames: deque[BufferedTrackNetFrame] = deque(maxlen=length)

    def append(self, frame: BufferedTrackNetFrame) -> None:
        if self._frames and frame.metadata.frame_index != (
            self._frames[-1].metadata.frame_index + 1
        ):
            self._frames.clear()
        self._frames.append(frame)

    @property
    def ready(self) -> bool:
        return len(self._frames) == self.length

    def newest_first(self) -> tuple[BufferedTrackNetFrame, ...]:
        if not self.ready:
            return ()
        return tuple(reversed(self._frames))

    def clear(self) -> None:
        """Discard temporal context before beginning a different source."""
        self._frames.clear()


def preprocess_frame(
    frame: npt.NDArray[np.uint8], width: int, height: int
) -> npt.NDArray[np.float32]:
    """Direct-resize BGR input as used by the selected tennis checkpoint."""
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def make_temporal_tensor(
    frames_newest_first: Sequence[BufferedTrackNetFrame],
) -> torch.Tensor:
    """Stack newest, previous, and pre-previous BGR frames into NCHW."""
    images = [item.image for item in frames_newest_first]
    stacked = np.concatenate(images, axis=2)
    channels_first = np.moveaxis(stacked, 2, 0)
    return torch.from_numpy(np.expand_dims(channels_first, axis=0)).float()


def decode_tracknet_class_map(
    class_map: npt.NDArray[np.uint8], threshold: int
) -> tuple[float, float, float] | None:
    """Decode the TrackNet intensity-class heatmap using its Hough contract."""
    _, heatmap = cv2.threshold(class_map, threshold, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=1,
        param1=50,
        param2=2,
        minRadius=2,
        maxRadius=7,
    )
    if circles is None:
        return None
    x, y, radius = circles[0][0]
    return float(x), float(y), float(radius)


def resolve_torch_device(requested: str | None) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch build")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_tracknet_detection(
    decoded: tuple[float, float, float],
    metadata: FrameMetadata,
    *,
    source_width: int,
    source_height: int,
    model_width: int,
    model_height: int,
) -> BallDetection:
    """Convert a decoded model-space circle to the shared source-space type."""
    model_x, model_y, model_radius = decoded
    transform = CoordinateTransform(
        source_width=source_width,
        source_height=source_height,
        model_width=model_width,
        model_height=model_height,
    )
    source_x, source_y = transform.model_to_source(model_x, model_y)
    radius_x = model_radius * source_width / model_width
    radius_y = model_radius * source_height / model_height
    return BallDetection(
        center_x=source_x,
        center_y=source_y,
        bounding_box=(
            source_x - radius_x,
            source_y - radius_y,
            source_x + radius_x,
            source_y + radius_y,
        ),
        # TrackNet's 256 intensity classes do not provide a calibrated
        # object-confidence score.
        confidence=None,
        frame_index=metadata.frame_index,
        timestamp_seconds=metadata.timestamp_seconds,
        detector_name="tracknet",
    )


class TrackNetBallDetector:
    """Experimental three-source-frame detector with tennis-trained weights."""

    def __init__(self, config: TrackNetConfig | None = None) -> None:
        self.config = config or TrackNetConfig()
        if self.config.temporal_frames != 3:
            raise ValueError("the selected TrackNet checkpoint requires 3 frames")
        checkpoint = Path(self.config.checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise RuntimeError(
                f"TrackNet checkpoint not found: {checkpoint}. "
                "Run: python scripts/download_tracknet_weights.py"
            )
        self.device = resolve_torch_device(self.config.device)
        self._model = TrackNetV1()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self._model.load_state_dict(state, strict=True)
        self._model.to(self.device)
        self._model.eval()
        self._buffer = TemporalFrameBuffer(self.config.temporal_frames)

    @property
    def name(self) -> str:
        return "tracknet"

    def reset(self) -> None:
        """Discard buffered frames when switching source videos."""
        self._buffer.clear()

    def detect(
        self, frame: npt.NDArray[np.uint8], metadata: FrameMetadata
    ) -> tuple[BallDetection, ...]:
        """Buffer a source frame and localize the ball once three are ready."""
        source_height, source_width = frame.shape[:2]
        processed = preprocess_frame(
            frame, self.config.input_width, self.config.input_height
        )
        self._buffer.append(BufferedTrackNetFrame(processed, metadata))
        temporal_frames = self._buffer.newest_first()
        if not temporal_frames:
            return ()

        inputs = make_temporal_tensor(temporal_frames).to(self.device)
        with torch.inference_mode():
            logits = self._model(inputs)
        class_map = (
            logits.argmax(dim=1)
            .reshape(self.config.input_height, self.config.input_width)
            .to(dtype=torch.uint8)
            .cpu()
            .numpy()
        )
        decoded = decode_tracknet_class_map(
            class_map, self.config.heatmap_threshold
        )
        if decoded is None:
            return ()

        return (
            make_tracknet_detection(
                decoded,
                metadata,
                source_width=source_width,
                source_height=source_height,
                model_width=self.config.input_width,
                model_height=self.config.input_height,
            ),
        )

"""Replaceable tennis-ball detection boundary using provisional COCO weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from tennis_vision.config import BallDetectorConfig
from tennis_vision.frames import FrameMetadata

if TYPE_CHECKING:
    from ultralytics import YOLO


@dataclass(frozen=True)
class BallDetection:
    """One image-space ball candidate without detector-specific objects."""

    center_x: float
    center_y: float
    bounding_box: tuple[float, float, float, float]
    confidence: float | None
    frame_index: int
    timestamp_seconds: float
    detector_name: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.center_x, self.center_y

    @property
    def is_valid(self) -> bool:
        """A materialized detection represents a valid direct localization."""
        return True


class YOLOBallDetector:
    """Detect COCO class 32 (sports ball) behind a replaceable interface.

    Generic COCO weights are provisional: small, blurred tennis balls are often
    outside their reliable operating range. No color assumption is made here.
    """

    def __init__(self, config: BallDetectorConfig | None = None) -> None:
        self.config = config or BallDetectorConfig()
        from ultralytics import YOLO

        self._model: YOLO = YOLO(self.config.model_path)

    @property
    def name(self) -> str:
        return "yolo"

    def detect(
        self, frame: npt.NDArray[np.uint8], metadata: FrameMetadata
    ) -> tuple[BallDetection, ...]:
        """Return all sports-ball candidates for one source frame."""
        options: dict[str, Any] = {
            "source": frame,
            "conf": self.config.confidence_threshold,
            "imgsz": self.config.image_size,
            "classes": [self.config.sports_ball_class_id],
            "verbose": False,
        }
        if self.config.device:
            options["device"] = self.config.device

        results = self._model.predict(**options)
        if not results or results[0].boxes is None:
            return ()

        boxes = results[0].boxes.xyxy.detach().cpu().numpy()
        confidences = results[0].boxes.conf.detach().cpu().numpy()
        detections: list[BallDetection] = []
        for box, confidence in zip(boxes, confidences, strict=False):
            x1, y1, x2, y2 = (float(value) for value in box)
            detections.append(
                BallDetection(
                    center_x=(x1 + x2) / 2.0,
                    center_y=(y1 + y2) / 2.0,
                    bounding_box=(x1, y1, x2, y2),
                    confidence=float(confidence),
                    frame_index=metadata.frame_index,
                    timestamp_seconds=metadata.timestamp_seconds,
                    detector_name="yolo",
                )
            )
        return tuple(detections)


# Backwards-compatible name for the original V0.2 detector API.
BallDetector = YOLOBallDetector

"""YOLO-based human pose inference with model-independent output types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import numpy.typing as npt

from tennis_vision.config import PoseConfig

if TYPE_CHECKING:
    from ultralytics import YOLO

FloatArray = npt.NDArray[np.float32]

# COCO keypoint ordering used by Ultralytics pose models.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PoseDetection:
    """Numerical pose output for one person, independent of Ultralytics."""

    keypoints: FloatArray
    confidences: FloatArray
    bounding_box: FloatArray
    detection_confidence: float

    def __post_init__(self) -> None:
        if self.keypoints.shape != (len(KEYPOINT_NAMES), 2):
            raise ValueError("keypoints must have shape (17, 2)")
        if self.confidences.shape != (len(KEYPOINT_NAMES),):
            raise ValueError("confidences must have shape (17,)")
        if self.bounding_box.shape != (4,):
            raise ValueError("bounding_box must have shape (4,)")

    def keypoint(self, name: str) -> tuple[float, float, float]:
        """Return x, y, and confidence for a named COCO keypoint."""
        try:
            index = KEYPOINT_NAMES.index(name)
        except ValueError as exc:
            raise KeyError(f"Unknown keypoint: {name}") from exc
        x, y = self.keypoints[index]
        return float(x), float(y), float(self.confidences[index])

    @property
    def bounding_box_area(self) -> float:
        x1, y1, x2, y2 = self.bounding_box
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


@dataclass(frozen=True)
class PoseFrame:
    """All pose detections from a frame and the selected primary person."""

    detections: tuple[PoseDetection, ...]
    primary_index: int | None

    @property
    def primary(self) -> PoseDetection | None:
        if self.primary_index is None:
            return None
        return self.detections[self.primary_index]


def select_primary_person(detections: Sequence[PoseDetection]) -> int | None:
    """Select the most prominent person by box area, then confidence."""
    if not detections:
        return None
    return max(
        range(len(detections)),
        key=lambda index: (
            detections[index].bounding_box_area,
            detections[index].detection_confidence,
        ),
    )


class PoseTracker:
    """Load one YOLO pose model and infer structured poses from BGR frames."""

    def __init__(self, config: PoseConfig | None = None) -> None:
        self.config = config or PoseConfig()
        # Keep this import local so data-structure tests do not require the
        # heavyweight inference dependency to be installed.
        from ultralytics import YOLO

        self._model: YOLO = YOLO(self.config.model_path)

    def process(self, frame: npt.NDArray[np.uint8]) -> PoseFrame:
        """Run one inference pass and return copied numerical pose data."""
        predict_options: dict[str, Any] = {
            "source": frame,
            "conf": self.config.confidence_threshold,
            "imgsz": self.config.image_size,
            "verbose": False,
        }
        if self.config.device:
            predict_options["device"] = self.config.device

        results = self._model.predict(**predict_options)
        if not results:
            return PoseFrame((), None)

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return PoseFrame((), None)

        coordinates = result.keypoints.xy.detach().cpu().numpy()
        keypoint_confidences = result.keypoints.conf
        if keypoint_confidences is None:
            confidence_values = np.ones(coordinates.shape[:2], dtype=np.float32)
        else:
            confidence_values = keypoint_confidences.detach().cpu().numpy()
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        box_confidences = result.boxes.conf.detach().cpu().numpy()

        count = min(
            len(coordinates), len(confidence_values), len(boxes), len(box_confidences)
        )
        detections = tuple(
            PoseDetection(
                keypoints=np.asarray(coordinates[index], dtype=np.float32).copy(),
                confidences=np.asarray(
                    confidence_values[index], dtype=np.float32
                ).copy(),
                bounding_box=np.asarray(boxes[index], dtype=np.float32).copy(),
                detection_confidence=float(box_confidences[index]),
            )
            for index in range(count)
        )
        return PoseFrame(detections, select_primary_person(detections))

"""Optional RF-DETR tennis-ball inference backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from tennis_vision.config import RFDETRConfig
from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.ball import BallDetection


class RFDETRBallDetector:
    """Load a reviewed-data checkpoint and return raw source-space boxes.

    All threshold-passing candidates are returned. The existing single-ball
    tracker resolves multiples using confidence on acquisition and proximity to
    its unsmoothed motion prediction thereafter.
    """

    def __init__(self, config: RFDETRConfig | None = None) -> None:
        self.config = config or RFDETRConfig()
        checkpoint = Path(self.config.checkpoint_path).expanduser()
        if not checkpoint.is_file():
            raise RuntimeError(
                f"RF-DETR checkpoint not found: {checkpoint}. Train the reviewed "
                "dataset first or pass --rfdetr-checkpoint."
            )
        try:
            from rfdetr import from_checkpoint
        except ImportError as exc:
            raise RuntimeError(
                'RF-DETR is optional. Install the pinned experiment dependency with '
                '`pip install -r requirements-rfdetr.txt` in a separate environment.'
            ) from exc
        options: dict[str, Any] = {}
        if self.config.device:
            options["device"] = self.config.device
        self._model = from_checkpoint(checkpoint, **options)

    @property
    def name(self) -> str:
        return "rfdetr"

    def detect(
        self, frame: npt.NDArray[np.uint8], metadata: FrameMetadata
    ) -> tuple[BallDetection, ...]:
        """Run RGB inference and expose RF-DETR's original-image coordinates."""
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        result = self._model.predict(
            rgb,
            threshold=self.config.confidence_threshold,
            shape=(self.config.input_size, self.config.input_size),
            include_source_image=False,
        )
        return detections_from_rfdetr(
            result,
            metadata,
            expected_class_name=self.config.class_name,
            minimum_confidence=self.config.confidence_threshold,
        )


def detections_from_rfdetr(
    result: Any,
    metadata: FrameMetadata,
    *,
    expected_class_name: str = "tennis ball",
    minimum_confidence: float = 0.0,
) -> tuple[BallDetection, ...]:
    """Convert a Supervision Detections-like result without scaling again."""
    boxes = np.asarray(getattr(result, "xyxy", np.empty((0, 4))), dtype=float)
    if boxes.size == 0:
        return ()
    boxes = boxes.reshape((-1, 4))
    raw_confidences = getattr(result, "confidence", None)
    confidences = (
        np.full(len(boxes), np.nan)
        if raw_confidences is None
        else np.asarray(raw_confidences, dtype=float)
    )
    class_ids = getattr(result, "class_id", None)
    class_ids_array = (
        np.zeros(len(boxes), dtype=int)
        if class_ids is None
        else np.asarray(class_ids, dtype=int)
    )
    data = getattr(result, "data", {}) or {}
    class_names = data.get("class_name") if isinstance(data, dict) else None
    names_array = None if class_names is None else np.asarray(class_names).astype(str)

    detections: list[BallDetection] = []
    for index, box in enumerate(boxes):
        # Fine-tuned RF-DETR maps the one leaf class to model class 0. Prefer
        # explicit names when present and reject any synthetic grouping class.
        if names_array is not None:
            if index >= len(names_array) or names_array[index] != expected_class_name:
                continue
        elif index >= len(class_ids_array) or class_ids_array[index] != 0:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if not np.all(np.isfinite((x1, y1, x2, y2))) or x2 <= x1 or y2 <= y1:
            continue
        confidence = None
        if index < len(confidences) and np.isfinite(confidences[index]):
            confidence = float(confidences[index])
        if (confidence is None and minimum_confidence > 0.0) or (
            confidence is not None and confidence < minimum_confidence
        ):
            continue
        detections.append(
            BallDetection(
                center_x=(x1 + x2) / 2.0,
                center_y=(y1 + y2) / 2.0,
                bounding_box=(x1, y1, x2, y2),
                confidence=confidence,
                frame_index=metadata.frame_index,
                timestamp_seconds=metadata.timestamp_seconds,
                detector_name="rfdetr",
            )
        )
    return tuple(
        sorted(
            detections,
            key=lambda item: 1.0 if item.confidence is None else -item.confidence,
        )
    )

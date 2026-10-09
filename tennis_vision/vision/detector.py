"""Common ball-detector backend contract."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from tennis_vision.frames import FrameMetadata
from tennis_vision.vision.ball import BallDetection


class BallDetectorBackend(Protocol):
    """Interface shared by independent-frame and temporal detectors."""

    @property
    def name(self) -> str: ...

    def detect(
        self, frame: npt.NDArray[np.uint8], metadata: FrameMetadata
    ) -> tuple[BallDetection, ...]: ...

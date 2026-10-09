"""Source-frame types that preserve temporal metadata through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class FrameMetadata:
    """Timing supplied with a decoded frame.

    ``timestamp_seconds`` is source time for files and elapsed capture time for
    cameras. ``timing_is_estimated`` is true when file metadata was unavailable.
    """

    frame_index: int
    timestamp_seconds: float
    source_fps: float | None
    timing_is_estimated: bool


@dataclass(frozen=True)
class SourceFrame:
    image: npt.NDArray[np.uint8]
    metadata: FrameMetadata

"""Offline tennis-ball dataset generation utilities."""

from tennis_vision.dataset.ball import (
    CandidateFrame,
    DatasetRecord,
    SplitSegment,
    VideoMetadata,
    build_candidate_box,
    build_coco_document,
    build_split_segments,
    select_diverse_frames,
)

__all__ = [
    "CandidateFrame",
    "DatasetRecord",
    "SplitSegment",
    "VideoMetadata",
    "build_candidate_box",
    "build_coco_document",
    "build_split_segments",
    "select_diverse_frames",
]

"""OpenCV rendering for structured pose output."""

from __future__ import annotations

import cv2
import numpy.typing as npt
import numpy as np

from tennis_vision.config import VisualizationConfig
from tennis_vision.tracking.ball import BallTrack, TrackState
from tennis_vision.vision.pose import PoseDetection

# COCO skeleton indices. Face-only connections are intentionally omitted to
# keep the webcam overlay focused on body mechanics.
SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
BODY_KEYPOINT_INDICES: tuple[int, ...] = tuple(range(5, 17))


def draw_pose(
    frame: npt.NDArray[np.uint8],
    pose: PoseDetection | None,
    config: VisualizationConfig,
) -> None:
    """Draw a primary pose in place, skipping low-confidence geometry."""
    if pose is None:
        return

    visible = pose.confidences >= config.keypoint_confidence_threshold
    points = np.rint(pose.keypoints).astype(int)
    for start, end in SKELETON_CONNECTIONS:
        if visible[start] and visible[end]:
            cv2.line(
                frame,
                tuple(points[start]),
                tuple(points[end]),
                config.skeleton_color,
                config.connection_thickness,
                cv2.LINE_AA,
            )
    for index in BODY_KEYPOINT_INDICES:
        if visible[index]:
            cv2.circle(
                frame,
                tuple(points[index]),
                config.joint_radius,
                config.joint_color,
                -1,
                cv2.LINE_AA,
            )


def draw_fps(
    frame: npt.NDArray[np.uint8], fps: float, config: VisualizationConfig
) -> None:
    """Draw a legible processing-throughput counter in place."""
    cv2.putText(
        frame,
        f"Processing FPS: {fps:.1f}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        config.fps_color,
        2,
        cv2.LINE_AA,
    )


def draw_ball_track(
    frame: npt.NDArray[np.uint8],
    track: BallTrack,
    config: VisualizationConfig,
) -> None:
    """Draw the current ball estimate, short trail, and compact state label."""
    if track.state is TrackState.LOST or track.position is None:
        return

    color = (
        config.ball_detected_color
        if track.state is TrackState.DETECTED
        else config.ball_predicted_color
    )
    points = [
        (int(round(point.x)), int(round(point.y))) for point in track.history
    ]
    for start, end in zip(points, points[1:], strict=False):
        cv2.line(
            frame,
            start,
            end,
            color,
            config.trail_thickness,
            cv2.LINE_AA,
        )

    center = (int(round(track.position[0])), int(round(track.position[1])))
    if track.detection is not None:
        x1, y1, x2, y2 = track.detection.bounding_box
        cv2.rectangle(
            frame,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            1,
            cv2.LINE_AA,
        )
        if track.detection.confidence is not None:
            cv2.putText(
                frame,
                f"{track.detection.confidence:.2f}",
                (int(round(x1)), max(12, int(round(y1)) - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
    cv2.circle(frame, center, config.ball_radius, color, 2, cv2.LINE_AA)
    cv2.circle(frame, center, 2, color, -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"BALL: {track.state.value}",
        (16, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_ball_model(
    frame: npt.NDArray[np.uint8], model_name: str, config: VisualizationConfig
) -> None:
    """Show the active ball backend without adding detector debug output."""
    cv2.putText(
        frame,
        f"BALL MODEL: {model_name.upper()}",
        (16, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        config.fps_color,
        1,
        cv2.LINE_AA,
    )

"""Configuration for the Tennis Vision application."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720


@dataclass(frozen=True)
class PoseConfig:
    model_path: str = "yolo11n-pose.pt"
    confidence_threshold: float = 0.35
    image_size: int = 640
    device: str | None = None


@dataclass(frozen=True)
class BallDetectorConfig:
    """Configuration for the provisional COCO sports-ball detector."""

    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.15
    image_size: int = 640
    device: str | None = None
    sports_ball_class_id: int = 32


@dataclass(frozen=True)
class TrackNetConfig:
    """Configuration for the experimental temporal tennis detector."""

    checkpoint_path: str = "models/tracknet/tracknet_v1_tennis.pth"
    input_width: int = 640
    input_height: int = 360
    temporal_frames: int = 3
    heatmap_threshold: int = 127
    device: str | None = None


@dataclass(frozen=True)
class RFDETRConfig:
    """Configuration for a fine-tuned, one-class RF-DETR checkpoint."""

    checkpoint_path: str = (
        "runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth"
    )
    confidence_threshold: float = 0.25
    input_size: int = 1024
    class_name: str = "tennis ball"
    device: str | None = None


@dataclass(frozen=True)
class BallTrackerConfig:
    """Motion-gating and missed-observation settings for a single ball."""

    maximum_missed_detections: int = 4
    maximum_history: int = 24
    base_jump_tolerance_pixels: float = 35.0
    maximum_speed_pixels_per_second: float = 2200.0
    velocity_smoothing: float = 0.5


@dataclass(frozen=True)
class VisualizationConfig:
    keypoint_confidence_threshold: float = 0.35
    joint_radius: int = 4
    connection_thickness: int = 2
    skeleton_color: tuple[int, int, int] = (60, 220, 60)
    joint_color: tuple[int, int, int] = (40, 180, 255)
    fps_color: tuple[int, int, int] = (255, 255, 255)
    ball_detected_color: tuple[int, int, int] = (0, 255, 255)
    ball_predicted_color: tuple[int, int, int] = (0, 165, 255)
    ball_radius: int = 10
    trail_thickness: int = 2

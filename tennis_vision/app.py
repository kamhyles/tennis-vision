"""Camera/video orchestration for pose and tennis-ball processing."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import cv2

from tennis_vision.config import (
    BallDetectorConfig,
    BallTrackerConfig,
    CameraConfig,
    PoseConfig,
    RFDETRConfig,
    TrackNetConfig,
    VisualizationConfig,
)
from tennis_vision.diagnostics import BallRunDiagnostics, DetectionCsvWriter
from tennis_vision.media import AnnotatedVideoWriter, open_source
from tennis_vision.performance import PerformanceMonitor
from tennis_vision.tracking.ball import BallTracker
from tennis_vision.vision.ball import YOLOBallDetector
from tennis_vision.vision.detector import BallDetectorBackend
from tennis_vision.vision.pose import PoseFrame, PoseTracker
from tennis_vision.vision.rfdetr import RFDETRBallDetector
from tennis_vision.vision.tracknet import TrackNetBallDetector
from tennis_vision.visualization import (
    draw_ball_model,
    draw_ball_track,
    draw_fps,
    draw_pose,
)

WINDOW_NAME = "Tennis Vision"


class FpsMeter:
    """Exponentially smoothed measurement of completed frames per second."""

    def __init__(self, smoothing: float = 0.15) -> None:
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        self.smoothing = smoothing
        self._previous_time: float | None = None
        self.fps = 0.0

    def tick(self, now: float | None = None) -> float:
        current_time = time.perf_counter() if now is None else now
        if self._previous_time is not None:
            elapsed = current_time - self._previous_time
            if elapsed > 0.0:
                instantaneous = 1.0 / elapsed
                self.fps = (
                    instantaneous
                    if self.fps == 0.0
                    else self.smoothing * instantaneous
                    + (1.0 - self.smoothing) * self.fps
                )
        self._previous_time = current_time
        return self.fps


def _positive_interval(value: str) -> int:
    interval = int(value)
    if interval < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return interval


def _positive_float(value: str) -> float:
    number = float(value)
    if not 0.0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def run(
    camera_config: CameraConfig,
    pose_config: PoseConfig,
    ball_detector_config: BallDetectorConfig,
    ball_tracker_config: BallTrackerConfig,
    *,
    ball_detector_kind: str = "yolo",
    tracknet_config: TrackNetConfig | None = None,
    rfdetr_config: RFDETRConfig | None = None,
    video_path: str | None = None,
    pose_every: int = 1,
    ball_every: int = 1,
    performance_log_interval: int = 120,
    detections_output: str | None = None,
    maximum_frames: int | None = None,
    output_video: str | None = None,
    output_fps_limit: float = 60.0,
    show_preview: bool = True,
) -> None:
    """Process webcam or video frames until q or clean video EOF."""
    pose_tracker = PoseTracker(pose_config)
    ball_detector: BallDetectorBackend
    if ball_detector_kind == "tracknet":
        if ball_every != 1:
            raise ValueError(
                "TrackNet requires --ball-every 1 so its buffer receives "
                "consecutive source frames"
            )
        ball_detector = TrackNetBallDetector(tracknet_config)
    elif ball_detector_kind == "yolo":
        ball_detector = YOLOBallDetector(ball_detector_config)
    elif ball_detector_kind == "rfdetr":
        ball_detector = RFDETRBallDetector(rfdetr_config)
    else:
        raise ValueError(f"Unknown ball detector: {ball_detector_kind}")
    ball_tracker = BallTracker(ball_tracker_config)
    source = open_source(camera_config, video_path)
    csv_writer: DetectionCsvWriter | None = None
    diagnostics: BallRunDiagnostics | None = None
    performance: PerformanceMonitor | None = None
    fps_meter: FpsMeter | None = None
    video_writer: AnnotatedVideoWriter | None = None
    completed_normally = False
    writer_error: RuntimeError | OSError | None = None
    try:
        visualization_config = VisualizationConfig()
        fps_meter = FpsMeter()
        performance = PerformanceMonitor(performance_log_interval)
        diagnostics = BallRunDiagnostics()
        csv_writer = DetectionCsvWriter(detections_output)
        pose_frame = PoseFrame((), None)

        source_fps_text = (
            "unknown" if source.source_fps is None else f"{source.source_fps:.3f}"
        )
        print(f"Input: {source.source_name}; source FPS: {source_fps_text}")

        while True:
            source_frame = source.read()
            if source_frame is None:
                if source.is_video:
                    print("End of video.")
                    break
                raise RuntimeError("The camera stopped returning frames.")

            frame = source_frame.image
            metadata = source_frame.metadata

            if metadata.frame_index % pose_every == 0:
                started = time.perf_counter()
                pose_frame = pose_tracker.process(frame)
                performance.record_pose(time.perf_counter() - started)

            detection_attempted = metadata.frame_index % ball_every == 0
            if detection_attempted:
                started = time.perf_counter()
                ball_detections = ball_detector.detect(frame, metadata)
                performance.record_ball(time.perf_counter() - started)
            else:
                ball_detections = ()
            ball_track = ball_tracker.update(
                ball_detections,
                metadata,
                detection_attempted=detection_attempted,
            )

            diagnostics.observe(
                ball_track, had_direct_localization=bool(ball_detections)
            )
            csv_writer.write(
                metadata,
                ball_detector.name,
                ball_track,
                ball_track.detection,
            )

            draw_pose(frame, pose_frame.primary, visualization_config)
            draw_ball_track(frame, ball_track, visualization_config)
            draw_ball_model(frame, ball_detector.name, visualization_config)
            pipeline_fps = fps_meter.tick()
            draw_fps(frame, pipeline_fps, visualization_config)
            performance.maybe_log(metadata.frame_index, pipeline_fps)

            if output_video is not None:
                if video_writer is None:
                    source_height, source_width = frame.shape[:2]
                    export_fps = min(
                        source.source_fps or output_fps_limit, output_fps_limit
                    )
                    video_writer = AnnotatedVideoWriter(
                        output_video,
                        width=source_width,
                        height=source_height,
                        output_fps=export_fps,
                    )
                    print(
                        f"Export: {output_video}; {export_fps:.3f} FPS H.264 "
                        "MP4 (video only)"
                    )
                video_writer.write(frame, metadata.timestamp_seconds)

            if show_preview:
                cv2.imshow(WINDOW_NAME, frame)

            if show_preview and cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if (
                maximum_frames is not None
                and metadata.frame_index + 1 >= maximum_frames
            ):
                print(f"Stopped after {maximum_frames} frames.")
                break
        completed_normally = True
    finally:
        if video_writer is not None:
            if completed_normally:
                try:
                    video_writer.close()
                except (RuntimeError, OSError) as exc:
                    writer_error = exc
            else:
                video_writer.abort()
        if csv_writer is not None:
            csv_writer.close()
        source.release()
        if show_preview:
            cv2.destroyAllWindows()
        if diagnostics is not None:
            print(
                f"Ball diagnostics ({ball_detector.name}): "
                f"{diagnostics.summary()}"
            )
        if performance is not None and fps_meter is not None:
            print(
                f"Performance: pipeline={fps_meter.fps:.1f} FPS; "
                f"{performance.summary()}"
            )
        if writer_error is not None:
            raise writer_error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pose tracking and provisional tennis-ball tracking"
    )
    parser.add_argument("--video", help="video file instead of webcam input")
    parser.add_argument("--camera", type=int, default=0, help="camera device index")
    parser.add_argument("--width", type=int, default=1280, help="capture width")
    parser.add_argument("--height", type=int, default=720, help="capture height")
    parser.add_argument(
        "--pose-model", "--model", dest="pose_model", default="yolo11n-pose.pt"
    )
    parser.add_argument("--pose-confidence", "--confidence", type=float, default=0.35)
    parser.add_argument("--pose-image-size", "--image-size", type=int, default=640)
    parser.add_argument("--ball-model", default="yolo11n.pt")
    parser.add_argument(
        "--ball-detector",
        choices=("yolo", "tracknet", "rfdetr"),
        default="yolo",
        help="interchangeable ball-localization backend",
    )
    parser.add_argument("--ball-confidence", type=float, default=0.15)
    parser.add_argument("--ball-image-size", type=int, default=640)
    parser.add_argument(
        "--tracknet-checkpoint",
        default="models/tracknet/tracknet_v1_tennis.pth",
        help="tennis TrackNet state-dict checkpoint",
    )
    parser.add_argument(
        "--rfdetr-checkpoint",
        default="runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth",
        help="fine-tuned RF-DETR checkpoint",
    )
    parser.add_argument(
        "--rfdetr-input-size",
        type=int,
        default=1024,
        help="square RF-DETR inference resolution",
    )
    parser.add_argument(
        "--device", default=None, help="inference device, e.g. cpu, mps, or cuda:0"
    )
    parser.add_argument(
        "--pose-every",
        type=_positive_interval,
        default=1,
        help="run pose inference every N source frames",
    )
    parser.add_argument(
        "--ball-every",
        type=_positive_interval,
        default=1,
        help="run ball detection every N source frames",
    )
    parser.add_argument(
        "--performance-log-interval",
        type=int,
        default=120,
        help="print stage timing every N frames; 0 disables logging",
    )
    parser.add_argument(
        "--detections-output",
        help="optional per-frame detector/tracker CSV output path",
    )
    parser.add_argument(
        "--max-frames",
        type=_positive_interval,
        default=None,
        help="stop after N source frames (useful for experiments)",
    )
    parser.add_argument(
        "--output-video",
        help="write timestamp-preserving annotated H.264 video to an .mp4 file",
    )
    parser.add_argument(
        "--output-fps",
        type=_positive_float,
        default=60.0,
        help="maximum export FPS (default: 60; source timing is preserved)",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="process without opening the live OpenCV preview window",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.performance_log_interval < 0:
        print("Error: --performance-log-interval cannot be negative")
        return 2
    if args.output_video is not None and args.video is None:
        print("Error: --output-video currently requires --video")
        return 2
    try:
        run(
            CameraConfig(index=args.camera, width=args.width, height=args.height),
            PoseConfig(
                model_path=args.pose_model,
                confidence_threshold=args.pose_confidence,
                image_size=args.pose_image_size,
                device=args.device,
            ),
            BallDetectorConfig(
                model_path=args.ball_model,
                confidence_threshold=args.ball_confidence,
                image_size=args.ball_image_size,
                device=args.device,
            ),
            BallTrackerConfig(),
            ball_detector_kind=args.ball_detector,
            tracknet_config=TrackNetConfig(
                checkpoint_path=args.tracknet_checkpoint,
                device=args.device,
            ),
            rfdetr_config=RFDETRConfig(
                checkpoint_path=args.rfdetr_checkpoint,
                confidence_threshold=args.ball_confidence,
                input_size=args.rfdetr_input_size,
                device=args.device,
            ),
            video_path=args.video,
            pose_every=args.pose_every,
            ball_every=args.ball_every,
            performance_log_interval=args.performance_log_interval,
            detections_output=args.detections_output,
            maximum_frames=args.max_frames,
            output_video=args.output_video,
            output_fps_limit=args.output_fps,
            show_preview=not args.no_preview,
        )
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0

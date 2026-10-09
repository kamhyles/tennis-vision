#!/usr/bin/env python3
"""Headless, same-video operational benchmark for ball backends."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tennis_vision.config import (  # noqa: E402
    BallDetectorConfig,
    BallTrackerConfig,
    CameraConfig,
    RFDETRConfig,
    TrackNetConfig,
)
from tennis_vision.diagnostics import BallRunDiagnostics  # noqa: E402
from tennis_vision.media import open_source  # noqa: E402
from tennis_vision.tracking.ball import BallTracker  # noqa: E402
from tennis_vision.vision.ball import YOLOBallDetector  # noqa: E402
from tennis_vision.vision.rfdetr import RFDETRBallDetector  # noqa: E402
from tennis_vision.vision.tracknet import TrackNetBallDetector  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ball-backend throughput and continuity")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--backend",
        action="append",
        choices=("yolo", "tracknet", "rfdetr"),
        required=True,
    )
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--rfdetr-checkpoint")
    parser.add_argument("--tracknet-checkpoint", default="models/tracknet/tracknet_v1_tennis.pth")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = []
    try:
        for backend in args.backend:
            detector = _make_detector(backend, args)
            tracker = BallTracker(BallTrackerConfig())
            diagnostics = BallRunDiagnostics()
            source = open_source(CameraConfig(), args.video)
            detector_seconds = 0.0
            wall_started = time.perf_counter()
            frames = 0
            try:
                while True:
                    source_frame = source.read()
                    if source_frame is None:
                        break
                    started = time.perf_counter()
                    detections = detector.detect(source_frame.image, source_frame.metadata)
                    detector_seconds += time.perf_counter() - started
                    track = tracker.update(detections, source_frame.metadata)
                    diagnostics.observe(track, had_direct_localization=bool(detections))
                    frames += 1
                    if args.max_frames is not None and frames >= args.max_frames:
                        break
            finally:
                source.release()
            wall_seconds = time.perf_counter() - wall_started
            reports.append(
                {
                    "backend": detector.name,
                    "frames": frames,
                    "detector_fps": None if detector_seconds == 0 else frames / detector_seconds,
                    "pipeline_fps_without_pose_or_display": None if wall_seconds == 0 else frames / wall_seconds,
                    "operational_diagnostics": diagnostics.summary(),
                    "accuracy_warning": "localization counts are not correctness metrics",
                }
            )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(reports, indent=2))
    return 0


def _make_detector(name: str, args: argparse.Namespace) -> object:
    if name == "yolo":
        return YOLOBallDetector(BallDetectorConfig(device=args.device))
    if name == "tracknet":
        return TrackNetBallDetector(
            TrackNetConfig(checkpoint_path=args.tracknet_checkpoint, device=args.device)
        )
    if not args.rfdetr_checkpoint:
        raise ValueError("--rfdetr-checkpoint is required for the rfdetr backend")
    return RFDETRBallDetector(
        RFDETRConfig(checkpoint_path=args.rfdetr_checkpoint, device=args.device)
    )


if __name__ == "__main__":
    raise SystemExit(main())

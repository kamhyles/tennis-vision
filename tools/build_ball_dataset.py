#!/usr/bin/env python3
"""Build an annotation-ready tennis-ball candidate dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vision.dataset.builder import (  # noqa: E402
    DatasetBuildConfig,
    build_dataset,
    describe_inspection,
    inspect_sources,
)
from tennis_vision.dataset.validation import validate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-resolution tennis-ball candidate images and "
            "unreviewed TrackNet COCO pre-annotations"
        )
    )
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-frames", type=int, default=300)
    parser.add_argument("--min-frame-gap", type=int, default=8)
    parser.add_argument("--min-time-gap", type=float, default=0.08)
    parser.add_argument(
        "--miss-review-fraction",
        type=float,
        default=0.10,
        help="fraction reserved for unconfirmed TrackNet-miss review frames",
    )
    parser.add_argument(
        "--boundary-guard-seconds",
        type=float,
        default=1.0,
        help="total excluded time centered on each temporal split boundary",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--spatial-columns", type=int, default=4)
    parser.add_argument("--spatial-rows", type=int, default=3)
    parser.add_argument("--box-scale", type=float, default=1.5)
    parser.add_argument("--min-box-size", type=float, default=12.0)
    parser.add_argument("--max-box-size", type=float, default=48.0)
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--tracknet-checkpoint",
        default="models/tracknet/tracknet_v1_tennis.pth",
    )
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="inspect source metadata and split boundaries without inference or writes",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.dry_run:
            inspected = inspect_sources(
                args.videos,
                boundary_guard_seconds=args.boundary_guard_seconds,
                train_fraction=args.train_fraction,
                validation_fraction=args.validation_fraction,
                test_fraction=args.test_fraction,
            )
            print(describe_inspection(inspected))
            print("Dry run only: no TrackNet inference or dataset files were created.")
            return 0

        config = DatasetBuildConfig(
            videos=args.videos,
            output=args.output,
            target_frames=args.target_frames,
            minimum_frame_gap=args.min_frame_gap,
            minimum_time_gap=args.min_time_gap,
            miss_review_fraction=args.miss_review_fraction,
            boundary_guard_seconds=args.boundary_guard_seconds,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
            spatial_columns=args.spatial_columns,
            spatial_rows=args.spatial_rows,
            box_scale=args.box_scale,
            minimum_box_size=args.min_box_size,
            maximum_box_size=args.max_box_size,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            seed=args.seed,
            checkpoint_path=args.tracknet_checkpoint,
            device=args.device,
        )
        summary = build_dataset(config)
        report = validate_dataset(args.output)
        print("Dataset summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        print(f"Sanity checks: {report.summary()}")
        for warning in report.warnings:
            print(f"  WARNING: {warning}")
        for error in report.errors:
            print(f"  ERROR: {error}")
        if not report.is_valid:
            return 1
        print(f"Review instructions: {args.output / 'REVIEW.md'}")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

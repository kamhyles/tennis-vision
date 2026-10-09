#!/usr/bin/env python3
"""Validate and temporally split the reviewed tennis-ball COCO export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tennis_vision.dataset.reviewed_coco import (  # noqa: E402
    inspect_reviewed_dataset,
    prepare_temporal_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a reviewed COCO export and create guarded temporal splits"
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--inspect-only", action="store_true", help="validate without writing a dataset"
    )
    parser.add_argument(
        "--skip-pixel-check",
        action="store_true",
        help="skip decoding every image to verify dimensions",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.inspect_only:
            report = inspect_reviewed_dataset(
                args.source, verify_pixels=not args.skip_pixel_check
            )
        else:
            if args.output is None:
                raise ValueError("--output is required unless --inspect-only is used")
            report = prepare_temporal_dataset(args.source, args.output)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2))
    if not report.is_valid:
        return 1
    print(
        "This dataset originates from one recording and is not sufficient evidence "
        "of cross-session generalization."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

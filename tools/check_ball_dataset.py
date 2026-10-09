#!/usr/bin/env python3
"""Run sanity checks against a generated candidate dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tennis_vision.dataset.validation import validate_dataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset)
    print(f"Dataset checks: {report.summary()}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

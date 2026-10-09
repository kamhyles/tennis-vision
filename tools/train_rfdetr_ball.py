#!/usr/bin/env python3
"""Run the pinned RF-DETR tennis-ball fine-tuning experiment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tennis_vision.dataset.reviewed_coco import (  # noqa: E402
    TENNIS_BALL_CLASS,
    inspect_reviewed_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RF-DETR on reviewed data")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rfdetr/tennis_ball_small_1024.json"),
    )
    parser.add_argument("--device", help="override config device, e.g. mps or cuda")
    parser.add_argument("--resume", type=Path, help="full Lightning checkpoint to resume")
    parser.add_argument(
        "--check-only", action="store_true", help="validate config/data/device only"
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"package_version", "model", "resolution", "dataset_dir", "output_dir", "training"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    if config["model"] != "RFDETRSmall":
        raise ValueError("This experiment script is pinned to RFDETRSmall")
    return config


def validate_prepared_dataset(dataset_dir: Path) -> None:
    expected = {"train": 175, "valid": 27, "test": 33}
    for split, expected_count in expected.items():
        report = inspect_reviewed_dataset(dataset_dir / split, verify_pixels=False)
        if not report.is_valid:
            raise ValueError(f"Invalid {split} split: {'; '.join(report.errors)}")
        if report.image_count != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split} images, found {report.image_count}"
            )
        if report.annotated_category_names != (TENNIS_BALL_CLASS,):
            raise ValueError(f"Unexpected annotated classes in {split}")


def validate_device(device: str) -> str:
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable in this process. Run locally where "
            "torch.backends.mps.is_available() is true, or use --device cuda on a GPU host."
        )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this process")
    if device == "cpu":
        raise RuntimeError(
            "The 1024px RF-DETR training experiment is intentionally not launched on CPU; "
            "use an available MPS or CUDA accelerator."
        )
    return device


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
        installed = importlib.metadata.version("rfdetr")
        if installed != config["package_version"]:
            raise RuntimeError(
                f"Expected rfdetr=={config['package_version']}, found {installed}"
            )
        dataset_dir = (PROJECT_ROOT / config["dataset_dir"]).resolve()
        validate_prepared_dataset(dataset_dir)
        training = dict(config["training"])
        device = validate_device(args.device or training.pop("device"))
        output_dir = (PROJECT_ROOT / config["output_dir"]).resolve()
        if args.check_only:
            print(
                f"Ready: rfdetr={installed}, device={device}, dataset={dataset_dir}, "
                f"output={output_dir}"
            )
            return 0

        from rfdetr import RFDETRSmall

        model = RFDETRSmall(device=device)
        training.update(
            {
                "dataset_dir": str(dataset_dir),
                "output_dir": str(output_dir),
                "resolution": int(config["resolution"]),
                "device": device,
                "notes": config.get("notes"),
            }
        )
        if args.resume:
            training["resume"] = str(args.resume.resolve())
        model.train(**training)
    except (FileNotFoundError, importlib.metadata.PackageNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

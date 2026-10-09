#!/usr/bin/env python3
"""Evaluate RF-DETR with COCO metrics and tennis-ball center error."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tennis_vision.evaluation.ball import (  # noqa: E402
    CenterLocalizationMetrics,
    coco_bbox_to_xyxy,
)
from tennis_vision.frames import FrameMetadata  # noqa: E402
from tennis_vision.vision.rfdetr import detections_from_rfdetr  # noqa: E402

PINNED_RFDETR_VERSION = "1.10.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned RF-DETR checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/tennis_ball_reviewed_temporal"),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-coco",
        action="store_true",
        help="skip standard RF-DETR COCO evaluation and run center metrics only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        version = importlib.metadata.version("rfdetr")
        if version != PINNED_RFDETR_VERSION:
            raise RuntimeError(
                f"Expected rfdetr=={PINNED_RFDETR_VERSION}, found {version}"
            )
        from rfdetr import from_checkpoint

        model = from_checkpoint(args.checkpoint, device=args.device)
        dataset = args.dataset.resolve()
        coco_metrics = {}
        if not args.skip_coco:
            coco_metrics = model.evaluate(
                split="test",
                dataset_dir=str(dataset),
                device=args.device,
                resolution=args.resolution,
                batch_size=1,
                eval_batch_size=1,
                num_workers=0,
                output_dir=str(PROJECT_ROOT / "runs/rfdetr/evaluation"),
            )
        center_metrics, timing = evaluate_centers(
            model,
            dataset / "test",
            threshold=args.threshold,
            resolution=args.resolution,
        )
        report = {
            "checkpoint": str(args.checkpoint.resolve()),
            "rfdetr_version": version,
            "dataset": str(dataset),
            "split": "test",
            "confidence_threshold": args.threshold,
            "resolution": args.resolution,
            "coco": coco_metrics,
            "center_localization": center_metrics,
            "timing": timing,
            "interpretation_limit": (
                "This dataset originates from one recording and is not sufficient "
                "evidence of cross-session generalization."
            ),
        }
        encoded = json.dumps(report, indent=2) + "\n"
        print(encoded, end="")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
    except (FileNotFoundError, importlib.metadata.PackageNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


def evaluate_centers(
    model: object,
    split_dir: Path,
    *,
    threshold: float,
    resolution: int,
) -> tuple[dict[str, float | int | None], dict[str, float | int | None]]:
    document = json.loads(
        (split_dir / "_annotations.coco.json").read_text(encoding="utf-8")
    )
    annotations_by_image: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in document["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(
            coco_bbox_to_xyxy(annotation["bbox"])
        )
    metrics = CenterLocalizationMetrics()
    latencies: list[float] = []
    for offset, image_record in enumerate(document["images"]):
        pixels = cv2.imread(str(split_dir / image_record["file_name"]))
        if pixels is None:
            raise OSError(f"Could not read {image_record['file_name']}")
        rgb = np.ascontiguousarray(pixels[:, :, ::-1])
        started = time.perf_counter()
        result = model.predict(  # type: ignore[attr-defined]
            rgb,
            threshold=threshold,
            shape=(resolution, resolution),
            include_source_image=False,
        )
        latencies.append(time.perf_counter() - started)
        predictions = detections_from_rfdetr(
            result,
            FrameMetadata(offset, float(offset), None, True),
        )
        metrics.add_image(
            annotations_by_image[int(image_record["id"])],
            predictions,
            image_width=int(image_record["width"]),
            image_height=int(image_record["height"]),
        )
    latency_array = np.asarray(latencies, dtype=float)
    timing = {
        "images": len(latencies),
        "mean_inference_milliseconds": (
            None if not latencies else float(np.mean(latency_array) * 1000.0)
        ),
        "median_inference_milliseconds": (
            None if not latencies else float(np.median(latency_array) * 1000.0)
        ),
        "inference_frames_per_second": (
            None if not latencies else float(1.0 / np.mean(latency_array))
        ),
    }
    return metrics.summary(), timing


if __name__ == "__main__":
    raise SystemExit(main())

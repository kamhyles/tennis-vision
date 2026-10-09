# V0.3B RF-DETR experiment

## Reviewed data audit

Authoritative export: `Tennis Ball Detection.coco/`. Roboflow exported all 235
images into `train/` with one `_annotations.coco.json`; no Roboflow validation or
test directories were present. The annotations came from Astra Boxes and were
then human-reviewed. TrackNet candidate boxes are not ground truth.

The audit found:

- 235 images, all readable at 1920×1080
- 234 annotations across 234 images
- one reviewed image with zero annotations and no images with multiple annotations
- categories `0: Tennis-Ball-Detection` (Roboflow grouping parent) and
  `1: tennis ball`; every annotation uses category 1
- no missing files, invalid references, zero-area boxes, or out-of-bounds boxes
- source frames 2 through 3279 encoded in filenames

The exact authoritative leaf class is `tennis ball` (with a space). The grouping
category is retained in copied COCO files, while RF-DETR 1.10.1 filters Roboflow
hierarchical grouping categories from its model classes.

### Reviewed box distribution

| Statistic | Width (px) | Height (px) | Area (px²) | Frame area (%) |
|---|---:|---:|---:|---:|
| min | 7.00 | 5.00 | 42.00 | 0.00203 |
| P10 | 8.00 | 8.00 | 64.00 | 0.00309 |
| P25 | 9.00 | 10.00 | 90.00 | 0.00434 |
| P50 | 12.00 | 12.00 | 143.00 | 0.00690 |
| P75 | 14.75 | 15.00 | 224.75 | 0.01084 |
| P90 | 18.70 | 18.00 | 307.40 | 0.01482 |
| max | 40.00 | 32.00 | 1280.00 | 0.06173 |
| mean | 12.74 | 12.79 | 180.26 | 0.00869 |

## Temporal split

The original V0.3A source-timeline boundaries were reconstructed from each
filename's `f########` token. Images were never shuffled or randomly assigned.

| Split | Half-open source-frame range | Images |
|---|---:|---:|
| train | [0, 2247) | 175 |
| guard (excluded) | [2247, 2367) | 0 |
| valid | [2367, 2742) | 27 |
| guard (excluded) | [2742, 2862) | 0 |
| test | [2862, 3296) | 33 |

Each guard is 120 source frames, approximately one second at the nominal source
rate. The prepared RF-DETR layout is
`datasets/tennis_ball_reviewed_temporal/{train,valid,test}`. Each split contains
its images and `_annotations.coco.json`; `dataset_report.json` records provenance,
ranges, counts, and statistics. Rebuild it only into a new/nonexistent directory:

```bash
.venv/bin/python tools/prepare_reviewed_ball_dataset.py \
  --source "Tennis Ball Detection.coco" \
  --output datasets/tennis_ball_reviewed_temporal
```

This dataset originates from one recording and is not sufficient evidence of
cross-session generalization.

## RF-DETR selection and resolution

The experiment is pinned to official `rfdetr[train]==1.10.1`. The current core
detection variants are Nano (384), Small (512), Medium (576), and Large (704).
Small is the baseline: it is modest enough for a 235-image proof of concept and
eventual latency testing, but has more decoder capacity and input detail than
Nano. The official `rf-detr-small.pth` pretrained weights are used; the model is
not trained from scratch.

Training and inference use 1024×1024, a supported custom size divisible by
Small's `patch_size × num_windows = 32`. RF-DETR directly square-resizes these
16:9 frames, so x and y scales differ. At candidate resolutions, a median 12×12
source box becomes approximately:

| Square input | Median width | Median height |
|---:|---:|---:|
| 512 | 3.2 px | 5.7 px |
| 704 | 4.4 px | 7.8 px |
| 1024 | 6.4 px | 11.4 px |

The higher baseline resolution trades speed and memory for materially more
horizontal pixels on the target. RF-DETR's prediction API returns boxes mapped
to the original image size; the application does not rescale them a second time.

Optional augmentation is intentionally restrained for this first run:
multi-scale training, expanded scales, and scale jitter are off, and an empty
torchvision augmentation config avoids destructive random crops/flips. The
required resize/normalization remains. Later photometric augmentation should be
an explicit follow-up experiment, not mixed into this baseline.

## Reproducible training

RF-DETR is isolated from the working YOLO/TrackNet environment:

The inspected project environment uses Python 3.12.4, PyTorch 2.14.0,
torchvision 0.29.0, NumPy 2.5.3, OpenCV 4.14, and Ultralytics 8.4.155. RF-DETR's
core PyTorch/torchvision minimums do not directly conflict with those versions,
but `rfdetr`, Transformers 5.x, Pydantic 2.x, Supervision 0.29+, Lightning 2.6,
and TorchMetrics training dependencies are absent. No package was installed,
upgraded, or downgraded in the working environment. A separate environment keeps
those additions from destabilizing the validated pose, YOLO, and TrackNet stack.

```bash
python3 -m venv .venv-rfdetr
.venv-rfdetr/bin/pip install -r requirements-rfdetr.txt
.venv-rfdetr/bin/python tools/train_rfdetr_ball.py --check-only
.venv-rfdetr/bin/python tools/train_rfdetr_ball.py
```

The checked-in config uses RF-DETR Small, 1024 input, 50 maximum epochs,
micro-batch 1, four-step gradient accumulation, AdamW, learning rate `1e-4`,
encoder learning rate `1e-5`, EMA, seed 7, and validation early stopping with
patience 10. It writes ignored artifacts to
`runs/rfdetr/tennis_ball_small_1024/`; the expected best model is
`checkpoint_best_total.pth`.

Official 1.10.1 code maps `device="mps"` into PyTorch Lightning's MPS
accelerator. CPU/MPS require an explicit integer batch size rather than CUDA's
auto-batch probe. Historical RF-DETR releases had MPS unsupported-operation
reports, so `--check-only` and a short real run are still required on the actual
Mac. The current Codex process reports PyTorch built with MPS but MPS unavailable,
so no training was launched here. If the local process also reports MPS
unavailable or 1024 training is impractical, the clean alternative is the same
script with `--device cuda` on an NVIDIA GPU host; inference may still run on the
Mac afterward.

## Evaluation and application use

After training:

```bash
.venv-rfdetr/bin/python tools/evaluate_rfdetr_ball.py \
  --checkpoint runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth \
  --device mps \
  --output runs/rfdetr/tennis_ball_small_1024/test_metrics.json

.venv-rfdetr/bin/python main.py \
  --video videos/IMG_8356.MOV \
  --ball-detector rfdetr \
  --rfdetr-checkpoint runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth \
  --rfdetr-input-size 1024 \
  --device mps
```

Evaluation reports RF-DETR's COCO metrics plus raw center error in pixels and as
a fraction of image diagonal, recall, misses, false positives, extra detections,
IoU, and inference timing. These same-recording results are diagnostic only.

The RF-DETR adapter returns every threshold-passing `tennis ball` candidate in
original coordinates. The existing tracker chooses highest confidence only when
acquiring a track; once active, it chooses the nearest candidate inside its
time-scaled motion gate. Direct RF-DETR centers are not averaged or smoothed.
Visualization shows the selected raw box, confidence, center, trail, and backend
label while retaining pose output.

For same-footage operational comparison (not accuracy):

```bash
.venv-rfdetr/bin/python tools/benchmark_ball_backends.py \
  --video videos/IMG_8356.MOV \
  --backend yolo --backend tracknet --backend rfdetr \
  --rfdetr-checkpoint runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth \
  --device mps --max-frames 300
```

No trained checkpoint exists yet, so no COCO metrics, center-error results,
RF-DETR inference timing, short-video inference, or visual comparison are claimed.

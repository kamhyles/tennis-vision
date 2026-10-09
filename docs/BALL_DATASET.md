# Tennis-ball candidate dataset workflow

This workflow creates a review queue, not training-ready ground truth. Every
TrackNet box must be checked by a human before detector training.

## Inspect the source and split plan

```bash
python tools/build_ball_dataset.py \
  --videos videos/ \
  --output datasets/tennis_ball_v1 \
  --target-frames 300 \
  --dry-run
```

Each video is divided into contiguous train, validation, and test regions before
sampling. A one-second excluded guard band surrounds each split boundary by
default. Frames never move between split groups after sampling. With only one
short source video, these splits are useful for workflow validation but are not
a strong independent test of generalization to new sessions, cameras, courts, or
lighting.

## Generate candidates

```bash
python tools/build_ball_dataset.py \
  --videos videos/ \
  --output datasets/tennis_ball_v1 \
  --target-frames 300 \
  --device mps
```

The generator decodes every source frame in order because TrackNet needs three
consecutive frames. It does not drop frames or modify/copy source videos.
Selection is then balanced across source videos, spatial grid cells, a rough
frame-to-frame displacement proxy, and confidence bands when confidence exists.
The current TrackNet model has no calibrated confidence, so its confidence band
is recorded as unavailable. Minimum frame and timestamp gaps are enforced.

`--target-frames` is a ceiling. The generator will return fewer examples instead
of violating spacing or filling the dataset with near duplicates. TrackNet-miss
frames can occupy a configurable fraction of the review queue, but remain
unlabeled and unconfirmed; a miss is never declared a true negative automatically.

Candidate box size starts from the source-space diameter of TrackNet's decoded
Hough circle, multiplies it by `--box-scale` (default 1.5), clamps the square to
`--min-box-size` and `--max-box-size` (12–48 pixels by default), and clips it to
the source image. This is only an initialization aid, not an accurate object
boundary.

## Output layout

```text
datasets/tennis_ball_v1/
  raw/                         # intentionally empty; source videos stay in videos/
  candidates/
    images/                    # full-resolution extracted frames
    previews/                  # box/center overlays for quick inspection
    labels/
      instances_candidates.json
      instances_train.json
      instances_validation.json
      instances_test.json
  reviewed/
    images/
    labels/
  metadata/
    source_videos.json
    build_config.json
    candidates.jsonl
    summary.json
  splits/
    train.txt
    validation.txt
    test.txt
  REVIEW.md
```

The COCO category is exactly `tennis_ball` with category ID 1. Images selected
from TrackNet misses appear in COCO's image list without annotations so a reviewer
can find a missed ball or confirm a true negative.

## Human review

Inspect `candidates/previews/`, then import `candidates/images/` with
`candidates/labels/instances_candidates.json` into a COCO-compatible annotation
tool such as CVAT. Approve or correct each box, remove false labels, annotate
missed visible balls, and explicitly confirm ball-free images. Export reviewed
COCO annotations under `reviewed/labels/` and retain the corresponding images or
a documented reference to `candidates/images/`.

Run the sanity checks again after generation or metadata changes:

```bash
python tools/check_ball_dataset.py datasets/tennis_ball_v1
```

The checker reports duplicate frames, near-adjacent selections, split/group
leakage, missing files or annotations, invalid class IDs, invalid/out-of-bounds
boxes, and suspicious box sizes.

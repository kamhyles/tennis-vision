# Tennis Vision

V0.3B preserves the validated human-pose system, provisional YOLO ball baseline,
and experimental temporal TrackNet backend while preparing a reviewed-data
RF-DETR fine-tuning experiment. RF-DETR training has not yet run. The project does
not calculate ball speed or perform racket, court, stroke, biomechanics, or
coaching analysis.

## Setup on macOS

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
python main.py
```

The first run downloads `yolo11n-pose.pt` for pose and `yolo11n.pt` for generic
COCO sports-ball detection. Allow camera access if macOS asks. Press `q` while
the video window is focused to quit.

Download and checksum-verify the experimental tennis TrackNet checkpoint once:

```bash
python scripts/download_tracknet_weights.py
```

Useful optional settings:

```bash
python main.py --camera 1 --width 1280 --height 720
python main.py --device mps
python main.py --video path/to/video.mp4
python main.py --video videos/IMG_8356.MOV --ball-detector tracknet --device mps
python main.py --help
```

If Metal inference is unsupported or unstable in the installed PyTorch build,
omit `--device mps` to let Ultralytics use its default device.

Pose and ball models can run at independent source-frame intervals for high-FPS
input. Both default to every frame:

```bash
python main.py --video path/to/120fps-video.mp4 --pose-every 4 --ball-every 1
```

TrackNet must receive every source frame, so it requires `--ball-every 1`. Pose
can still run less often. Offline analysis intentionally keeps all source frames:

```bash
python main.py \
  --video videos/IMG_8356.MOV \
  --ball-detector tracknet \
  --device mps \
  --pose-every 8 \
  --detections-output outputs/IMG_8356-tracknet.csv
```

For a short experimental run, add `--max-frames 120`. The CSV records source
frame index and timestamp, detector validity, source-space coordinates,
confidence when meaningful, and downstream tracking state. End-of-run statistics
are operational continuity diagnostics, not accuracy metrics.

The terminal prints source FPS at startup and periodically reports total pipeline
FPS plus average pose and ball inference times. Video frame indices, source
timestamps, and source FPS remain attached to detections and tracks. If container
timestamps are unavailable, the application uses source FPS; only when neither is
available does it mark processing-time timestamps as estimated.

## Ball detector limitation

`yolo11n.pt` is a generic COCO detector restricted to class 32, `sports ball`.
It is a provisional architecture test, not a tennis-specific model. Small,
fast, blurred, distant, or partly occluded tennis balls may not be detected. The
detector interface is isolated so a custom tennis-ball model can replace it
without changing tracking or visualization. The implementation does not assume
one tennis-ball color.

The TrackNet backend is a three-frame PyTorch TrackNet V1-style experiment using
tennis-trained weights. It directly resizes three consecutive BGR source frames
to 640×360, stacks them newest-first into nine channels, and predicts a 256-class
heatmap for the newest frame. Coordinates are scaled independently back onto the
untouched original frame; the current 1920×1080 footage and model input are both
16:9, so there is no aspect-ratio distortion for that test video.

The weights were trained on 1280×720, 30 FPS tennis broadcast clips. Local 120 FPS
night-court phone footage is a domain and temporal-rate shift. TrackNet heatmap
classes do not yield calibrated object confidence, so TrackNet CSV confidence is
left blank rather than fabricated. See `THIRD_PARTY.md` for checkpoint hash,
provenance, licensing caveats, and rejected alternatives.

## Local V0.2 validation

Run the same sequence once with `--ball-detector yolo` and again with
`--ball-detector tracknet`. Keep the window focused and verify each case:

1. Hold a tennis ball still at several distances and backgrounds.
2. Move it slowly and confirm the yellow marker and trail follow it.
3. Move it rapidly and note missed detections or broken tracks.
4. Toss it vertically.
5. Throw it across the frame.
6. Hide it briefly behind your body and look for the orange `PREDICTED` state.
7. Remove it for several detection frames, confirm the track disappears, then
   bring it back and confirm a new track begins.
8. Run representative footage with `python main.py --video path/to/video.mp4` and
   verify the printed source FPS matches the file metadata.

Also confirm the pose skeleton still tracks correctly and record pipeline FPS,
pose inference milliseconds, ball inference milliseconds, lighting, camera
resolution, ball distance, and any false detections. A consistently missing ball
means tennis-specific data collection/training is the next step; do not compensate
by treating a fixed color threshold as a production detector.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Tennis-ball dataset candidates

V0.3A includes an offline generator for full-resolution, human-reviewable COCO
candidate data. TrackNet annotations are preliminary and must not be used as
ground truth without review. Inspect the source and leakage-safe temporal split
plan first:

```bash
python tools/build_ball_dataset.py \
  --videos videos/ \
  --output datasets/tennis_ball_v1 \
  --target-frames 300 \
  --dry-run
```

Then generate with `--device mps` (or omit it for automatic device selection).
See [docs/BALL_DATASET.md](docs/BALL_DATASET.md) for sampling, box sizing, COCO
output, review, and sanity-check instructions.

## RF-DETR experiment

The reviewed COCO export has been validated and rebuilt into guarded contiguous
temporal splits. RF-DETR is an optional third backend and uses an isolated pinned
environment so its training dependencies do not change the working YOLO/TrackNet
stack. Dataset statistics, rationale, setup, training, evaluation, benchmarking,
and inference commands are in
[docs/RFDETR_EXPERIMENT.md](docs/RFDETR_EXPERIMENT.md).

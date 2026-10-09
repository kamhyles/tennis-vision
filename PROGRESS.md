# Tennis Vision Progress

## Current Milestone
V0.3B — Fine-tuned RF-DETR tennis-ball detector experiment

## Status
RF-DETR SMALL TRAINED AND EVALUATED ON THE GUARDED SAME-RECORDING TEST SPLIT;
NEW-SESSION GENERALIZATION AND FINAL RAW-VIDEO VISUAL VALIDATION REMAIN OPEN

## Completed
- [x] Modular project scaffold and `python main.py` entry point
- [x] Configurable macOS webcam capture with clear initialization errors
- [x] YOLO11n Pose integration with one model instance and one inference pass per frame
- [x] Model-independent pose data containing all 17 COCO keypoint coordinates and confidences
- [x] Deterministic primary-person selection (largest bounding box, confidence tie-breaker)
- [x] Separate minimal skeleton, joint, and measured-FPS rendering
- [x] Safe handling of frames with no detected person
- [x] Camera/window cleanup on normal exit and processing errors
- [x] Unit tests for pose data, primary selection, inference conversion, FPS, and CLI configuration
- [x] README setup and run instructions
- [x] Real-world webcam validation
- [ ] Fast-motion validation
- [x] Pose skeleton successfully tracks the user locally
- [x] Approximately 15 pipeline FPS observed during V0.1 local validation
- [x] Provisional, replaceable COCO sports-ball detector
- [x] Structured ball detections with bounding box, center, confidence, frame index, and timestamp
- [x] Separate constant-velocity temporal ball tracker
- [x] `DETECTED`, `PREDICTED`, and `LOST` tracking states
- [x] Time-scaled jump rejection, brief missed-detection tolerance, and reset/reacquisition
- [x] Ball marker, state label, and short trajectory trail
- [x] Prerecorded video input with source FPS, frame index, and timestamp preservation
- [x] Independently configurable pose and ball inference frame intervals
- [x] Separate pipeline FPS, pose inference time, and ball inference time instrumentation
- [ ] Real tennis-ball detection/tracking validation
- [ ] Tennis-footage and fast-motion validation
- [x] Physical YOLO ball validation: stationary and slow ball detected/tracked
- [x] Physical YOLO limitation confirmed: detection becomes unreliable and the
  track is lost once the tennis ball moves faster than roughly a few mph
- [x] Interchangeable `yolo` and `tracknet` ball detector backends
- [x] Three-consecutive-source-frame TrackNet buffer
- [x] Explicit 640×360 preprocessing and original-resolution coordinate mapping
- [x] Optional per-frame CSV output and operational continuity diagnostics
- [x] TrackNet checkpoint download with fixed SHA-256 verification
- [x] Short real-footage TrackNet inference experiment
- [ ] Full-video TrackNet visual validation by the user
- [ ] Same-sequence YOLO versus TrackNet visual comparison
- [x] Split-first tennis-ball dataset planning with temporal boundary guards
- [x] Temporally spaced, spatially balanced TrackNet candidate sampling
- [x] Full-resolution candidate extraction without copying source videos
- [x] One-class COCO `tennis_ball` pre-annotation export
- [x] Per-image source provenance and review metadata
- [x] Candidate preview rendering and COCO-compatible review instructions
- [x] Dataset duplicate, adjacency, leakage, image, annotation, and box checks
- [x] Generated 236 real candidate images: 199 TrackNet prelabels and 37
  TrackNet miss/manual-review candidates
- [x] Human-reviewed 235 Astra Boxes annotations in Roboflow
- [x] Validated the authoritative reviewed COCO export and exact `tennis ball` class
- [x] Measured reviewed bounding-box pixel and source-frame-area distributions
- [x] Reconstructed contiguous train/valid/test regions from filename provenance
  with one-second temporal guard gaps
- [x] Prepared RF-DETR COCO layout with 175 train, 27 valid, and 33 test images
- [x] Pinned an isolated RF-DETR 1.10.1 Small/1024 reproducible baseline config
- [x] Added RF-DETR as an optional third detector backend without changing YOLO,
  TrackNet, or pose model loading
- [x] Added raw RF-DETR box, confidence, center, and trajectory visualization
- [x] Added COCO/center-localization evaluation and same-video benchmark tools
- [x] Train RF-DETR Small on Apple MPS and evaluate the best EMA checkpoint
- [x] Added timestamp-preserving normal-speed H.264 annotated-video export
- [x] Added lightweight animated and full-video README demo assets
- [ ] Perform short-video raw RF-DETR visual validation

## Current Models
- Pose: Ultralytics YOLO11n Pose (`yolo11n-pose.pt`). It successfully tracked the
  user in local webcam validation at approximately 15 total pipeline FPS in V0.1.
- Ball: Ultralytics YOLO11n detection (`yolo11n.pt`) restricted to COCO class 32,
  `sports ball`. Preserved as a baseline after physical testing confirmed it works
  for stationary/slow tennis balls but fails on faster motion.
- Experimental ball: temporal TrackNet V1-style PyTorch model from the unofficial
  `yastrebksv/TrackNet` implementation, using tennis checkpoint
  `tracknet_v1_tennis.pth`. SHA-256:
  `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076`.
  The Hugging Face artifact tagged MIT is byte-identical to the Google Drive
  checkpoint linked by the upstream repository. The upstream repository itself
  has no explicit license; see `THIRD_PARTY.md` before redistribution.
- Experimental RF-DETR: official RF-DETR Small from `rfdetr==1.10.1`, initialized
  from `rf-detr-small.pth` and fine-tuned at 1024×1024 on the reviewed one-class
  data. The best EMA checkpoint is
  `runs/rfdetr/tennis_ball_small_1024/checkpoint_best_total.pth`. Same-recording
  evaluation is promising but does not establish superiority over TrackNet.

## TrackNet Input / Output Contract
- Training domain reported upstream: tennis broadcast clips from ten videos,
  19,835 labeled frames, 1280×720 at 30 FPS.
- Input: three consecutive BGR frames, direct-resized to 640×360, normalized to
  `[0, 1]`, and stacked newest-first as one 9-channel tensor.
- Output: 256-class per-pixel intensity heatmap for the newest frame. The argmax
  map is thresholded and decoded using the upstream Hough-circle procedure.
- Coordinates are scaled explicitly from 640×360 to the original source frame.
  Current 1920×1080 footage is also 16:9, so this footage has no aspect distortion.
- The model does not provide calibrated object confidence; internal TrackNet
  detections use `confidence=None` rather than a fabricated score.

## Tracking Method
The single-ball tracker uses constant-velocity image-space prediction with a
time-scaled spatial gate. It selects the candidate nearest the predicted position,
rejects implausible jumps, predicts through up to four failed detection attempts,
then clears the history and becomes `LOST`. A detection after loss starts a new
track. Scheduled detector skips predict without counting as missed detections.

## Dataset Generation Method
- Source videos are assigned to contiguous train, validation, and test segments
  before sampling. A configurable guard band centered on each boundary is excluded
  to reduce leakage between immediate temporal neighborhoods.
- Selection enforces configurable frame and timestamp gaps and round-robins across
  source videos, TrackNet spatial grid cells, a rough normalized displacement
  proxy, and confidence bands when confidence exists. Current TrackNet confidence
  is unavailable, so no confidence ranking is fabricated.
- TrackNet misses can be retained as `TRACKNET_MISS_REVIEW` examples with no COCO
  annotation. They require a human to find a missed ball or confirm a true negative;
  they are not automatically classified as negatives.
- Candidate boxes start from TrackNet's source-space Hough-circle support diameter,
  use a configurable 1.5 scale factor, clamp to 12–48 pixels by default, and clip
  to image bounds. These boxes are initialization aids, not object boundaries.
- Output uses one-class COCO JSON (`tennis_ball`, category ID 1), split-specific
  COCO files and image lists, JSONL per-image metadata, source metadata, untouched
  full-resolution candidate images, and separate overlay previews.
- Generated datasets are ignored by source control. Source videos remain in place;
  the dataset `raw/` directory is intentionally empty.

**TrackNet pre-annotations are NOT ground truth and must be human-reviewed before
model training.**

## Reviewed RF-DETR Dataset
- Authoritative source: `Tennis Ball Detection.coco/train/`, exported from
  Roboflow after Astra Boxes annotation and human review.
- Exact leaf class: `tennis ball` (category ID 1, with a space). The export also
  retains Roboflow's unannotated `Tennis-Ball-Detection` grouping category ID 0.
- Audit: 235 readable 1920×1080 images, 234 valid boxes, 234 annotated images,
  one zero-annotation image (`...f00000661...`), no multi-annotation images, and
  no missing files, invalid references, zero-area boxes, or out-of-bounds boxes.
- Box width pixels: min 7, P10 8, P25 9, median 12, P75 14.75, P90 18.7,
  max 40, mean 12.74.
- Box height pixels: min 5, P10 8, P25 10, median 12, P75 15, P90 18,
  max 32, mean 12.79.
- Box area pixels: min 42, P10 64, P25 90, median 143, P75 224.75,
  P90 307.4, max 1280, mean 180.26.
- Box area as source-image percentage: min 0.00203%, P10 0.00309%,
  P25 0.00434%, median 0.00690%, P75 0.01084%, P90 0.01482%,
  max 0.06173%, mean 0.00869%.
- Split layout: `datasets/tennis_ball_reviewed_temporal/`. Train uses source
  frames [0, 2247) with 175 images; validation uses [2367, 2742) with 27;
  test uses [2862, 3296) with 33. The excluded guard ranges [2247, 2367) and
  [2742, 2862) are each 120 source frames (about one second) and contain no
  reviewed images.

This dataset originates from one recording and is not sufficient evidence of
cross-session generalization.

## RF-DETR Baseline Configuration
- Package: isolated `rfdetr[train]==1.10.1`; the existing project environment was
  not modified. RF-DETR core Nano–Large detection models use Apache-2.0; the
  selected Small model is in that core set.
- Existing Python 3.12.4 / torch 2.14.0 / torchvision 0.29.0 versions satisfy
  RF-DETR's broad minimums, but RF-DETR, Transformers 5.x, Pydantic 2.x,
  Supervision 0.29+, Lightning 2.6, and TorchMetrics training extras are absent.
- Variant/checkpoint: `RFDETRSmall`, official `rf-detr-small.pth` pretrained
  weights, not training from scratch.
- Resolution: 1024×1024 for training and inference. Direct square resize makes a
  median 12×12 source box approximately 6.4×11.4 input pixels, versus 3.2×5.7 at
  Small's 512 default. RF-DETR maps predictions back to source coordinates.
- Training: 50 maximum epochs, micro-batch 1, four-step gradient accumulation,
  AdamW, learning rate 1e-4, encoder learning rate 1e-5, weight decay 1e-4,
  seed 7, EMA, validation early stopping patience 10, and best-checkpoint skip 3.
- Augmentation: torchvision backend with optional augmentation config empty;
  multi-scale, expanded scales, and scale jitter disabled to avoid destroying
  tiny targets. Required resize/normalization remains.
- Actual device: Apple MPS with explicit micro-batch 1. Training validated epochs
  0–16, selected the EMA checkpoint from epoch 6, and ran for approximately
  57 minutes (13:28:42–14:26:05 local file timestamps).
- Output: `runs/rfdetr/tennis_ball_small_1024/`; best checkpoint
  `checkpoint_best_total.pth` (126 MB, ignored by Git).

## Validation Performed
- V0.1 physical webcam validation completed by the user: the camera opened and
  the pose skeleton tracked successfully at approximately 15 FPS.
- Python syntax/bytecode compilation completed successfully.
- 55 non-camera unit tests passed in the project virtual environment, including
  all previous pose, YOLO, TrackNet, media, and tracker tests.
- Synthetic tracker tests passed for continuous motion, temporary misses,
  recovery, implausible jumps, prolonged disappearance, a new track after loss,
  and scheduled detector skips.
- Ball-detector boundary tests passed for structured/timestamped detections and
  safe empty/no-result behavior.
- Media timing tests passed for source FPS, timestamps, frame indices, and timing
  fallback behavior.
- CLI argument/help path ran successfully.
- Actual YOLO11n Pose and YOLO11n detection weights both initialized and ran
  inference on a synthetic blank frame without false detections.
- A generated 12-frame, 120 FPS MP4 decoded through OpenCV with source FPS
  120.000, sequential indices, source timestamps, and clean EOF.
- A headless end-to-end run processed that video with the actual pose and ball
  models at independent inference intervals and exited cleanly at EOF.
- TrackNet-specific tests cover consecutive temporal buffering, discontinuity
  reset, newest-first channel order, preprocessing, 1920×1080 coordinate mapping,
  heatmap decoding, missing detections, internal detection conversion, and CLI
  backend/checkpoint selection.
- The 42.9 MB tennis checkpoint loaded strictly under the existing PyTorch stack
  and accepted the expected `(1, 9, 360, 640)` temporal input.
- Real video `videos/IMG_8356.MOV` inspected: H.264, 1920×1080, nominal 120 FPS,
  average metadata rate 118.483 FPS, 3,296 frames, 27.355 seconds.
- A frame-preserving 30-frame real-footage TrackNet run produced 28 direct
  localizations (93.3% of all source frames; the first two are buffer warm-up),
  zero predicted frames, two `LOST` warm-up frames, zero interruptions, and a
  28-frame continuous track.
- On source frame 2, TrackNet localized `(940.5, 310.5)`; direct visual inspection
  confirmed that marker was on the airborne tennis ball.
- This short section is promising enough to justify full visual comparison, but
  it is not sufficient to call TrackNet successful or superior to YOLO.
- Full-sequence accuracy, false positives, fast-ball continuity, and superiority
  over YOLO have not been established without full visual review/annotations.
- Dataset dry-run inspection found one source, `videos/IMG_8356.MOV`: 1920×1080,
  118.483 FPS average metadata rate, 3,296 frames, and 27.355 seconds container
  duration. With default 70/15/15 splits and a one-second boundary guard, planned
  eligible regions are train frames 0–2246, validation 2367–2741, and test
  2862–3295. No files were written by the dry run.
- Dataset tests cover temporal spacing, spatial diversity, source-coordinate box
  creation and clipping, metadata, split grouping/leakage prevention, one-class
  COCO validity, TrackNet misses, temporal-buffer reset, and sanity-check failures.
- An end-to-end smoke build used the real TrackNet checkpoint on a generated
  12-frame blank video. It scanned all 12 frames, generated three full-resolution
  miss-review candidates, wrote all dataset artifacts, and passed sanity checks
  with zero errors. Duplicate-content warnings correctly identified the identical
  synthetic frames.
- The real V0.3A run generated 236 candidates: 199 TrackNet prelabels and 37
  TrackNet miss/manual-review candidates. Roboflow/Astra plus human review
  produced 235 approved authoritative images.
- Full pixel decoding and COCO validation passed for all 235 reviewed images.
  The audit found 234 valid annotations and one zero-annotation image, with no
  missing files, invalid references, non-positive boxes, or out-of-bounds boxes.
- Filename provenance parsing assigned all reviewed images to the original
  guarded timeline: 175 train, 27 valid, 33 test, and zero inside guard gaps.
- Prepared split COCO files preserve the exact `tennis ball` category.
- RF-DETR adapter tests cover RGB conversion, source-coordinate preservation,
  exact-class filtering, confidence ordering, no detections, threshold/config
  forwarding, and multiple candidates. Center metric tests cover exact/error,
  misses, false positives, extras, recall, and IoU.
- Python compilation, the 55-test suite, reviewed-dataset inspection, prepared
  split creation, and application help/CLI checks completed successfully.
- RF-DETR 1.10.1 trained successfully in the isolated `.venv-rfdetr` environment
  on Apple MPS. Validation ran for epochs 0–16; best EMA mAP@50:95 was 0.639 at
  epoch 6, which supplied `checkpoint_best_total.pth`.
- Guarded test evaluation produced mAP@50:95 0.534, mAP@50 0.924, mAP@75 0.653,
  mAR 0.642, precision 1.000, recall 0.879, and F1 0.935.
- At confidence 0.25 the center-localization pass returned a candidate on all 33
  annotated test images with zero misses, but emitted 17 extra detections. There
  were no negative test images, so false-positive behavior on ball-absent frames
  was not measured.
- Matched-center error was median 0.57 px, P90 1.42 px, and mean 21.16 px. The
  mean is dominated by a few severe candidate-selection outliers: 90.9% of
  matched centers were within 5 px. Median IoU was 0.802.
- RF-DETR center evaluation measured mean 284.6 ms and median 239.6 ms per image,
  or about 3.51 inference FPS at 1024×1024 on Apple MPS.
- Offline video-export tests cover CLI selection, 120-to-60 FPS timestamp
  sampling, and gap duplication. A real FFmpeg smoke export encoded 120 synthetic
  timestamped frames as a one-second, 60-frame H.264 MP4.
- A headless end-to-end application smoke run analyzed four frames from
  `videos/IMG_8356.MOV` with the actual YOLO pose and ball models and exported a
  readable 1920×1080 H.264/yuv420p MP4 at 60 FPS. This validated the export path,
  not the full RF-DETR video or its visual accuracy.
- A separate two-frame export loaded the real fine-tuned RF-DETR checkpoint,
  produced direct detections on CPU, and wrote a readable H.264 MP4. MPS could not
  be retested inside the restricted execution environment because PyTorch reported
  `torch.backends.mps.is_available() == False`; prior local MPS training and
  evaluation remain the applicable hardware validation.
- The completed 27.35-second annotated export was transcoded into a 4.5 MB,
  640-pixel animated README preview and a 1.9 MB, 1280×720, 30 FPS H.264 full demo.
  Sampled frames were visually inspected for readable pose, ball, and trajectory
  overlays; this was not a frame-by-frame accuracy review.
- RF-DETR short-video visual validation and same-footage three-backend benchmark
  have not yet been completed.

## Observed Performance
V0.1 pose-only webcam pipeline: approximately 15 FPS, reported from local user
validation.

V0.2 combined pose-and-ball pipeline FPS: TBD — requires local validation.

TrackNet short-run CPU measurement in the Codex environment: average TrackNet
stage time 790.4 ms per source frame (about 1.27 inference FPS); one sampled pose
inference took 1011.8 ms. MPS was unavailable in this environment and remains to
be measured locally.

Dataset smoke build on CPU: 12 synthetic source frames scanned in 9.1 seconds.
The completed real-dataset generator performance was not recorded in this task.

RF-DETR Apple MPS training: approximately 57 minutes for 17 validation epochs;
best EMA validation mAP@50:95 0.639 at epoch 6. Guarded test mAP@50:95 0.534;
center-evaluation throughput approximately 3.51 FPS at 1024×1024. These are
same-recording experimental results, not cross-session evidence.

Resolution: Configured for 1280x720 capture and 640-pixel inference by default;
actual camera-negotiated resolution not yet validated.

Hardware: Local V0.1 hardware details not recorded.

## Known Limitations / Issues
- Generic YOLO11n COCO detection is physically confirmed inadequate for faster
  tennis-ball motion and remains a baseline only.
- The selected TrackNet checkpoint is tennis-trained, but its 30 FPS broadcast
  training domain differs from 120 FPS wide-angle night phone footage.
- The selected upstream PyTorch implementation has no explicit repository license;
  only the redistributed checkpoint source labels the artifact MIT.
- TrackNet inference is slow on CPU; Apple MPS performance is not yet measured.
- The 256-class heatmap supplies no calibrated object-confidence probability.
- Operational localization/continuity counts do not prove correctness; a detector
  can localize the wrong object and still increase these metrics.
- A false positive can start a new track while the tracker is `LOST`; motion gating
  primarily protects continuity after a track has begun.
- Motion-gate parameters are configurable but have not been tuned on tennis footage.
- Predictions cover only a small number of failed detection attempts and are not
  physical trajectory or velocity estimates.
- Primary-person selection remains frame-by-frame; persistent pose identity
  tracking is not part of V0.2.
- This is 2D image-space pose estimation. Occlusion, motion blur, poor lighting,
  distance, and fast tennis motion can reduce keypoint accuracy.
- The first run needs internet access unless both model weight files are present.
- Only one short real source video is currently available. Contiguous splits avoid
  direct adjacent-frame leakage, but they do not constitute a strong independent
  test across sessions, cameras, courts, or lighting.
- The prepared test split is temporally separated but comes from the same
  recording. It cannot establish cross-session generalization.
- Split boundaries are time-based heuristics and do not know rally boundaries; a
  rally spanning a guarded boundary can still create semantic similarity.
- Spatial and displacement bins encourage diversity but do not prove examples are
  stationary, fast, blurred, near the net, or otherwise semantically balanced.
- Only 235 reviewed images and 234 positive boxes are available; this is a small
  proof-of-concept dataset vulnerable to overfitting and scene memorization.
- Direct 16:9-to-square RF-DETR resizing distorts aspect ratio, though prediction
  coordinates are mapped back to the source frame. The 1024 input improves tiny
  target sampling at substantial memory and latency cost.
- RF-DETR training and still-image evaluation ran on Apple MPS, but end-to-end
  video throughput and raw visual stability remain unvalidated.
- The test split has no ball-absent frames, and RF-DETR produced 17 extra
  detections on its 33 positive frames. Negative-frame false positives and
  multiple-candidate selection need explicit testing.
- The RF-DETR confidence threshold (0.25) and tracker motion gate are provisional
  and must be evaluated, not treated as calibrated.

## Decisions
- Preserve the working YOLO11n Pose V0.1 subsystem without coupling it to ball work.
- Preserve generic YOLO11n COCO sports-ball detection as a baseline, not primary.
  Do not add fixed-color thresholding to mask inadequate detector recall.
- Keep `BallDetector` replaceable so a tennis-specific model requires no tracker
  or visualization rewrite.
- Select the temporal PyTorch TrackNet V1 tennis checkpoint for this experiment
  because it is compatible with the existing PyTorch environment and verified as
  tennis-trained. Reject available badminton/shuttlecock checkpoints.
- Reject the official TensorFlow TrackNetV4 integration for now: although its code
  is MIT-licensed and supports tennis, published tennis checkpoint links are
  placeholders and its Python 3.9/TensorFlow 2.17 environment would add a second
  heavyweight framework without a usable checkpoint.
- Reject the ArtLabss included checkpoint because inspection showed a single-frame
  three-channel TrackNet I model, not the requested temporal model.
- Require `--ball-every 1` for TrackNet so no source frame is dropped from its
  temporal input. Pose inference may still run at a lower independent rate.
- Use explainable constant-velocity prediction and spatial gating instead of a
  more complex filter until real footage justifies additional complexity.
- Preserve source frame index, timestamp, FPS, and timing provenance.
- Allow pose and ball inference to use independent source-frame intervals.
- Keep pose inference and numerical outputs independent from visualization.
- Render only the selected person's body skeleton without YOLO boxes or labels.
- Use the largest visible person as primary, with detection confidence as a
  deterministic tie-breaker.
- Start with local webcam/video before phone streaming.
- Do not calculate velocity from the current image-space track.
- Do not begin later milestones until ball detection and tracking are validated.
- Keep dataset generation separate from inference/application orchestration and
  do not duplicate source videos into the dataset workspace.
- Split each video into contiguous guarded temporal regions before sampling; never
  randomly split adjacent high-FPS frames after extraction.
- Preserve TrackNet misses only as unconfirmed review candidates, never as assumed
  negatives.
- Treat only the Roboflow/Astra plus human-reviewed COCO boxes as authoritative;
  never train from the TrackNet candidate boxes.
- Preserve the exact dataset leaf category `tennis ball`; do not silently rename
  it to the earlier generator's `tennis_ball` spelling.
- Reuse the original contiguous V0.3A split boundaries and one-second guards;
  never use Roboflow/random image-level splits for this same-recording dataset.
- Select one RF-DETR Small baseline at 1024 rather than a model/parameter sweep.
  Resolution is prioritized because the median source box is only 12×12 pixels.
- Fine-tune official `rf-detr-small.pth`; do not train from scratch.
- Keep RF-DETR dependencies in an isolated environment pinned to 1.10.1.
- Disable aggressive geometric augmentation for the first tiny-object baseline.
- Preserve all threshold-passing RF-DETR candidates. On acquisition the existing
  tracker chooses confidence; while tracking it uses nearest plausible motion.
  Do not smooth direct RF-DETR coordinates during this comparison.
- Do not silently run 1024 RF-DETR training on CPU. Use verified local MPS or a
  supported CUDA host and record the actual device and duration.

## Next Step
Run the best RF-DETR checkpoint on a short representative source-video section,
benchmark YOLO, TrackNet, and RF-DETR on identical frames, and have the user
visually assess whether raw RF-DETR centers remain locked to the moving ball.
Investigate duplicate/outlier candidate selection without smoothing coordinates.
Then acquire a completely new recording containing both ball-present and
ball-absent frames for a true held-out generalization and false-positive test.

## Future Milestones
- V0.3 Racket tracking
- V0.4 Court calibration
- V0.5 Ball velocity
- V0.6 Contact/stroke detection

# Third-party model provenance

## Experimental TrackNet tennis backend

The experimental backend implements the three-frame TrackNet V1-style network
described by Huang et al., *TrackNet: A Deep Learning Network for Tracking
High-speed and Tiny Objects in Sports Applications* (2019).

- PyTorch architecture inspected: `yastrebksv/TrackNet`, an unofficial TrackNet
  implementation. That repository does not include an explicit software license,
  so this project does not vendor the repository.
- Checkpoint: `tracknet_weights.pth`, trained for tennis and linked by the same
  upstream repository.
- Redistributed checkpoint source: Hugging Face model
  `vishnushenoy09/tracknet-v1-tennis`, which labels the artifact MIT.
- Verified SHA-256:
  `c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076`.
- Provenance check: the Hugging Face artifact is byte-for-byte identical to the
  Google Drive checkpoint linked by `yastrebksv/TrackNet`.
- Training domain reported by upstream: tennis broadcast clips from ten videos,
  1280×720 at 30 FPS, with 19,835 labeled frames.
- Input contract: three consecutive BGR frames resized to 640×360, concatenated
  newest-first into nine channels, float32 in `[0, 1]`.
- Output contract: a 256-class per-pixel intensity heatmap for the newest frame;
  the upstream implementation decodes a thresholded argmax map with Hough circles.

This experiment does not assert that the checkpoint license tag covers the
unlicensed upstream implementation. Review that caveat before redistribution or
commercial use. The checkpoint is intentionally ignored by source control.

## Alternatives investigated

- Official TrackNetV4: MIT code, TensorFlow 2.17/Python 3.9 reference environment,
  three 512×288 frames and three output heatmaps. Its published tennis result
  table currently links checkpoints to placeholders, so no usable official tennis
  checkpoint was available for this integration.
- TrackNetV2/V3 PyTorch releases with working checkpoints were rejected because
  the available weights were trained on badminton shuttlecock data.
- ArtLabss `tennis-tracking`: Unlicense code and an included tennis checkpoint,
  but inspection showed a single-frame, three-channel TrackNet I model rather than
  the temporal model required for this milestone.

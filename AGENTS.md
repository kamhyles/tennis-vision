# Tennis Vision — Codex Instructions

## Project Goal
Build a computer-vision tennis analysis system that can eventually track:
- human pose/body mechanics
- tennis racket
- tennis ball
- ball trajectory and velocity
- court geometry
- stroke/contact events
- biomechanical metrics
- player-specific coaching recommendations

## Development Philosophy
Build incrementally and validate each subsystem before moving on.

At the beginning of a task, read PROGRESS.md. At the end of a task, update it to reflect only work actually completed and validated.

Do not implement future milestones unless explicitly requested.

Prioritize:
1. measurement accuracy
2. temporal stability
3. modularity
4. real-time performance
5. scientifically defensible outputs

Never fabricate physical measurements when the available camera data cannot reliably support them.

## Architecture
Keep these concerns separated:
- detection/inference
- temporal tracking
- physical/geometry calculations
- visualization
- biomechanics
- application/UI

Raw model outputs and measurements must remain accessible independently of visualization.

## Current Development Order
1. Human pose tracking
2. Ball detection
3. Ball temporal tracking
4. Racket tracking
5. Court detection/calibration
6. Ball velocity
7. Contact/stroke detection
8. Advanced pose/3D
9. Biomechanics
10. Coaching

## Coding Rules
- Python
- Type hints for public APIs
- Modular classes/functions
- Avoid giant files
- No duplicate model loading
- Handle missing detections gracefully
- Document non-obvious geometry/math
- Prefer configurable values over magic numbers
- Don't silently catch important errors

## Validation
A feature is not complete merely because the code runs.

For every milestone:
- test it
- report what was actually tested
- distinguish tested behavior from assumptions
- document known limitations
- do not claim hardware/camera testing if the environment cannot perform it
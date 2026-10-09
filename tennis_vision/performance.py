"""Low-clutter processing-time instrumentation."""

from __future__ import annotations


class PerformanceMonitor:
    """Accumulate and periodically report independent inference timings."""

    def __init__(self, log_interval_frames: int = 120) -> None:
        if log_interval_frames < 0:
            raise ValueError("log_interval_frames cannot be negative")
        self.log_interval_frames = log_interval_frames
        self._pose_total = 0.0
        self._pose_count = 0
        self._ball_total = 0.0
        self._ball_count = 0

    def record_pose(self, seconds: float) -> None:
        self._pose_total += seconds
        self._pose_count += 1

    def record_ball(self, seconds: float) -> None:
        self._ball_total += seconds
        self._ball_count += 1

    @property
    def average_pose_ms(self) -> float | None:
        if not self._pose_count:
            return None
        return 1000.0 * self._pose_total / self._pose_count

    @property
    def average_ball_ms(self) -> float | None:
        if not self._ball_count:
            return None
        return 1000.0 * self._ball_total / self._ball_count

    def maybe_log(self, frame_index: int, pipeline_fps: float) -> None:
        if (
            not self.log_interval_frames
            or frame_index <= 0
            or frame_index % self.log_interval_frames
        ):
            return
        pose = self.average_pose_ms
        ball = self.average_ball_ms
        pose_text = "n/a" if pose is None else f"{pose:.1f} ms"
        ball_text = "n/a" if ball is None else f"{ball:.1f} ms"
        print(
            f"Performance @ frame {frame_index}: pipeline={pipeline_fps:.1f} FPS, "
            f"pose={pose_text}, ball={ball_text}"
        )

    def summary(self) -> str:
        pose = self.average_pose_ms
        ball = self.average_ball_ms
        pose_text = "n/a" if pose is None else f"{pose:.1f} ms"
        ball_text = "n/a" if ball is None else f"{ball:.1f} ms"
        return f"average pose inference={pose_text}; ball inference={ball_text}"

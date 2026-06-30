"""RCDW phase 时间结构定义。

v1.2 YAGNI 决策：仅保留 ``STANDARD_EXPOSURE`` 一种 schedule。
对未实现 profile 显式 ``raise NotImplementedError``，禁止隐式回退。

未来激活其他 schedule（``VARIABLE_ONSET`` / ``FAST_TRANSIENT`` /
``INCOMPLETE_RECOVERY`` / ``MULTI_PULSE``）需按方案 §5.2 末尾三步路径
同步改动 ``generate_condition_rows`` (1:N)、本文件、validation 不变量。

dataclass 结构与 HG 主线 ``src/sim/generation/phases.py`` 等价，
但仅保留 RCDW 实际使用的一个 schedule 实例。对应方案 §2.5 / §5.2。
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PhaseSegment:
    name: str
    duration_frac: float
    blend_shape: str
    blend_floor: float = 0.0


@dataclass(frozen=True, slots=True)
class PhaseSchedule:
    name: str
    segments: tuple[PhaseSegment, ...]

    def __post_init__(self) -> None:
        if len(self.segments) < 1:
            raise ValueError("phase schedule must contain at least one segment")
        if any(segment.duration_frac <= 0.0 for segment in self.segments):
            raise ValueError("phase segment duration_frac values must be > 0")
        if any(
            segment.blend_floor < 0.0 or segment.blend_floor > 1.0
            for segment in self.segments
        ):
            raise ValueError("phase segment blend_floor values must be in [0, 1]")
        total = sum(segment.duration_frac for segment in self.segments)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("phase segment duration_frac values must sum to 1.0")

    def boundaries(self, timesteps: int) -> tuple[int, ...]:
        _validate_timesteps(timesteps)
        if timesteps < len(self.segments):
            raise ValueError(
                f"timesteps must be >= number of phase segments ({len(self.segments)})"
            )
        boundaries: list[int] = []
        accumulated = 0.0
        for segment in self.segments[:-1]:
            accumulated += segment.duration_frac
            boundary = int(timesteps * accumulated + 1e-9)
            boundaries.append(min(timesteps - 1, max(1, boundary)))
        result = tuple(boundaries)
        previous = 0
        for boundary in result:
            if boundary <= previous:
                raise ValueError(
                    f"schedule {self.name!r} collapses to an empty phase at "
                    f"timesteps={timesteps}; increase timesteps or widen the shortest segment"
                )
            previous = boundary
        return result

    def phase_for_timestep(self, timestep: int, timesteps: int) -> str:
        segment, _start, _end = self.segment_for_timestep(timestep, timesteps)
        return segment.name

    def blend_for_timestep(self, timestep: int, timesteps: int) -> float:
        segment, start, end = self.segment_for_timestep(timestep, timesteps)
        return _blend_at(segment, timestep - start, max(1, end - start))

    def segment_for_timestep(
        self, timestep: int, timesteps: int
    ) -> tuple[PhaseSegment, int, int]:
        _validate_timestep(timestep, timesteps)
        starts = (0, *self.boundaries(timesteps))
        ends = (*self.boundaries(timesteps), timesteps)
        for segment, start, end in zip(self.segments, starts, ends, strict=True):
            if start <= timestep < end:
                return segment, start, end
        raise ValueError(f"timestep {timestep} is outside [0, {timesteps})")

    def resolve_timeline(
        self, timesteps: int
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """逐时间步返回 ``(phase_id, blend)``，整段只计算一次阶段边界。"""
        bounds = self.boundaries(timesteps)
        starts = (0, *bounds)
        ends = (*bounds, timesteps)
        phase_ids: list[str] = []
        blends: list[float] = []
        for segment, start, end in zip(self.segments, starts, ends, strict=True):
            length = max(1, end - start)
            for timestep in range(start, end):
                phase_ids.append(segment.name)
                blends.append(_blend_at(segment, timestep - start, length))
        return tuple(phase_ids), tuple(blends)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "segments": [
                {
                    "name": segment.name,
                    "duration_frac": float(segment.duration_frac),
                    "blend_shape": segment.blend_shape,
                    "blend_floor": float(segment.blend_floor),
                }
                for segment in self.segments
            ],
        }

    def jittered(self, rng: random.Random, jitter_frac: float) -> "PhaseSchedule":
        if jitter_frac == 0.0:
            return self
        if jitter_frac < 0.0 or jitter_frac >= 1.0:
            raise ValueError("jitter_frac must be in [0, 1)")
        factors = [
            rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac)
            for _segment in self.segments
        ]
        raw_durations = [
            segment.duration_frac * factor
            for segment, factor in zip(self.segments, factors, strict=True)
        ]
        total = sum(raw_durations)
        return PhaseSchedule(
            name=self.name,
            segments=tuple(
                PhaseSegment(
                    name=segment.name,
                    duration_frac=duration / total,
                    blend_shape=segment.blend_shape,
                    blend_floor=segment.blend_floor,
                )
                for segment, duration in zip(
                    self.segments, raw_durations, strict=True
                )
            ),
        )


# v1.2 YAGNI：RCDW 仅保留这一个 schedule 实例。
# 与 HG 主线一致的 4 段：baseline / exposure / steady / recovery，
# 各占 15% / 25% / 35% / 25%；recovery 段末尾 blend 不归零到 0，
# 而是回到 0.05 的 floor，模拟真实标定中"恢复未到稳态"的物理。
STANDARD_EXPOSURE = PhaseSchedule(
    name="standard_exposure",
    segments=(
        PhaseSegment("baseline", 0.15, "hold0"),
        PhaseSegment("exposure", 0.25, "ramp_up"),
        PhaseSegment("steady", 0.35, "hold1"),
        PhaseSegment("recovery", 0.25, "ramp_down", blend_floor=0.05),
    ),
)

# 显式注册表：v1.2 仅含一项。若未来按方案 §5.2 三步路径激活其他 schedule，
# 在此添加注册项并相应更新 resolve_phase_schedule 的分支查找。
PHASE_SCHEDULES: dict[str, PhaseSchedule] = {
    STANDARD_EXPOSURE.name: STANDARD_EXPOSURE,
}


def resolve_phase_schedule(stage_profile: str | PhaseSchedule) -> PhaseSchedule:
    """解析 stage_profile 字符串或 PhaseSchedule 实例。

    v1.2 仅支持 ``"standard_exposure"``；其他 profile 显式 raise
    ``NotImplementedError``，禁止隐式回退到默认 schedule。
    """
    if isinstance(stage_profile, PhaseSchedule):
        return stage_profile
    if stage_profile in PHASE_SCHEDULES:
        return PHASE_SCHEDULES[stage_profile]
    raise NotImplementedError(
        f"RCDW v1.x 仅支持 stage_profile='standard_exposure',收到 {stage_profile!r}。"
        " 要新增其他 schedule (VARIABLE_ONSET / FAST_TRANSIENT / "
        "INCOMPLETE_RECOVERY / MULTI_PULSE 等),请按 "
        "docs/学长算法/RCDW_数据集主线对齐改动方案.md §5.2 末尾的"
        "「未来激活路径」同步改三处后再添加。"
    )


def phase_boundaries(timesteps: int) -> tuple[int, int, int]:
    """便捷函数：返回 STANDARD_EXPOSURE 的三个边界 (q1, q2, q3)。"""
    q1, q2, q3 = STANDARD_EXPOSURE.boundaries(timesteps)
    return q1, q2, q3


def phase_for_timestep(timestep: int, timesteps: int) -> str:
    return STANDARD_EXPOSURE.phase_for_timestep(timestep, timesteps)


def blend_for_timestep(timestep: int, timesteps: int) -> float:
    return STANDARD_EXPOSURE.blend_for_timestep(timestep, timesteps)


def _validate_timesteps(timesteps: int) -> None:
    if timesteps < 4:
        raise ValueError("timesteps must be >= 4")


def _validate_timestep(timestep: int, timesteps: int) -> None:
    _validate_timesteps(timesteps)
    if timestep < 0 or timestep >= timesteps:
        raise ValueError(f"timestep must be in [0, {timesteps}), got {timestep}")


def _blend_at(segment: PhaseSegment, local: int, length: int) -> float:
    if segment.blend_shape == "hold0":
        return 0.0
    if segment.blend_shape == "ramp_up":
        return min(1.0, (local + 1) / length)
    if segment.blend_shape == "hold1":
        return 1.0
    if segment.blend_shape == "ramp_down":
        return segment.blend_floor + (1.0 - segment.blend_floor) * (
            1.0 - ((local + 1) / length)
        )
    raise ValueError(f"unknown blend_shape: {segment.blend_shape!r}")

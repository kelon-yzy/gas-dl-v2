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
        if any(segment.blend_floor < 0.0 or segment.blend_floor > 1.0 for segment in self.segments):
            raise ValueError("phase segment blend_floor values must be in [0, 1]")
        total = sum(segment.duration_frac for segment in self.segments)
        if abs(total - 1.0) > 1e-9:
            raise ValueError("phase segment duration_frac values must sum to 1.0")

    def boundaries(self, timesteps: int) -> tuple[int, ...]:
        _validate_timesteps(timesteps)
        if timesteps < len(self.segments):
            raise ValueError(f"timesteps must be >= number of phase segments ({len(self.segments)})")
        boundaries = []
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
                    f"schedule {self.name!r} collapses to an empty phase at timesteps={timesteps}; "
                    "increase timesteps or widen the shortest segment"
                )
            previous = boundary
        return result

    def phase_for_timestep(self, timestep: int, timesteps: int) -> str:
        segment, _start, _end = self.segment_for_timestep(timestep, timesteps)
        return segment.name

    def blend_for_timestep(self, timestep: int, timesteps: int) -> float:
        segment, start, end = self.segment_for_timestep(timestep, timesteps)
        return _blend_at(segment, timestep - start, max(1, end - start))

    def segment_for_timestep(self, timestep: int, timesteps: int) -> tuple[PhaseSegment, int, int]:
        _validate_timestep(timestep, timesteps)
        starts = (0, *self.boundaries(timesteps))
        ends = (*self.boundaries(timesteps), timesteps)
        for segment, start, end in zip(self.segments, starts, ends, strict=True):
            if start <= timestep < end:
                return segment, start, end
        raise ValueError(f"timestep {timestep} is outside [0, {timesteps})")

    def resolve_timeline(self, timesteps: int) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """逐时间步返回 ``(phase_id, blend)``，整段只计算一次阶段边界。

        等价于对每个 timestep 调用 ``phase_for_timestep``/``blend_for_timestep``，
        但避免长序列生成循环中重复计算 boundaries。
        """
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

    def jittered(self, rng: random.Random, jitter_frac: float) -> PhaseSchedule:
        if jitter_frac == 0.0:
            return self
        if jitter_frac < 0.0 or jitter_frac >= 1.0:
            raise ValueError("jitter_frac must be in [0, 1)")
        factors = [rng.uniform(1.0 - jitter_frac, 1.0 + jitter_frac) for _segment in self.segments]
        raw_durations = [segment.duration_frac * factor for segment, factor in zip(self.segments, factors, strict=True)]
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
                for segment, duration in zip(self.segments, raw_durations, strict=True)
            ),
        )


STANDARD_EXPOSURE = PhaseSchedule(
    name="standard_exposure",
    segments=(
        PhaseSegment("baseline", 0.25, "hold0"),
        PhaseSegment("exposure", 0.25, "ramp_up"),
        PhaseSegment("steady", 0.25, "hold1"),
        PhaseSegment("recovery", 0.25, "ramp_down"),
    ),
)
VARIABLE_ONSET = PhaseSchedule(
    name="variable_onset",
    segments=(
        PhaseSegment("baseline", 0.35, "hold0"),
        PhaseSegment("exposure", 0.20, "ramp_up"),
        PhaseSegment("steady", 0.25, "hold1"),
        PhaseSegment("recovery", 0.20, "ramp_down"),
    ),
)
FAST_TRANSIENT = PhaseSchedule(
    name="fast_transient",
    segments=(
        PhaseSegment("baseline", 0.45, "hold0"),
        PhaseSegment("exposure", 0.12, "ramp_up"),
        PhaseSegment("steady", 0.08, "hold1"),
        PhaseSegment("recovery", 0.35, "ramp_down"),
    ),
)
INCOMPLETE_RECOVERY = PhaseSchedule(
    name="incomplete_recovery",
    segments=(
        PhaseSegment("baseline", 0.25, "hold0"),
        PhaseSegment("exposure", 0.25, "ramp_up"),
        PhaseSegment("steady", 0.25, "hold1"),
        PhaseSegment("recovery", 0.25, "ramp_down", blend_floor=0.2),
    ),
)
MULTI_PULSE = PhaseSchedule(
    name="multi_pulse",
    segments=(
        PhaseSegment("baseline", 1.0 / 12.0, "hold0"),
        PhaseSegment("exposure", 1.0 / 12.0, "ramp_up"),
        PhaseSegment("steady", 1.0 / 12.0, "hold1"),
        PhaseSegment("recovery", 1.0 / 12.0, "ramp_down"),
        PhaseSegment("baseline", 1.0 / 12.0, "hold0"),
        PhaseSegment("exposure", 1.0 / 12.0, "ramp_up"),
        PhaseSegment("steady", 1.0 / 12.0, "hold1"),
        PhaseSegment("recovery", 1.0 / 12.0, "ramp_down"),
        PhaseSegment("baseline", 1.0 / 12.0, "hold0"),
        PhaseSegment("exposure", 1.0 / 12.0, "ramp_up"),
        PhaseSegment("steady", 1.0 / 12.0, "hold1"),
        PhaseSegment("recovery", 1.0 / 12.0, "ramp_down"),
    ),
)

PHASE_SCHEDULES = {
    STANDARD_EXPOSURE.name: STANDARD_EXPOSURE,
    VARIABLE_ONSET.name: VARIABLE_ONSET,
    FAST_TRANSIENT.name: FAST_TRANSIENT,
    INCOMPLETE_RECOVERY.name: INCOMPLETE_RECOVERY,
    MULTI_PULSE.name: MULTI_PULSE,
}


def resolve_phase_schedule(stage_profile: str | PhaseSchedule) -> PhaseSchedule:
    if isinstance(stage_profile, PhaseSchedule):
        return stage_profile
    try:
        return PHASE_SCHEDULES[stage_profile]
    except KeyError as exc:
        raise ValueError(f"stage_profile must be one of {sorted(PHASE_SCHEDULES)}, got {stage_profile!r}") from exc


def phase_boundaries(timesteps: int) -> tuple[int, int, int]:
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
        return segment.blend_floor + (1.0 - segment.blend_floor) * (1.0 - ((local + 1) / length))
    raise ValueError(f"unknown blend_shape: {segment.blend_shape!r}")

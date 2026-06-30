"""测试 RCDW phases.py: 仅保留 STANDARD_EXPOSURE + 显式 NotImplementedError 契约。

对应方案 §2.5 / §5.2 / §11.1。
"""
from __future__ import annotations

import random

import pytest

from rcdw.sim.generation.phases import (
    PHASE_SCHEDULES,
    STANDARD_EXPOSURE,
    PhaseSchedule,
    PhaseSegment,
    blend_for_timestep,
    phase_boundaries,
    phase_for_timestep,
    resolve_phase_schedule,
)


def test_registry_only_contains_standard_exposure():
    """v1.2 YAGNI: PhaseSchedule 注册表仅含一项。"""
    assert set(PHASE_SCHEDULES.keys()) == {"standard_exposure"}
    assert PHASE_SCHEDULES["standard_exposure"] is STANDARD_EXPOSURE


def test_standard_exposure_segments():
    """4 段: baseline / exposure / steady / recovery,占比 15/25/35/25。"""
    assert STANDARD_EXPOSURE.name == "standard_exposure"
    assert len(STANDARD_EXPOSURE.segments) == 4
    names = [s.name for s in STANDARD_EXPOSURE.segments]
    assert names == ["baseline", "exposure", "steady", "recovery"]
    fracs = [s.duration_frac for s in STANDARD_EXPOSURE.segments]
    assert fracs == [0.15, 0.25, 0.35, 0.25]
    assert abs(sum(fracs) - 1.0) < 1e-9
    # recovery 段 blend_floor = 0.05
    assert STANDARD_EXPOSURE.segments[3].blend_floor == pytest.approx(0.05)


def test_duration_frac_sum_invariant():
    """所有 PhaseSchedule 实例的 duration_frac 之和必须 = 1。"""
    for schedule in PHASE_SCHEDULES.values():
        total = sum(s.duration_frac for s in schedule.segments)
        assert abs(total - 1.0) < 1e-9


def test_invalid_duration_frac_sum_rejected():
    """构造时若 duration_frac 不和为 1, 抛 ValueError。"""
    with pytest.raises(ValueError, match="sum to 1.0"):
        PhaseSchedule(
            name="bad",
            segments=(
                PhaseSegment("a", 0.5, "hold0"),
                PhaseSegment("b", 0.3, "hold1"),
            ),
        )


def test_boundaries_shape_and_monotonic():
    bounds = STANDARD_EXPOSURE.boundaries(128)
    assert len(bounds) == 3  # n_segments - 1
    assert bounds[0] < bounds[1] < bounds[2]
    assert all(0 < b < 128 for b in bounds)


def test_phase_for_timestep_assignment():
    """各 timestep 应被正确归类到 4 段之一。"""
    T = 100
    bounds = STANDARD_EXPOSURE.boundaries(T)
    # baseline: [0, bounds[0])
    assert phase_for_timestep(0, T) == "baseline"
    assert phase_for_timestep(bounds[0] - 1, T) == "baseline"
    # exposure: [bounds[0], bounds[1])
    assert phase_for_timestep(bounds[0], T) == "exposure"
    assert phase_for_timestep(bounds[1] - 1, T) == "exposure"
    # steady: [bounds[1], bounds[2])
    assert phase_for_timestep(bounds[1], T) == "steady"
    # recovery: [bounds[2], T)
    assert phase_for_timestep(T - 1, T) == "recovery"


def test_blend_shape_values():
    """blend 在各段的取值范围: hold0=0, ramp_up [0,1], hold1=1, ramp_down [0.05,1]。"""
    T = 100
    # baseline (hold0): blend == 0
    assert blend_for_timestep(0, T) == 0.0
    # steady (hold1): blend == 1
    bounds = STANDARD_EXPOSURE.boundaries(T)
    assert blend_for_timestep(bounds[1] + 1, T) == 1.0
    # recovery 末端: blend → blend_floor = 0.05
    last_blend = blend_for_timestep(T - 1, T)
    assert abs(last_blend - 0.05) < 1e-9


def test_resolve_timeline_full_length():
    """resolve_timeline 返回的 phase_ids/blends 长度应 == timesteps。"""
    T = 64
    phase_ids, blends = STANDARD_EXPOSURE.resolve_timeline(T)
    assert len(phase_ids) == T
    assert len(blends) == T
    # 第一段应是 baseline,最后一段应是 recovery
    assert phase_ids[0] == "baseline"
    assert phase_ids[-1] == "recovery"


def test_phase_boundaries_convenience():
    q1, q2, q3 = phase_boundaries(100)
    assert q1 < q2 < q3


def test_resolve_phase_schedule_string():
    schedule = resolve_phase_schedule("standard_exposure")
    assert schedule is STANDARD_EXPOSURE


def test_resolve_phase_schedule_instance_passthrough():
    schedule = resolve_phase_schedule(STANDARD_EXPOSURE)
    assert schedule is STANDARD_EXPOSURE


@pytest.mark.parametrize(
    "unimplemented",
    ["variable_onset", "fast_transient", "incomplete_recovery", "multi_pulse"],
)
def test_resolve_phase_schedule_unimplemented_raises(unimplemented):
    """v1.2 契约: 未实现 profile 必须 raise NotImplementedError, 不许隐式回退。"""
    with pytest.raises(NotImplementedError, match=unimplemented):
        resolve_phase_schedule(unimplemented)


def test_resolve_phase_schedule_unknown_raises():
    with pytest.raises(NotImplementedError):
        resolve_phase_schedule("totally_unknown_profile")


def test_jittered_preserves_sum():
    rng = random.Random(0)
    jittered = STANDARD_EXPOSURE.jittered(rng, jitter_frac=0.2)
    total = sum(s.duration_frac for s in jittered.segments)
    assert abs(total - 1.0) < 1e-9
    # 段名与 blend_shape 不变
    for orig, new in zip(STANDARD_EXPOSURE.segments, jittered.segments, strict=True):
        assert orig.name == new.name
        assert orig.blend_shape == new.blend_shape


def test_jittered_zero_is_identity():
    rng = random.Random(0)
    jittered = STANDARD_EXPOSURE.jittered(rng, jitter_frac=0.0)
    assert jittered is STANDARD_EXPOSURE


def test_to_dict_roundtrip_keys():
    payload = STANDARD_EXPOSURE.to_dict()
    assert payload["name"] == "standard_exposure"
    segments = payload["segments"]
    assert isinstance(segments, list)
    assert len(segments) == 4
    for seg_dict in segments:
        assert isinstance(seg_dict, dict)
        assert {"name", "duration_frac", "blend_shape", "blend_floor"} <= seg_dict.keys()


def test_timesteps_too_small_raises():
    with pytest.raises(ValueError, match="timesteps"):
        STANDARD_EXPOSURE.boundaries(3)

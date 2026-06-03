import random

from sim.generation.phases import (
    INCOMPLETE_RECOVERY,
    MULTI_PULSE,
    PHASE_SCHEDULES,
    STANDARD_EXPOSURE,
    blend_for_timestep,
    phase_boundaries,
    phase_for_timestep,
    resolve_phase_schedule,
)
from sim.generation.slow import _path_l_m_for_timestep


def test_phase_boundaries_split_sequence_into_four_ordered_regions():
    assert phase_boundaries(8) == (2, 4, 6)
    assert [phase_for_timestep(t, 8) for t in range(8)] == [
        "baseline",
        "baseline",
        "exposure",
        "exposure",
        "steady",
        "steady",
        "recovery",
        "recovery",
    ]


def test_blend_progression_matches_baseline_exposure_steady_recovery():
    assert blend_for_timestep(0, 8) == 0.0
    assert blend_for_timestep(1, 8) == 0.0
    assert blend_for_timestep(2, 8) == 0.5
    assert blend_for_timestep(3, 8) == 1.0
    assert blend_for_timestep(4, 8) == 1.0
    assert blend_for_timestep(6, 8) == 0.5
    assert blend_for_timestep(7, 8) == 0.0


def test_standard_schedule_preserves_legacy_phase_api():
    schedule = resolve_phase_schedule("standard_exposure")

    assert schedule is STANDARD_EXPOSURE
    assert schedule.boundaries(128) == phase_boundaries(128)
    assert [schedule.phase_for_timestep(t, 8) for t in range(8)] == [phase_for_timestep(t, 8) for t in range(8)]
    assert [schedule.blend_for_timestep(t, 8) for t in range(8)] == [blend_for_timestep(t, 8) for t in range(8)]
    assert schedule.to_dict()["name"] == "standard_exposure"


def test_stage_profile_library_contains_long_sequence_profiles():
    assert {"standard_exposure", "variable_onset", "fast_transient", "incomplete_recovery", "multi_pulse"}.issubset(PHASE_SCHEDULES)
    assert [MULTI_PULSE.phase_for_timestep(t, 12) for t in range(12)] == [
        "baseline",
        "exposure",
        "steady",
        "recovery",
        "baseline",
        "exposure",
        "steady",
        "recovery",
        "baseline",
        "exposure",
        "steady",
        "recovery",
    ]


def test_incomplete_recovery_keeps_nonzero_blend_floor():
    assert INCOMPLETE_RECOVERY.blend_for_timestep(7, 8) == 0.2


def test_schedule_jitter_is_seed_reproducible_and_keeps_phase_order():
    first = STANDARD_EXPOSURE.jittered(random.Random(123), 0.2)
    second = STANDARD_EXPOSURE.jittered(random.Random(123), 0.2)

    assert first.to_dict() == second.to_dict()
    assert [segment.name for segment in first.segments] == ["baseline", "exposure", "steady", "recovery"]
    assert abs(sum(segment.duration_frac for segment in first.segments) - 1.0) < 1e-12
    assert first.boundaries(128) != STANDARD_EXPOSURE.boundaries(128)


def test_path_scan_maps_short_phase_to_scan_endpoints():
    path_lms = (0.2, 0.3, 0.4, 0.5, 0.6)

    assert _path_l_m_for_timestep(1.0, 0, 2, 4, 6, True, False, path_lms) == 0.2
    assert _path_l_m_for_timestep(1.0, 1, 2, 4, 6, True, False, path_lms) == 0.6
    assert _path_l_m_for_timestep(1.0, 4, 2, 4, 6, False, True, path_lms) == 0.2
    assert _path_l_m_for_timestep(1.0, 5, 2, 4, 6, False, True, path_lms) == 0.6

from sim.generation.phases import blend_for_timestep, phase_boundaries, phase_for_timestep
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


def test_path_scan_maps_short_phase_to_scan_endpoints():
    path_lms = (0.2, 0.3, 0.4, 0.5, 0.6)

    assert _path_l_m_for_timestep(1.0, 0, 2, 4, 6, True, False, path_lms) == 0.2
    assert _path_l_m_for_timestep(1.0, 1, 2, 4, 6, True, False, path_lms) == 0.6
    assert _path_l_m_for_timestep(1.0, 4, 2, 4, 6, False, True, path_lms) == 0.2
    assert _path_l_m_for_timestep(1.0, 5, 2, 4, 6, False, True, path_lms) == 0.6

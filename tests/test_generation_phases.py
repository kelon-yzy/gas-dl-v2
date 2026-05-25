from sim.generation.phases import blend_for_timestep, phase_boundaries, phase_for_timestep


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

from __future__ import annotations


def phase_boundaries(timesteps: int) -> tuple[int, int, int]:
    if timesteps < 4:
        raise ValueError("timesteps must be >= 4")
    q1 = timesteps // 4
    q2 = timesteps // 2
    q3 = (timesteps * 3) // 4
    return q1, q2, q3


def phase_for_timestep(timestep: int, timesteps: int) -> str:
    q1, q2, q3 = phase_boundaries(timesteps)
    if timestep < q1:
        return "baseline"
    if timestep < q2:
        return "exposure"
    if timestep < q3:
        return "steady"
    return "recovery"


def blend_for_timestep(timestep: int, timesteps: int) -> float:
    q1, q2, q3 = phase_boundaries(timesteps)
    if timestep < q1:
        return 0.0
    if timestep < q2:
        return (timestep - q1 + 1) / max(q2 - q1, 1)
    if timestep < q3:
        return 1.0
    recovery_length = max(1, q3 - q2)
    return max(0.0, 1.0 - ((timestep - q3 + 1) / recovery_length))

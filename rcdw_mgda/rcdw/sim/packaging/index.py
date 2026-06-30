"""sequence_index.csv 行构造。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def build_sequence_index_rows(
    conditions: Iterable[Mapping[str, object]],
    *,
    stage_profile: str,
    timesteps: int,
    dt_s: float,
    status: str = "synthetic_measurement",
) -> list[dict[str, str]]:
    return [
        {
            "sequence_id": str(row["sequence_id"]),
            "mixture_id": str(row["mixture_id"]),
            "stage_profile": stage_profile,
            "status": status,
            "n_timesteps": str(timesteps),
            "dt_s": str(dt_s),
        }
        for row in conditions
    ]

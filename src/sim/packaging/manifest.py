from __future__ import annotations

from sim.core.schema import SCHEMA_VERSION


def build_manifest(
    *,
    dataset_slug: str,
    sequence_count: int,
    seed: int,
    timesteps: int,
    dt_s: float,
    storage: str,
    multi_path_phase: str,
    sampling_strategy: str,
    path_lms: tuple[float, ...],
    optical_absorption_backend: str,
    shapes: dict[str, list[int]],
    slow_channels: tuple[str, ...],
    labels: tuple[str, ...],
    optical_absorption_metadata: dict[str, object] | None = None,
    acoustic_model_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_slug": dataset_slug,
        "sequence_count": int(sequence_count),
        "seed": int(seed),
        "timesteps": int(timesteps),
        "dt_s": float(dt_s),
        "storage": storage,
        "multi_path_phase": multi_path_phase,
        "sampling_strategy": sampling_strategy,
        "path_lms": [float(path_l_m) for path_l_m in path_lms],
        "optical_absorption_backend": optical_absorption_backend,
        "primary_key": "mixture_id",
        "instance_key": "sequence_id",
        "split_group_field": "mixture_id",
        "shapes": shapes,
        "slow_channels": list(slow_channels),
        "labels": list(labels),
    }
    if optical_absorption_metadata is not None:
        manifest.update(optical_absorption_metadata)
    if acoustic_model_metadata is not None:
        manifest.update(acoustic_model_metadata)
    return manifest

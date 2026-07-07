from __future__ import annotations

from tv3.sim.core.schema import SCHEMA_VERSION as DEFAULT_SCHEMA_VERSION


def build_manifest(
    *,
    dataset_slug: str,
    sequence_count: int,
    seed: int,
    timesteps: int,
    dt_s: float,
    storage: str,
    multi_path_phase: str,
    stage_profile: str,
    stage_jitter: float,
    phase_schedule: dict[str, object],
    sampling_strategy: str,
    path_lms: tuple[float, ...],
    optical_absorption_backend: str,
    shapes: dict[str, list[int]],
    slow_channels: tuple[str, ...],
    labels: tuple[str, ...],
    optical_absorption_metadata: dict[str, object] | None = None,
    acoustic_model_metadata: dict[str, object] | None = None,
    sim_revision: dict[str, object] | None = None,
    schema_version: str = DEFAULT_SCHEMA_VERSION,
    composition_scheme: str = "tunnel_ventilation",
    background_fields: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a benchmark manifest.

    composition_scheme:
        "tunnel_ventilation" (default, CO2/O2/N2 sum=100% closure, no residual
        head). Downstream loaders use this to switch label semantics.
    background_fields:
        Components that participate in physics but are not predicted. Empty
        tuple for tunnel_ventilation (all three components are predicted).
    """
    manifest = {
        "schema_version": schema_version,
        "composition_scheme": composition_scheme,
        "dataset_slug": dataset_slug,
        "sequence_count": int(sequence_count),
        "seed": int(seed),
        "timesteps": int(timesteps),
        "dt_s": float(dt_s),
        "storage": storage,
        "multi_path_phase": multi_path_phase,
        "stage_profile": stage_profile,
        "stage_jitter": float(stage_jitter),
        "phase_schedule": phase_schedule,
        "sampling_strategy": sampling_strategy,
        "path_lms": [float(path_l_m) for path_l_m in path_lms],
        "optical_absorption_backend": optical_absorption_backend,
        "primary_key": "mixture_id",
        "instance_key": "sequence_id",
        "split_group_field": "mixture_id",
        "shapes": shapes,
        "slow_channels": list(slow_channels),
        "labels": list(labels),
        "background_fields": list(background_fields),
    }
    if optical_absorption_metadata is not None:
        manifest.update(optical_absorption_metadata)
    if acoustic_model_metadata is not None:
        manifest.update(acoustic_model_metadata)
    if sim_revision is not None:
        manifest["sim_revision"] = sim_revision
    return manifest

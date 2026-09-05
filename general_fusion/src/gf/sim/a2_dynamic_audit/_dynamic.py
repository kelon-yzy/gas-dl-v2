"""动态非退化审计（§10.5 六条判据的实现）。"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from gf.sim.a2_dynamic_dataset import DynamicDataset
from gf.sim.a2_dynamic_audit._shared import FAMILIES, MIN_UNIQUE_QUANTIZED_LEVELS


def _audit_dynamic_non_degenerate(
    dataset: DynamicDataset,
    data_config: Mapping[str, Any],
    experiment_config: Mapping[str, Any] | None,
    *,
    subset_split: str | None = None,
    exclude_pure: bool = False,
) -> dict[str, Any]:
    """动态非退化审计（默认全部观测，冻结审计时限定 test 并剔除 pure）。

    pure 顶点目标等于 purge（§10.5），没有可激励的动力学，必须单列为边界
    审计，不能进入幅值 / t50 统计分母。
    """
    if not isinstance(experiment_config, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires the frozen experiment_config")
    experiment_pilot = experiment_config.get("pilot")
    if not isinstance(experiment_pilot, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires experiment_config.pilot")
    noise_base = np.asarray(
        experiment_pilot["observation_noise_std_by_sensor"],
        dtype=np.float64,
    )
    if noise_base.shape != (len(data_config["sensor_ids"]),) or not np.isfinite(noise_base).all():
        raise ValueError("experiment pilot observation_noise_std_by_sensor is invalid")
    if np.any(noise_base < 0.0):
        raise ValueError("experiment pilot observation noise must be non-negative")
    thresholds = experiment_pilot["dynamic_gate"]
    if not isinstance(thresholds, Mapping):
        raise ValueError("dynamic non-degeneracy audit requires pilot.dynamic_gate")
    minimum_t50_separation = int(thresholds["minimum_t50_separation_samples"])
    if minimum_t50_separation < 0:
        raise ValueError("minimum_t50_separation_samples must be non-negative")
    minimum_transition_variance_ratio = float(thresholds["minimum_transition_variance_ratio"])
    if not math.isfinite(minimum_transition_variance_ratio) or minimum_transition_variance_ratio < 0.0:
        raise ValueError("minimum_transition_variance_ratio must be finite and non-negative")
    noise_by_id = {
        str(profile["noise_profile_id"]): float(profile["white_noise_scale"])
        for profile in data_config["noise_profiles"]
    }
    clean = np.transpose(dataset.clean_device_signals, (0, 2, 1))
    observed = np.transpose(dataset.signals[:, :, :, 0], (0, 2, 1))
    phase_id = dataset.phase_id
    active_fraction: dict[str, float] = {}
    active_fraction_unscaled: dict[str, float] = {}
    quantization_fraction: dict[str, float] = {}
    family_degenerate_fraction: dict[str, float] = {}
    t50_separation_fraction: dict[str, float] = {}
    phase_coverage_fraction: dict[str, float] = {}
    transition_variance_ratio_quantiles: dict[str, dict[str, float]] = {}
    transition_variance_ratio_fraction: dict[str, float] = {}
    family_row_counts: dict[str, int] = {}
    for family in FAMILIES:
        indices = dataset.indices(family=family, split=subset_split)
        if exclude_pure:
            indices = np.asarray(
                [
                    index
                    for index in indices
                    if str(dataset.records[int(index)]["composition_region"]) != "pure"
                ],
                dtype=np.int64,
            )
        family_row_counts[family] = int(indices.size)
        if indices.size == 0:
            raise ValueError(f"dynamic audit family {family!r} has no auditable observations")
        clean_family = clean[indices]
        # F4：判据按该行注册 noise profile 的 white_noise_scale 缩放，
        # NOISE-10X 行不再享受相对放宽 10 倍的名义门。未缩放值同时报告，
        # 便于与历史数值对照。
        row_scales = np.asarray(
            [
                noise_by_id[str(dataset.records[int(index)]["noise_profile_id"])]
                for index in indices
            ],
            dtype=np.float64,
        )
        p2p = np.ptp(clean_family, axis=1)
        active = p2p > (5.0 * noise_base)[None, :] * row_scales[:, None]
        active_unscaled = p2p > (5.0 * noise_base)[None, :]
        active_count = np.sum(active, axis=1)
        active_count_unscaled = np.sum(active_unscaled, axis=1)
        active_fraction[family] = float(np.mean(active_count >= 2))
        active_fraction_unscaled[family] = float(np.mean(active_count_unscaled >= 2))
        quantized = np.asarray(
            [
                np.min(
                    [
                        np.unique(observed[index, :, channel]).size
                        for channel in range(observed.shape[2])
                    ]
                )
                >= MIN_UNIQUE_QUANTIZED_LEVELS
                for index in indices
            ],
            dtype=bool,
        )
        quantization_fraction[family] = float(np.mean(quantized))
        # F6 第 4 条：四阶段均非空；任一为空判该行退化。
        transition_phase_index = _phase_index(data_config, "transition")
        phase_counts = np.stack(
            [
                np.bincount(
                    phase_id[int(index)].astype(np.int64),
                    minlength=len(data_config["phases"]),
                )
                for index in indices
            ],
            axis=0,
        )
        phase_covered = np.all(phase_counts > 0, axis=1)
        phase_coverage_fraction[family] = float(np.mean(phase_covered))
        # F6 第 5 条：clean signal 的 transition 方差不能全部由白噪声解释
        # （§10.5 原文语义）。逐通道计算方差比，任一通道的比值达到
        # minimum_transition_variance_ratio 即通过；binary 无 CO₂ 等组成
        # 使单通道恒定属合法动态，不判退化。比值分位数仍逐通道报告。
        transition_variance_ratio = _transition_variance_ratio(
            clean_family,
            phase_id,
            indices,
            transition_phase_index,
            noise_base=noise_base,
            row_scales=row_scales,
        )
        transition_variance_ratio_quantiles[family] = {
            "p05": [float(x) for x in np.percentile(transition_variance_ratio, 5, axis=0)],
            "p50": [float(x) for x in np.percentile(transition_variance_ratio, 50, axis=0)],
            "p95": [float(x) for x in np.percentile(transition_variance_ratio, 95, axis=0)],
        }
        transition_variance_pass = (
            transition_variance_ratio >= minimum_transition_variance_ratio
        ).any(axis=1)
        transition_variance_ratio_fraction[family] = float(
            np.mean(transition_variance_pass)
        )
        degenerate = (
            (active_count < 2)
            | (~quantized)
            | (~phase_covered)
            | (~transition_variance_pass)
        )
        family_degenerate_fraction[family] = float(np.mean(degenerate))
        t50_separation_fraction[family] = _t50_separation_fraction(
            clean_family,
            min_separation_samples=minimum_t50_separation,
        )
    minimum_active = float(thresholds["minimum_active_channel_fraction"])
    minimum_quantized = float(thresholds["minimum_quantized_level_fraction"])
    minimum_t50 = float(thresholds["minimum_t50_pair_fraction"])
    maximum_degenerate = float(thresholds["maximum_family_degenerate_fraction"])
    total_rows = sum(family_row_counts.values())
    global_active_fraction = sum(
        active_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_active_fraction_unscaled = sum(
        active_fraction_unscaled[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_quantized_fraction = sum(
        quantization_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_t50_fraction = sum(
        t50_separation_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_phase_coverage = sum(
        phase_coverage_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    global_transition_variance_ratio_fraction = sum(
        transition_variance_ratio_fraction[family] * family_row_counts[family]
        for family in FAMILIES
    ) / total_rows
    minimum_transition_variance_pass_fraction = float(
        thresholds["minimum_transition_variance_ratio_pass_fraction"]
    )
    checks = {
        "active_channel_fraction": global_active_fraction >= minimum_active,
        "quantized_level_fraction": global_quantized_fraction >= minimum_quantized,
        "t50_pair_fraction": global_t50_fraction >= minimum_t50,
        "phase_coverage": global_phase_coverage >= 1.0,
        "transition_variance_ratio": (
            global_transition_variance_ratio_fraction
            >= minimum_transition_variance_pass_fraction
        ),
        "family_degenerate_fraction": max(family_degenerate_fraction.values()) <= maximum_degenerate,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "active_channel_fraction": active_fraction,
        "global_active_channel_fraction": global_active_fraction,
        "active_channel_fraction_unscaled": active_fraction_unscaled,
        "global_active_channel_fraction_unscaled": global_active_fraction_unscaled,
        "quantized_level_fraction": quantization_fraction,
        "global_quantized_level_fraction": global_quantized_fraction,
        "t50_pair_fraction": t50_separation_fraction,
        "global_t50_pair_fraction": global_t50_fraction,
        "configured_minimum_t50_separation_samples": minimum_t50_separation,
        "phase_coverage_fraction": phase_coverage_fraction,
        "global_phase_coverage_fraction": global_phase_coverage,
        "transition_variance_ratio_quantiles": transition_variance_ratio_quantiles,
        "transition_variance_ratio_fraction": transition_variance_ratio_fraction,
        "global_transition_variance_ratio_fraction": global_transition_variance_ratio_fraction,
        "configured_minimum_transition_variance_ratio": minimum_transition_variance_ratio,
        "configured_minimum_transition_variance_ratio_pass_fraction": minimum_transition_variance_pass_fraction,
        "family_degenerate_fraction": family_degenerate_fraction,
    }


def _phase_index(data_config: Mapping[str, Any], phase_name: str) -> int:
    names = [str(phase["phase_id"]) for phase in data_config["phases"]]
    if phase_name not in names:
        raise ValueError(f"phase {phase_name!r} is not registered in data_config.phases")
    return names.index(phase_name)


def _transition_variance_ratio(
    clean_family: np.ndarray,
    phase_id: np.ndarray,
    indices: np.ndarray,
    transition_phase_index: int,
    *,
    noise_base: np.ndarray,
    row_scales: np.ndarray,
) -> np.ndarray:
    """clean transition 段方差与该行白噪方差之比（每行每通道）。

    第 5 条要求 clean signal 的 transition 方差不能全部由白噪声解释：
    比值是每通道单独计算的最小统计基础，报告分位数并设最小门限。
    """

    ratios = np.empty((len(indices), clean_family.shape[2]), dtype=np.float64)
    for position, index in enumerate(np.asarray(indices, dtype=np.int64)):
        transition_mask = phase_id[int(index)] == transition_phase_index
        if not np.any(transition_mask):
            ratios[position] = 0.0
            continue
        segment = clean_family[position][transition_mask]
        variance = np.var(segment, axis=0)
        noise_variance = (noise_base * row_scales[position]) ** 2
        ratios[position] = variance / np.maximum(noise_variance, 1.0e-30)
    return ratios


def _t50_separation_fraction(clean: np.ndarray, *, min_separation_samples: int) -> float:
    if clean.ndim != 3 or clean.shape[0] == 0:
        raise ValueError("clean family signals must have shape (N,T,3)")
    fractions: list[bool] = []
    for sequence in clean:
        t50 = [_t50_index(sequence[:, channel]) for channel in (1, 2)]
        fractions.append(
            t50[0] is not None
            and t50[1] is not None
            and abs(t50[0] - t50[1]) >= min_separation_samples
        )
    return float(np.mean(fractions))


def _t50_index(values: np.ndarray) -> int | None:
    baseline = float(values[0])
    excursion = values - baseline
    endpoint = float(values[int(np.argmax(np.abs(excursion)))])
    delta = endpoint - baseline
    if abs(delta) <= 0.0:
        return None
    threshold = baseline + 0.5 * delta
    if delta > 0.0:
        candidates = np.flatnonzero(values >= threshold)
    else:
        candidates = np.flatnonzero(values <= threshold)
    return int(candidates[0]) if candidates.size else None

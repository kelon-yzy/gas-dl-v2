"""The unique Raw-to-DSP derivation chain."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..common.io import sha256_bytes
from ..contract import validate_dsp_provenance


def dsp_config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def derive_dsp(raw_waveform: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    raw = np.asarray(raw_waveform, dtype=np.float64)
    if raw.ndim != 2 or not np.all(np.isfinite(raw)):
        raise ValueError("raw_waveform must be a finite [channel, time] array")
    frame_length = int(config["frame_length"])
    hop_length = int(config["hop_length"])
    if frame_length <= 1 or hop_length <= 0 or raw.shape[1] < frame_length:
        raise ValueError("invalid DSP frame configuration")
    starts = range(0, raw.shape[1] - frame_length + 1, hop_length)
    time_axis = np.arange(frame_length, dtype=np.float64)
    centered_time = time_axis - np.mean(time_axis)
    slope_denominator = float(centered_time @ centered_time)
    frames = []
    for start in starts:
        frame = raw[:, start : start + frame_length]
        means = np.mean(frame, axis=1)
        standard_deviations = np.std(frame, axis=1)
        slopes = (frame @ centered_time) / slope_denominator
        frames.append(np.concatenate([means, standard_deviations, slopes]))
    return np.asarray(frames, dtype=np.float64)


def build_dsp_provenance(
    *,
    source_raw_manifest_id: str,
    raw_manifest_sha256: str,
    dsp_config_sha256_value: str,
    code_sha256: str,
) -> dict[str, object]:
    provenance = {
        "source_raw_manifest_id": source_raw_manifest_id,
        "raw_manifest_sha256": raw_manifest_sha256,
        "dsp_config_sha256": dsp_config_sha256_value,
        "code_sha256": code_sha256,
        "derived_from": ["raw_waveform"],
    }
    validate_dsp_provenance(
        provenance,
        raw_manifest_sha256=raw_manifest_sha256,
        dsp_config_sha256=dsp_config_sha256_value,
        code_sha256=code_sha256,
    )
    return provenance


__all__ = ["build_dsp_provenance", "derive_dsp", "dsp_config_sha256"]

"""通道吸光度积分：Σ filter * exp(-OD) dν / Σ filter dν（等价 HG 主线）。"""

from __future__ import annotations

import math

import numpy as np


def integrate_channel_absorbance(
    *,
    wavenumber_cm1: np.ndarray,
    optical_depth: np.ndarray,
    filter_response: np.ndarray,
) -> dict[str, float]:
    _validate_same_shape(wavenumber_cm1, optical_depth, filter_response)
    filter_area = float(np.trapezoid(filter_response, wavenumber_cm1))
    if filter_area <= 0.0:
        raise ValueError("filter_response integral must be > 0")
    transmittance = np.exp(-optical_depth)
    channel_transmittance = float(
        np.trapezoid(filter_response * transmittance, wavenumber_cm1) / filter_area
    )
    if channel_transmittance <= 0.0:
        raise ValueError("channel transmittance must be > 0")
    return {
        "absorbance_observed": -math.log(channel_transmittance),
        "transmittance_channel": channel_transmittance,
        "filter_area": filter_area,
    }


def _validate_same_shape(*arrays: np.ndarray) -> None:
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(
            "wavenumber_cm1, optical_depth, and filter_response must have the same shape"
        )
    if arrays[0].ndim != 1:
        raise ValueError("spectral arrays must be one-dimensional")
    if arrays[0].size < 2:
        raise ValueError("spectral arrays must contain at least two samples")

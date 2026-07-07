from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NDIRFilter:
    channel: str
    center_cm1: float
    fwhm_cm1: float


def gaussian_filter(wavenumber_cm1: np.ndarray, spec: NDIRFilter) -> np.ndarray:
    if spec.fwhm_cm1 <= 0.0:
        raise ValueError("fwhm_cm1 must be > 0")
    sigma = spec.fwhm_cm1 / 2.3548200450309493
    response = np.exp(-0.5 * ((wavenumber_cm1 - spec.center_cm1) / sigma) ** 2)
    area = float(np.trapezoid(response, wavenumber_cm1))
    if area <= 0.0:
        raise ValueError("filter response integral must be > 0")
    return response.astype(np.float64)

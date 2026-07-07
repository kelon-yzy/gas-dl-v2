from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpticalCrosstalkSpec:
    ch4_channel_co2_response: float = 0.035
    co2_channel_ch4_response: float = 0.012


DEFAULT_OPTICAL_CROSSTALK_SPEC = OpticalCrosstalkSpec()


def apply_optical_crosstalk(
    *,
    absorption_ch4: float,
    absorption_co2: float,
    spec: OpticalCrosstalkSpec = DEFAULT_OPTICAL_CROSSTALK_SPEC,
) -> dict[str, float]:
    ch4_cross_from_co2 = spec.ch4_channel_co2_response * absorption_co2
    co2_cross_from_ch4 = spec.co2_channel_ch4_response * absorption_ch4
    return {
        "absorption_ch4_true": absorption_ch4,
        "absorption_co2_true": absorption_co2,
        "absorption_ch4_cross_from_co2": ch4_cross_from_co2,
        "absorption_co2_cross_from_ch4": co2_cross_from_ch4,
        "absorption_ch4_observed": absorption_ch4 + ch4_cross_from_co2,
        "absorption_co2_observed": absorption_co2 + co2_cross_from_ch4,
    }

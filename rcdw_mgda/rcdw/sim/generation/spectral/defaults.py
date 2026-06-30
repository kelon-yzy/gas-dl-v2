"""RCDW spectral 默认配置加载。

从 ``rcdw_mgda/configs/spectral-defaults.json`` 加载 RCDW 专用配置，
仅注册 (CO2, H2O) 两种气体和 co2 单通道。与 HG 主线完全隔离。
"""

from __future__ import annotations

import json
from pathlib import Path

from rcdw.sim.generation.spectral.filters import NDIRFilter
from rcdw.sim.generation.spectral.hitran_backend import HitranGasSpec, HitranGridSpec


# rcdw_mgda/configs/spectral-defaults.json: 本文件位于
# rcdw_mgda/rcdw/sim/generation/spectral/defaults.py
# parents[0]=spectral, [1]=generation, [2]=sim, [3]=rcdw, [4]=rcdw_mgda
SPECTRAL_DEFAULTS_CONFIG_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "spectral-defaults.json"
)
SPECTRAL_DEFAULTS_PAYLOAD = json.loads(
    SPECTRAL_DEFAULTS_CONFIG_PATH.read_text(encoding="utf-8")
)

# RCDW 仅注册 (CO2, H2O)。删除 CH4（RCDW 无 CH4 组分）。
DEFAULT_HITRAN_GAS_SPECS = tuple(
    HitranGasSpec(
        gas=str(spec["gas"]),
        table_name=str(spec["table_name"]),
        molecule_id=int(spec["molecule_id"]),
        isotopologue_id=int(spec["isotopologue_id"]),
    )
    for spec in SPECTRAL_DEFAULTS_PAYLOAD["gas_specs"]
)

# RCDW 仅注册 co2 通道。get_default_ndir_filter("ch4") 应抛 ValueError。
DEFAULT_NDIR_FILTERS = {
    channel: NDIRFilter(
        channel=str(spec["channel"]),
        center_cm1=float(spec["center_cm1"]),
        fwhm_cm1=float(spec["fwhm_cm1"]),
    )
    for channel, spec in SPECTRAL_DEFAULTS_PAYLOAD["filters"].items()
}

DEFAULT_HITRAN_GRID_SPECS = {
    channel: HitranGridSpec(
        wavenumber_min_cm1=float(spec["wavenumber_min_cm1"]),
        wavenumber_max_cm1=float(spec["wavenumber_max_cm1"]),
        wavenumber_step_cm1=float(spec["wavenumber_step_cm1"]),
        temperature_k=float(spec["temperature_k"]),
        pressure_atm=float(spec["pressure_atm"]),
    )
    for channel, spec in SPECTRAL_DEFAULTS_PAYLOAD["hitran_grids"].items()
}


def get_default_ndir_filter(channel: str) -> NDIRFilter:
    """获取 channel 对应的 NDIR 滤波器。

    RCDW 仅注册 co2 通道，请求 ch4 等会抛 ValueError。
    """
    try:
        return DEFAULT_NDIR_FILTERS[channel.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown NDIR channel: {channel!r}. "
            f"RCDW 仅支持 {list(DEFAULT_NDIR_FILTERS)} (O2/N2 在中红外无吸收,不提供 NDIR 通道)。"
        ) from exc


def get_default_hitran_grid(channel: str) -> HitranGridSpec:
    try:
        return DEFAULT_HITRAN_GRID_SPECS[channel.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown HITRAN grid channel: {channel!r}. "
            f"Available: {list(DEFAULT_HITRAN_GRID_SPECS)}"
        ) from exc

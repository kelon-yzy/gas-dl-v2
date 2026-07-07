from __future__ import annotations

import json
from pathlib import Path

from sg.sim.generation.spectral.filters import NDIRFilter
from sg.sim.generation.spectral.hitran_backend import HitranGasSpec, HitranGridSpec


SPECTRAL_DEFAULTS_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "data" / "spectral-defaults.json"
SPECTRAL_DEFAULTS_PAYLOAD = json.loads(SPECTRAL_DEFAULTS_CONFIG_PATH.read_text(encoding="utf-8"))

DEFAULT_HITRAN_GAS_SPECS = tuple(
    HitranGasSpec(
        gas=str(spec["gas"]),
        table_name=str(spec["table_name"]),
        molecule_id=int(spec["molecule_id"]),
        isotopologue_id=int(spec["isotopologue_id"]),
    )
    for spec in SPECTRAL_DEFAULTS_PAYLOAD["gas_specs"]
)

# Industry-reference placeholders, not the actual datasheet of the target
# TraceGas-HC-NDIR series sensor (Shenzhen Trace Gas Sensing Technology Co., Ltd.).
# ch4: InfraTec LIM-262 NBP filter, CWL 3.3 um, FWHM 160 nm (~147 cm-1 at 3.3 um).
#      MDPI Sensors 2012, doi:10.3390/s120912729.
# co2: InfraTec standard CO2 NBP filter, CWL 4.26-4.27 um, HPBW 170 nm
#      (~93 cm-1 at 4.26 um). InfraTec gas analysis docs (infratec-infrared.com).
# See configs/data/spectral-defaults.json -> filter_source for the same record.
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


# Syngas (4-component H2/CH4/CO2/CO + N2 background) HITRAN configuration.
# Kept separate from hg defaults so that hg DEFAULT_HITRAN_GAS_SPECS stays
# (CH4, CO2, H2O) and hg cache requirements do not expand to require CO cache.
_SYNGAS_BLOCK = SPECTRAL_DEFAULTS_PAYLOAD.get("syngas", {})

_SYNGAS_EXTRA_GAS_SPECS = tuple(
    HitranGasSpec(
        gas=str(spec["gas"]),
        table_name=str(spec["table_name"]),
        molecule_id=int(spec["molecule_id"]),
        isotopologue_id=int(spec["isotopologue_id"]),
    )
    for spec in _SYNGAS_BLOCK.get("gas_specs", [])
)

# Union of hg gas specs and syngas-only extras (e.g. CO), indexed by gas name
# for fast per-channel lookup.
SYNGAS_HITRAN_GAS_SPECS_BY_NAME = {
    spec.gas: spec for spec in (*DEFAULT_HITRAN_GAS_SPECS, *_SYNGAS_EXTRA_GAS_SPECS)
}

SYNGAS_NDIR_FILTERS = dict(DEFAULT_NDIR_FILTERS) | {
    channel: NDIRFilter(
        channel=str(spec["channel"]),
        center_cm1=float(spec["center_cm1"]),
        fwhm_cm1=float(spec["fwhm_cm1"]),
    )
    for channel, spec in _SYNGAS_BLOCK.get("filters", {}).items()
}

SYNGAS_HITRAN_GRID_SPECS = dict(DEFAULT_HITRAN_GRID_SPECS) | {
    channel: HitranGridSpec(
        wavenumber_min_cm1=float(spec["wavenumber_min_cm1"]),
        wavenumber_max_cm1=float(spec["wavenumber_max_cm1"]),
        wavenumber_step_cm1=float(spec["wavenumber_step_cm1"]),
        temperature_k=float(spec["temperature_k"]),
        pressure_atm=float(spec["pressure_atm"]),
    )
    for channel, spec in _SYNGAS_BLOCK.get("hitran_grids", {}).items()
}

# Channel -> tuple of HitranGasSpec. Drives which gases participate in each
# NDIR channel's forward model and which cache files must be precomputed.
SYNGAS_CHANNEL_GAS_SPECS = {
    channel: tuple(SYNGAS_HITRAN_GAS_SPECS_BY_NAME[gas_name] for gas_name in gas_names)
    for channel, gas_names in _SYNGAS_BLOCK.get("channel_gas_specs", {}).items()
}


def get_default_ndir_filter(channel: str) -> NDIRFilter:
    try:
        return DEFAULT_NDIR_FILTERS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown NDIR channel: {channel!r}. Available: {list(DEFAULT_NDIR_FILTERS)}") from exc


def get_default_hitran_grid(channel: str) -> HitranGridSpec:
    try:
        return DEFAULT_HITRAN_GRID_SPECS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown HITRAN grid channel: {channel!r}. Available: {list(DEFAULT_HITRAN_GRID_SPECS)}") from exc


def get_syngas_ndir_filter(channel: str) -> NDIRFilter:
    try:
        return SYNGAS_NDIR_FILTERS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown syngas NDIR channel: {channel!r}. Available: {list(SYNGAS_NDIR_FILTERS)}") from exc


def get_syngas_hitran_grid(channel: str) -> HitranGridSpec:
    try:
        return SYNGAS_HITRAN_GRID_SPECS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown syngas HITRAN grid channel: {channel!r}. Available: {list(SYNGAS_HITRAN_GRID_SPECS)}") from exc


def get_syngas_channel_gas_specs(channel: str) -> tuple[HitranGasSpec, ...]:
    try:
        return SYNGAS_CHANNEL_GAS_SPECS[channel.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown syngas channel: {channel!r}. Available: {list(SYNGAS_CHANNEL_GAS_SPECS)}") from exc

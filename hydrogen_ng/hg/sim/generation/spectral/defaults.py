from __future__ import annotations

import json
from pathlib import Path

from hg.sim.generation.spectral.filters import NDIRFilter
from hg.sim.generation.spectral.hitran_backend import HitranGasSpec, HitranGridSpec


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

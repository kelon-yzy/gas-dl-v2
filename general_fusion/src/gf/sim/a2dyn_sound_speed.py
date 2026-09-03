"""Formal A2-DYN sound-speed routing with CoolProp HEOS as the generator."""

from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from threading import local
from typing import Mapping

from .ar_he_co2 import sound_speed_for_model as historical_sound_speed_for_model

DIRECT_HEOS_SOUND_SPEED_MODEL_ID = "a2dyn_direct_multifluid_eos_v1"
_MODEL_ASSET_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data"
    / "a2dyn_direct_heos_v1.json"
)
_THREAD_STATE = local()


@lru_cache(maxsize=1)
def _model_asset() -> dict[str, object]:
    payload = json.loads(_MODEL_ASSET_PATH.read_text(encoding="utf-8"))
    if payload.get("model_id") != DIRECT_HEOS_SOUND_SPEED_MODEL_ID:
        raise RuntimeError("A2-DYN direct HEOS model asset has an unexpected model_id")
    return payload


@lru_cache(maxsize=1)
def coolprop_runtime_identity() -> dict[str, str]:
    try:
        import CoolProp
        import CoolProp.CoolProp as coolprop_core
    except ImportError as exc:
        raise RuntimeError(
            "A2-DYN direct HEOS requires the pinned CoolProp runtime"
        ) from exc

    package = _model_asset()["package"]
    if not isinstance(package, dict):
        raise RuntimeError("A2-DYN direct HEOS model asset package entry is invalid")
    expected_version = str(package["version"])
    if CoolProp.__version__ != expected_version:
        raise RuntimeError(
            "CoolProp version mismatch: "
            f"expected {expected_version}, got {CoolProp.__version__}"
        )

    binary_path = Path(coolprop_core.__file__).resolve()
    binary_sha256 = hashlib.sha256(binary_path.read_bytes()).hexdigest()
    expected_sha256 = str(package["binary_sha256"])
    if binary_sha256 != expected_sha256:
        raise RuntimeError(
            "CoolProp binary hash mismatch: "
            f"expected {expected_sha256}, got {binary_sha256}"
        )
    return {
        "package": "CoolProp",
        "version": CoolProp.__version__,
        "binary_sha256": binary_sha256,
        "source_revision": str(package["source_revision"]),
    }


def _validated_inputs(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> tuple[list[float], float, float]:
    asset = _model_asset()
    component_order = tuple(asset["component_order"])
    if set(mole_fractions) != set(component_order):
        raise ValueError(
            "mole_fractions must contain exactly the components "
            f"{component_order}"
        )

    fractions = [float(mole_fractions[name]) for name in component_order]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in fractions):
        raise ValueError("mole fractions must be finite and within [0, 1]")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mole fractions must sum to 1 within an absolute tolerance of 1e-12")

    temperature = float(temperature_k)
    pressure = float(pressure_pa)
    temperature_range = asset["temperature_range_k"]
    pressure_range = asset["pressure_range_pa"]
    if not (
        math.isfinite(temperature)
        and float(temperature_range[0]) <= temperature <= float(temperature_range[1])
    ):
        raise ValueError(
            "temperature_k is outside the formal A2-DYN range "
            f"[{temperature_range[0]}, {temperature_range[1]}] K"
        )
    if not (
        math.isfinite(pressure)
        and float(pressure_range[0]) <= pressure <= float(pressure_range[1])
    ):
        raise ValueError(
            "pressure_pa is outside the formal A2-DYN range "
            f"[{pressure_range[0]}, {pressure_range[1]}] Pa"
        )
    return fractions, temperature, pressure


def _state_for_composition(fractions: list[float]):
    import CoolProp

    asset = _model_asset()
    component_order = tuple(asset["component_order"])
    fluid_name_map = asset["fluid_name_map"]
    nonzero = tuple(index for index, value in enumerate(fractions) if value > 0.0)
    state_key = component_order[nonzero[0]] if len(nonzero) == 1 else "ternary"
    cache = getattr(_THREAD_STATE, "states", None)
    if cache is None:
        cache = {}
        _THREAD_STATE.states = cache
    state = cache.get(state_key)
    if state is None:
        if len(nonzero) == 1:
            fluids = str(fluid_name_map[state_key])
        else:
            fluids = "&".join(str(fluid_name_map[name]) for name in component_order)
        state = CoolProp.AbstractState(str(asset["backend"]), fluids)
        state.specify_phase(CoolProp.iphase_gas)
        cache[state_key] = state
    if len(nonzero) > 1:
        state.set_mole_fractions(fractions)
    return state


def direct_multifluid_heos_sound_speed(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
) -> float:
    """Return the pinned CoolProp HEOS gas-phase speed of sound in m/s."""

    import CoolProp

    coolprop_runtime_identity()
    fractions, temperature, pressure = _validated_inputs(
        mole_fractions,
        temperature_k,
        pressure_pa,
    )
    state = _state_for_composition(fractions)
    state.update(CoolProp.PT_INPUTS, pressure, temperature)
    value = float(state.speed_sound())
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"CoolProp HEOS returned an invalid speed of sound: {value}")
    return value


def a2dyn_sound_speed_for_model(
    mole_fractions: Mapping[str, float],
    temperature_k: float,
    pressure_pa: float,
    *,
    model_id: str,
) -> float:
    """Route formal HEOS and preserved historical A1/A2-DYN models explicitly."""

    if model_id == DIRECT_HEOS_SOUND_SPEED_MODEL_ID:
        return direct_multifluid_heos_sound_speed(
            mole_fractions,
            temperature_k,
            pressure_pa,
        )
    return historical_sound_speed_for_model(
        mole_fractions,
        temperature_k,
        pressure_pa,
        model_id=model_id,
    )

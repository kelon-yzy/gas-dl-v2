"""Ar-He-CO2 数据生成、物理模型、打包和质量检查。

公开符号保持兼容，但按需导入，避免物理生成进程无条件加载全部数据集与训练依赖。
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "gf.sim.ar_he_co2": (
        "A1_SOUND_SPEED_MODEL_ID",
        "A2DYN_COEFFICIENT_VERSION",
        "A2DYN_SOUND_SPEED_MODEL_ID",
        "PilotCondition",
        "PilotRecord",
        "a2dyn_cp_t_virial_sound_speed",
        "a2dyn_ideal_heat_capacity",
        "a2dyn_mixture_virial",
        "a2dyn_thermodynamic_state",
        "build_pilot_record",
        "sound_speed_for_model",
    ),
    "gf.sim.a1_dataset": (
        "A1Condition",
        "A1Dataset",
        "A1PhysicsConfig",
        "DEFAULT_A1_PHYSICS",
        "assign_a1_splits",
        "deterministic_signal_vector",
        "generate_a1_conditions",
        "generate_dataset",
        "load_dataset",
        "load_dataset_splits",
    ),
    "gf.sim.a2m_dataset": (
        "A2M_AXES",
        "A2M_DATA_VERSION_PREFIX",
        "A2MObservation",
        "A2MDataset",
        "A2M_PRIMARY_AXES",
        "A2M_SCHEMA_VERSION",
        "A2MTestLockError",
        "compute_a2m_split_hash",
        "generate_a2m_formal_holdout",
        "load_a2m_dataset",
        "validate_a2m_data_config",
    ),
    "gf.sim.a2h_dataset": (
        "A2HCondition",
        "A2HDataset",
        "A2HObservation",
        "A2HPhysicsConfig",
        "CalibrationProfile",
        "NoiseProfile",
        "composition_region",
        "compute_split_family_hash",
        "deterministic_a2h_signal_vector",
        "generate_a2h_dataset",
        "load_a2h_dataset",
        "nominal_signal_parity",
    ),
    "gf.sim.a2_dynamic_physics": (
        "DynamicPhysicsError",
        "DynamicTransportLayers",
        "PhysicsAuditError",
        "analytic_exponential_update",
        "apply_observation_chain",
        "audit_coolprop_sound_speed_grid",
        "build_inlet_composition",
        "evaluate_shared_physics",
        "generate_ar1_noise",
        "generate_shared_noise",
        "protocol_inlet_coefficient",
        "quantize_signal",
        "simulate_dynamic_layers",
        "simulate_local_transport",
        "simulate_well_mixed_chamber",
    ),
    "gf.sim.a2dyn_sound_speed": (
        "DIRECT_HEOS_SOUND_SPEED_MODEL_ID",
        "a2dyn_sound_speed_for_model",
        "coolprop_runtime_identity",
        "direct_multifluid_heos_sound_speed",
    ),
    "gf.sim.a2_sensor_devices": (
        "NDIRDeviceProfile",
        "NDIRSimulationResult",
        "SensorDeviceError",
        "TCDSimulationResult",
        "ThermalDeviceProfile",
        "UltrasonicAcquisitionProfile",
        "UltrasonicAcquisitionResult",
        "UltrasonicLockError",
        "acquire_ultrasonic_tof",
        "estimate_ultrasonic_tof_series",
        "simulate_ndir",
        "simulate_tcd",
        "ultrasonic_signal_amplitude",
    ),
}
_EXPORT_MODULE = {
    name: module_name
    for module_name, names in _MODULE_EXPORTS.items()
    for name in names
}
__all__ = list(_EXPORT_MODULE)


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

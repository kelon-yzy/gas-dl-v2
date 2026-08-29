from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gf.dl.adapters.base import AdapterError
from gf.dl.contracts import UnifiedSample
from gf.sim.ar_he_co2 import PilotCondition, SENSOR_TYPES, build_pilot_record


class ArHeCO2Adapter:
    dataset_id = "ar_he_co2"

    def __init__(self, *, conditions: Sequence[Mapping[str, object]], timesteps: int, dt_s: float) -> None:
        if not conditions:
            raise AdapterError("Ar-He-CO2 smoke config must contain at least one condition")
        self._conditions = tuple(dict(condition) for condition in conditions)
        self._timesteps = timesteps
        self._dt_s = dt_s

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "ArHeCO2Adapter":
        if config.get("dataset_id") != cls.dataset_id:
            raise AdapterError(f"expected dataset_id {cls.dataset_id!r}, got {config.get('dataset_id')!r}")
        conditions = config.get("conditions")
        if not isinstance(conditions, list):
            raise AdapterError("conditions must be a list")
        try:
            timesteps = int(config["timesteps"])
            dt_s = float(config["dt_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AdapterError("timesteps and dt_s must be valid numeric values") from exc
        return cls(conditions=conditions, timesteps=timesteps, dt_s=dt_s)

    def load_samples(self) -> list[UnifiedSample]:
        samples: list[UnifiedSample] = []
        seen_mixture_ids: set[str] = set()
        for raw_condition in self._conditions:
            try:
                mixture_id = raw_condition["mixture_id"]
                split = raw_condition["split"]
                if not isinstance(mixture_id, str) or not mixture_id:
                    raise ValueError("mixture_id must be a non-empty string")
                if not isinstance(split, str):
                    raise ValueError("split must be a string")
                condition = PilotCondition(
                    mixture_id=mixture_id,
                    x_ar_pct=float(raw_condition["x_Ar_pct"]),
                    x_he_pct=float(raw_condition["x_He_pct"]),
                    x_co2_pct=float(raw_condition["x_CO2_pct"]),
                    split=split,
                )
                record = build_pilot_record(condition, timesteps=self._timesteps, dt_s=self._dt_s)
            except (KeyError, TypeError, ValueError) as exc:
                raise AdapterError(f"invalid Ar-He-CO2 condition {raw_condition!r}: {exc}") from exc
            if condition.mixture_id in seen_mixture_ids:
                raise AdapterError(f"duplicate mixture_id {condition.mixture_id!r}")
            seen_mixture_ids.add(condition.mixture_id)

            sensor_ids = tuple(SENSOR_TYPES)
            signals = tuple(record.signals[sensor_id] for sensor_id in sensor_ids)
            valid_mask = tuple(np.ones_like(signal, dtype=np.bool_) for signal in signals)
            quality = tuple(np.ones(signal.shape[0], dtype=np.float32) for signal in signals)
            time = tuple(record.time_s for _ in sensor_ids)
            target = np.array(
                [condition.x_ar_pct, condition.x_he_pct, condition.x_co2_pct],
                dtype=np.float32,
            )
            samples.append(
                UnifiedSample(
                    signals=signals,
                    sensor_id=sensor_ids,
                    sensor_type=tuple(SENSOR_TYPES[sensor_id] for sensor_id in sensor_ids),
                    valid_mask=valid_mask,
                    quality=quality,
                    time=time,
                    target=target,
                    target_mask=np.ones(3, dtype=np.bool_),
                    group_id=condition.mixture_id,
                    dataset_id=self.dataset_id,
                    metadata={
                        "mixture_id": condition.mixture_id,
                        "split": condition.split,
                        "x_Ar_pct": condition.x_ar_pct,
                        "x_He_pct": condition.x_he_pct,
                        "x_CO2_pct": condition.x_co2_pct,
                    },
                )
            )
        return samples

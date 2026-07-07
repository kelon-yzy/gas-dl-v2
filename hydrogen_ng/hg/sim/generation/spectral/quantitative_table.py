from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sim.generation.spectral.tabulated_backend import TabulatedSpectrum


QUANTITATIVE_TABLE_BACKEND = "quantitative_table_v1"
SUPPORTED_QUANTITATIVE_UNITS = ("per_percent_m", "per_fraction_m", "per_ppm_m")
DEFAULT_WAVENUMBER_COLUMN = "wavenumber_cm1"
DEFAULT_COEFF_COLUMN = "absorption_coeff"


def load_quantitative_spectrum_csv(
    path: Path | str,
    *,
    gas: str,
    unit: str,
    source_version: str | None = None,
    wavenumber_column: str = DEFAULT_WAVENUMBER_COLUMN,
    coeff_column: str = DEFAULT_COEFF_COLUMN,
) -> TabulatedSpectrum:
    """Load a quantitative IR spectrum CSV into the shared tabulated contract.

    The CSV is intentionally small and explicit: one wavenumber column and one
    coefficient column. Unit conversion is controlled by ``unit`` so PNNL/NIST
    files can be adapted without guessing silently.
    """
    table_path = Path(path)
    rows = _read_numeric_rows(
        table_path,
        wavenumber_column=wavenumber_column,
        coeff_column=coeff_column,
    )
    wavenumber = np.asarray([row[0] for row in rows], dtype=np.float64)
    coeff = np.asarray([row[1] for row in rows], dtype=np.float64)
    _validate_spectrum_arrays(wavenumber, coeff, source=str(table_path))

    order = np.argsort(wavenumber)
    wavenumber = wavenumber[order]
    coeff = coeff[order]
    if np.any(np.diff(wavenumber) <= 0.0):
        raise ValueError("wavenumber_cm1 values must be unique")

    return TabulatedSpectrum(
        gas=gas.upper(),
        wavenumber_cm1=wavenumber,
        absorption_coeff_per_percent_m=convert_quantitative_coeff_to_per_percent_m(coeff, unit=unit),
        source_version=source_version or f"{QUANTITATIVE_TABLE_BACKEND}:{table_path.name}:{unit}",
    )


def convert_quantitative_coeff_to_per_percent_m(values: np.ndarray, *, unit: str) -> np.ndarray:
    """Convert external quantitative spectra to per-1%-per-meter coefficients."""
    coeff = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(coeff)):
        raise ValueError("absorption coefficient values must be finite")
    if unit == "per_percent_m":
        return coeff
    if unit == "per_fraction_m":
        return coeff * 0.01
    if unit == "per_ppm_m":
        return coeff * 10000.0
    raise ValueError(f"Unsupported quantitative spectrum unit: {unit!r}. Available: {SUPPORTED_QUANTITATIVE_UNITS}")


def resample_spectrum_to_grid(spectrum: TabulatedSpectrum, target_wavenumber_cm1: np.ndarray) -> TabulatedSpectrum:
    """Interpolate a spectrum onto a target grid, refusing extrapolation."""
    target = np.asarray(target_wavenumber_cm1, dtype=np.float64)
    _validate_grid(target, name="target_wavenumber_cm1")
    _validate_grid(spectrum.wavenumber_cm1, name=f"{spectrum.gas} wavenumber_cm1")
    if spectrum.absorption_coeff_per_percent_m.shape != spectrum.wavenumber_cm1.shape:
        raise ValueError("absorption_coeff_per_percent_m must match wavenumber_cm1 shape")
    tolerance = _grid_endpoint_tolerance(target, spectrum.wavenumber_cm1)
    if target[0] < spectrum.wavenumber_cm1[0] - tolerance or target[-1] > spectrum.wavenumber_cm1[-1] + tolerance:
        raise ValueError(
            f"{spectrum.gas} spectrum does not cover target grid "
            f"{target[0]:.4f}-{target[-1]:.4f} cm-1"
        )
    return TabulatedSpectrum(
        gas=spectrum.gas,
        wavenumber_cm1=target,
        absorption_coeff_per_percent_m=np.interp(
            target,
            spectrum.wavenumber_cm1,
            spectrum.absorption_coeff_per_percent_m,
        ).astype(np.float64),
        source_version=spectrum.source_version,
    )


def _read_numeric_rows(path: Path, *, wavenumber_column: str, coeff_column: str) -> list[tuple[float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Quantitative spectrum CSV not found: {path}")
    rows: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Quantitative spectrum CSV must have a header row")
        missing = {wavenumber_column, coeff_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Quantitative spectrum CSV missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            try:
                rows.append((float(row[wavenumber_column]), float(row[coeff_column])))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric value in {path} at line {line_number}") from exc
    return rows


def _validate_spectrum_arrays(wavenumber: np.ndarray, coeff: np.ndarray, *, source: str) -> None:
    if wavenumber.shape != coeff.shape:
        raise ValueError("wavenumber and absorption coefficient columns must have the same length")
    if wavenumber.ndim != 1:
        raise ValueError("quantitative spectrum columns must be one-dimensional")
    if wavenumber.size < 2:
        raise ValueError(f"Quantitative spectrum must contain at least two rows: {source}")
    if not np.all(np.isfinite(wavenumber)):
        raise ValueError("wavenumber_cm1 values must be finite")
    if not np.all(np.isfinite(coeff)):
        raise ValueError("absorption coefficient values must be finite")


def _validate_grid(grid: np.ndarray, *, name: str) -> None:
    if grid.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if grid.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} values must be finite")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} values must be strictly increasing")


def _grid_endpoint_tolerance(*grids: np.ndarray) -> float:
    min_step = min(float(np.min(np.diff(grid))) for grid in grids)
    return max(1e-9, min_step * 1e-9)

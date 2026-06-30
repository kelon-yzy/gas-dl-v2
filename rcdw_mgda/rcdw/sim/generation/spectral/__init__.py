"""RCDW 光学吸收 spectral 子包。

完全复用 HG 主线的 ``filters / cache / integration / tabulated_backend /
hitran_backend`` 逻辑，但通过独立的 ``defaults.py`` 强制仅注册 (CO2, H2O)
两种气体与 co2 单通道。不 import ``sim.generation.spectral.*``。
"""

from rcdw.sim.generation.spectral.cache import (
    SpectralCacheKey,
    cache_path,
    read_cached_spectrum,
    write_cached_spectrum,
)
from rcdw.sim.generation.spectral.defaults import (
    DEFAULT_HITRAN_GAS_SPECS,
    DEFAULT_HITRAN_GRID_SPECS,
    DEFAULT_NDIR_FILTERS,
    SPECTRAL_DEFAULTS_CONFIG_PATH,
    SPECTRAL_DEFAULTS_PAYLOAD,
    get_default_hitran_grid,
    get_default_ndir_filter,
)
from rcdw.sim.generation.spectral.filters import NDIRFilter, gaussian_filter
from rcdw.sim.generation.spectral.hitran_backend import (
    BOLTZMANN_J_PER_K,
    HITRAN_ABSORPTION_BACKEND,
    STANDARD_ATMOSPHERE_PA,
    HitranGasSpec,
    HitranGridSpec,
    MissingHitranCacheError,
    MissingHitranTableError,
    compute_hitran_ndir_absorbance,
    convert_hitran_coeff_to_per_percent_m,
    hitran_cache_key,
)
from rcdw.sim.generation.spectral.integration import integrate_channel_absorbance
from rcdw.sim.generation.spectral.tabulated_backend import (
    PreparedTabulatedSpectra,
    TabulatedSpectrum,
    compute_prepared_tabulated_ndir_absorbance,
    compute_tabulated_ndir_absorbance,
    prepare_tabulated_spectra,
)

__all__ = [
    "BOLTZMANN_J_PER_K",
    "DEFAULT_HITRAN_GAS_SPECS",
    "DEFAULT_HITRAN_GRID_SPECS",
    "DEFAULT_NDIR_FILTERS",
    "HITRAN_ABSORPTION_BACKEND",
    "HitranGasSpec",
    "HitranGridSpec",
    "MissingHitranCacheError",
    "MissingHitranTableError",
    "NDIRFilter",
    "PreparedTabulatedSpectra",
    "SPECTRAL_DEFAULTS_CONFIG_PATH",
    "SPECTRAL_DEFAULTS_PAYLOAD",
    "SpectralCacheKey",
    "STANDARD_ATMOSPHERE_PA",
    "TabulatedSpectrum",
    "cache_path",
    "compute_hitran_ndir_absorbance",
    "compute_prepared_tabulated_ndir_absorbance",
    "compute_tabulated_ndir_absorbance",
    "convert_hitran_coeff_to_per_percent_m",
    "gaussian_filter",
    "get_default_hitran_grid",
    "get_default_ndir_filter",
    "hitran_cache_key",
    "integrate_channel_absorbance",
    "prepare_tabulated_spectra",
    "read_cached_spectrum",
    "write_cached_spectrum",
]

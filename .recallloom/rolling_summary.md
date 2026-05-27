<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-27 -->
<!-- file-state: revision=27 | updated-at=2026-05-27T09:27:52+08:00 | writer-id=Codex | base-workspace-revision=51 -->

<!-- section: current_state -->
- 2026-05-27 HITRAN default spec is now genuinely runnable from repo-root CLI. The root-level pipeline launcher package forwards python -m pipeline.<tool> to src/pipeline and injects src for sim/dl/ml imports, so README pipeline commands no longer require PYTHONPATH.
- HITRAN backend now reuses local real HAPI .data/.header line tables before remote fetch when a per-condition .npz cache entry is missing. It only calls hapi.fetch when the raw local line table is absent; no fake spectra or silent fallback were introduced.
- Default benchmark HITRAN cache for sequences=32, seed=42, sampling_strategy=lhs is complete under data/hitran_cache: precompute_hitran_benchmark_cache reports required_cache_entries=192 for 32 conditions.
- A real default HITRAN benchmark smoke dataset was generated at outputs/runs/hitran-usable-20260527-seed42 with sequences=32, timesteps=8, storage=npz. Manifest uses optical_absorption_backend=hitran_hapi_v1, hitran_cache_policy=cache_only_prechecked, primary_key=mixture_id, split_group_field=mixture_id, and quality validation status=pass.
- Numeric spot check passed for the generated dataset: slow shape=(32,8,8), y shape=(32,4), slow/labels are finite, NDIR channels have non-zero dynamic ranges, and ultrasonic/fiber_mic int16 waveforms have non-zero dynamic range.
- sim core remains around 90% complete; dl/data around 45%; dl/models around 25%; dl/training around 20% with loss and metrics only; pipeline around 30%; configs around 8%.
- Tests currently pass: targeted HITRAN/import group is 20 passed; full python -m pytest tests is 134 passed; git diff --check passes with only Windows LF/CRLF warnings.

<!-- section: active_judgments -->
- First phase remains focused on core contracts rather than full feature breadth.
- Benchmark default generation uses hitran_hapi_v1; empirical_v1 remains an explicit opt-in comparison/regression path.
- Benchmark dataset generation must stay cache-only for HITRAN: it prechecks required cache keys and fails visibly on missing cache; HAPI import, network fetch, and cache writes belong to precompute CLIs.
- HITRAN T/P cache key granularity remains temperature_k=round(T_C+273.15,3) and pressure_atm=round(P_MPa/0.101325,6).
- H2O is derived from T/P/RH as an optical absorber only; it is not part of label composition or 100% component validation.
- HITRAN multigas filter integration expresses channel cross-response; default path should not add empirical optical crosstalk on top.
- configs/data/spectral-defaults.json is the runtime source of truth for spectral defaults; Python code should construct dataclasses from that JSON instead of maintaining a second constants mirror.
- Default HITRAN grid must cover filter center +/- FWHM; filter/grid/default changes require regenerating the corresponding HITRAN cache.

<!-- section: risks_open_questions -->
- TraceGas-HC-NDIR target datasheet is still missing; current CH4 fwhm 147 cm-1 and CO2 fwhm 93 cm-1 are InfraTec NBP industry-reference placeholders, not vendor-confirmed values.
- External PNNL/NIST or instrument quantitative spectra have not been imported yet; the generic CSV sanity-check path exists but still needs real external data.
- HITRAN cache is spec-specific. Changing sampling_strategy, seed, sequences, filter, spectral defaults, or grid requires re-running the matching precompute step.
- End-to-end trainer, checkpoint management, training config, and experiment tracking are still not implemented.
- TCN channel counts, kernel size, depth, and formal experiment configs remain unset; time-step phase distribution is still fixed four-way segmentation.

<!-- section: next_step -->
- HITRAN default spec is now usable. If continuing the engineering mainline, return to the minimal Trainer/checkpoint/training-config smoke path.

<!-- section: recent_pivots -->
- 2026-05-27: Made repo-root pipeline CLI usable without PYTHONPATH by adding a thin root-level pipeline launcher package; added tests/test_import_bootstrap.py to guard python -m pipeline.generate_benchmark --help from repo root.
- 2026-05-27: Updated HITRAN backend to reuse local HAPI .data/.header tables before hapi.fetch on .npz miss; added a regression test proving local table reuse does not fetch.
- 2026-05-27: Verified default benchmark HITRAN cache for sequences=32, seed=42, sampling_strategy=lhs; required_cache_entries=192 and precompute command exits successfully.
- 2026-05-27: Generated outputs/runs/hitran-usable-20260527-seed42 with default HITRAN backend and passed manifest, validation, and numeric spot checks.
- 2026-05-27: Earlier generated full default HITRAN cache after initial small smoke, using local real HITRAN tables and HAPI absorptionCoefficient_Voigt rather than fake spectra.
- 2026-05-26: Resolved review findings from docs/CODE_REVIEW_2026-05-26.md and committed fix(benchmark): resolve review findings at 0c004abafa352747ee53c2c613f52418adb54f2f.
- 2026-05-26: Connected hitran_hapi_v1 as the default benchmark optical backend with cache-only precheck, per-timestep NDIR equilibrium, and benchmark HITRAN precompute CLI.

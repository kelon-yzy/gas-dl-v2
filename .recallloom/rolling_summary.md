<!-- recallloom:file=rolling_summary version=1.0 lang=zh-CN -->
<!-- last-writer: [Codex] | 2026-05-28 -->
<!-- file-state: revision=29 | updated-at=2026-05-28T10:35:15+08:00 | writer-id=Codex | base-workspace-revision=55 -->

<!-- section: current_state -->
- 2026-05-28 acoustic-chain research and short-term contract fix are complete. docs/当前声学链路问题.md now contains a literature-backed section 9 with solutions for ultrasonic TOF, system delay/transducer response, molecular relaxation absorption, Fabry-Perot/fiber acoustic sensing, and DAQ boundary modeling.
- New benchmark generation records acoustic model boundaries in manifest.json and metadata/waveform_spec.json: ultrasonic_model=simplified_tof_proxy_v1, fiber_mic_model=acoustic_proxy_v1, fiber_optical_demodulation_model=not_implemented, acoustic_attenuation_model=semi_empirical_relaxation_proxy_v1.
- Ultrasonic TOF observables are now explicit generated assets: sequences/ultrasonic_tof_s.npy, ultrasonic_peak_index.npy, ultrasonic_sound_speed_m_per_s.npy, and ultrasonic_alpha_true_npm.npy. They are written for npy/memmap and included in waveform_sequence.npz.
- Validation after the acoustic-chain changes passed: targeted tests for benchmark generation, CLI, HITRAN integration, and ML baselines reported 21 passed; full python -m pytest tests reported 143 passed.
- HITRAN default spec remains genuinely runnable from repo-root CLI. The root-level pipeline launcher forwards python -m pipeline.<tool> to src/pipeline and injects src for sim/dl/ml imports, so README pipeline commands do not require PYTHONPATH.
- HITRAN backend reuses local real HAPI .data/.header line tables before remote fetch when a per-condition .npz cache entry is missing. Benchmark generation remains cache-only and fails visibly on missing HITRAN cache.
- Default benchmark HITRAN cache for sequences=32, seed=42, sampling_strategy=lhs is complete under data/hitran_cache: required_cache_entries=192 for 32 conditions.
- src/ml has a dependency-light traditional baseline path: MLFeatureConfig/load_feature_matrix for slow/ultrasonic/fiber_mic tabular features, MeanRegressor, pure-numpy closed-form RidgeRegressor, numpy regression metrics, and train_regressor_on_dataset for split evaluation. It intentionally avoids scikit-learn for now.

<!-- section: active_judgments -->
- First phase remains focused on core contracts rather than full feature breadth.
- Benchmark default generation uses hitran_hapi_v1; empirical_v1 remains an explicit opt-in comparison/regression path.
- Benchmark dataset generation must stay cache-only for HITRAN: it prechecks required cache keys and fails visibly on missing cache; HAPI import, network fetch, and cache writes belong to precompute CLIs.
- Formal docs and data contracts must describe current fiber_mic as acoustic_proxy_v1 / simplified proxy model, not as completed fiber-interferometric attenuation measurement.
- Do not fake a full fiber interferometric model without real probe, demodulator, amplifier, and DAQ calibration parameters. The next legitimate model should be a separately named fiber_interferometric_proxy_v1.
- hidden_attenuation_v2 is currently a semi_empirical_relaxation_proxy_v1. It is useful as a molecular relaxation proxy, but not a standardized or instrument-calibrated acoustic absorption model.
- Ultrasonic waveform currently remains simplified_tof_proxy_v1. TOF observables are explicit, but system delay, transducer response, trigger jitter, frontend response, and tof_quality are still future model work.
- Traditional ML baseline should stay dependency-light unless the project explicitly accepts optional scikit-learn; SVR/RandomForest remain deferred.
- Python 3.14 is not a supported project environment yet because scientific/deep-learning wheels are not treated as stable there for this project.

<!-- section: risks_open_questions -->
- True fiber probe, interferometric demodulator, photodetector, amplifier, and DAQ calibration parameters are still missing, so the complete fiber-interferometric chain remains unimplemented.
- Ultrasonic system delay, transducer bandwidth/impulse response, trigger jitter, frontend response, and TOF quality estimation are still not modeled.
- New generated datasets now include explicit ultrasonic TOF assets and acoustic model metadata. Existing generated datasets do not contain these fields and need regeneration if downstream code expects them.
- TraceGas-HC-NDIR target datasheet is still missing; current CH4 fwhm 147 cm-1 and CO2 fwhm 93 cm-1 are InfraTec NBP industry-reference placeholders, not vendor-confirmed values.
- External PNNL/NIST or instrument quantitative spectra have not been imported yet; the generic CSV sanity-check path exists but still needs real external data.
- HITRAN cache is spec-specific. Changing sampling_strategy, seed, sequences, filter, spectral defaults, or grid requires re-running the matching precompute step.
- End-to-end DL trainer, checkpoint management, training config, and experiment tracking are still not implemented.
- TCN channel counts, kernel size, depth, and formal experiment configs remain unset; time-step phase distribution is still fixed four-way segmentation.

<!-- section: next_step -->
- If continuing the acoustic mainline, implement WaveformSpec system_delay_s / trigger_jitter_std_s / transducer_response_model and derive tof_observed_s, sound_speed_estimated_m_per_s, and tof_quality.
- Then add FiberProbeSpec and a distinct fiber_interferometric_proxy_v1 that models probe pressure-to-phase or pressure-to-cavity-length transduction, optical transmission, demodulation, photodetector/amplifier noise, saturation, and DAQ quantization.
- If returning to the broader engineering mainline, confirm downstream dl/ml consumers tolerate the new acoustic assets and continue the minimal Trainer/checkpoint/training-config smoke path.

<!-- section: recent_pivots -->
- 2026-05-28: Added literature-backed solution section to docs/当前声学链路问题.md with references to AGA-9, GERG TM11, NIST AGA8 sound-speed material, acoustic relaxation absorption work, CO2 DBR fiber-laser absorption measurement, and FPI fiber-optic microphone/pressure sensor papers.
- 2026-05-28: Added acoustic model metadata to generated manifest.json and metadata/waveform_spec.json; README now states ultrasonic is a simplified TOF proxy and fiber_mic is acoustic_proxy_v1 with optical demodulation not implemented.
- 2026-05-28: Added explicit ultrasonic TOF derived arrays and tests; full python -m pytest tests passed with 143 tests.
- 2026-05-27: Added versioned dependency setup for new machines: pyproject package metadata/dependencies, requirements.txt pip entry, Python >=3.10,<3.14, and README environment bootstrap notes.
- 2026-05-27: Added dependency-light src/ml baseline with tabular v4 feature extraction, numpy Mean/Ridge regressors, regression metrics, train_regressor_on_dataset, and tests/test_ml_baselines.py.
- 2026-05-27: Made repo-root pipeline CLI usable without PYTHONPATH by adding a thin root-level pipeline launcher package; added tests/test_import_bootstrap.py to guard python -m pipeline.generate_benchmark --help from repo root.
- 2026-05-27: Updated HITRAN backend to reuse local HAPI .data/.header tables before hapi.fetch on .npz miss and generated a real default HITRAN smoke dataset.

"""合成气 benchmark 并行 chunk 生成与合并。

从 benchmark.py 拆出以保持单文件 <400 行。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sg.sim.generation.phases import PhaseSchedule
from sg.sim.generation.syngas.slow import build_sequence_arrays
from sg.sim.generation.waveforms import FiberMicSpec, WaveformSpec

# benchmark.py 定义的 ARRAY_KEYS 由调用方传入，避免循环 import。
# 使用类型别名保持清晰。
ArrayKeys = tuple[str, ...]


def build_arrays_parallel(
    *,
    conditions: list[dict[str, str]],
    spec: object,
    phase_schedule: PhaseSchedule,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
    staging_dir: Path,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    """多进程 chunk 生成 + 合并。

    spec 需要有 workers / chunk_size / temp_dir / keep_chunks / timesteps /
    dt_s / seed / multi_path_phase / path_lms / stage_jitter /
    optical_absorption_backend / hitran_cache_root 属性。
    """
    from sg.sim.generation.syngas.benchmark import default_chunk_size

    chunk_size = spec.chunk_size or default_chunk_size(len(conditions), spec.workers)  # type: ignore[attr-defined]
    temp_dir = Path(spec.temp_dir) if spec.temp_dir is not None else staging_dir / ".chunks"  # type: ignore[attr-defined]
    chunk_specs = _condition_chunks(conditions, chunk_size)
    temp_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(spec.workers, len(chunk_specs))) as executor:  # type: ignore[attr-defined]
        futures = [
            executor.submit(
                _generate_chunk_file,
                chunk_index,
                chunk_conditions,
                start_index,
                temp_dir,
                spec,
                phase_schedule,
                ultrasonic_spec,
                fiber_mic_spec,
                array_keys,
            )
            for chunk_index, start_index, chunk_conditions in chunk_specs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    arrays = _merge_chunk_files(results, sequence_count=len(conditions), temp_dir=temp_dir, array_keys=array_keys)
    if not spec.keep_chunks and spec.temp_dir is None:  # type: ignore[attr-defined]
        arrays["_temp_dir_to_cleanup"] = str(temp_dir)
    return arrays


def cleanup_parallel_temp_arrays(arrays: dict[str, object], array_keys: ArrayKeys) -> None:
    """关闭 memmap 并删除临时目录。"""
    import shutil

    temp_dir_value = arrays.pop("_temp_dir_to_cleanup", None)
    if temp_dir_value is None:
        return
    for key in array_keys:
        array = arrays.get(key)
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()
    temp_dir = Path(str(temp_dir_value))
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _condition_chunks(
    conditions: list[dict[str, str]],
    chunk_size: int,
) -> list[tuple[int, int, list[dict[str, str]]]]:
    chunks: list[tuple[int, int, list[dict[str, str]]]] = []
    for chunk_index, start in enumerate(range(0, len(conditions), chunk_size)):
        chunks.append((chunk_index, start, conditions[start : start + chunk_size]))
    return chunks


def _generate_chunk_file(
    chunk_index: int,
    conditions: list[dict[str, str]],
    start_sequence_index: int,
    temp_dir: Path,
    spec: object,
    phase_schedule: PhaseSchedule,
    ultrasonic_spec: WaveformSpec,
    fiber_mic_spec: FiberMicSpec,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    arrays = build_sequence_arrays(
        conditions,
        timesteps=spec.timesteps,  # type: ignore[attr-defined]
        dt_s=spec.dt_s,  # type: ignore[attr-defined]
        seed=spec.seed,  # type: ignore[attr-defined]
        multi_path_phase=spec.multi_path_phase,  # type: ignore[attr-defined]
        ultrasonic_spec=ultrasonic_spec,
        fiber_mic_spec=fiber_mic_spec,
        path_lms=spec.path_lms,  # type: ignore[attr-defined]
        phase_schedule=phase_schedule,
        stage_jitter=spec.stage_jitter,  # type: ignore[attr-defined]
        optical_absorption_backend=spec.optical_absorption_backend,  # type: ignore[attr-defined]
        hitran_cache_root=spec.hitran_cache_root,  # type: ignore[attr-defined]
        start_sequence_index=start_sequence_index,
        enable_co_crosstalk=spec.enable_co_crosstalk,  # type: ignore[attr-defined]
    )
    chunk_path = temp_dir / f"chunk-{chunk_index:05d}.npz"
    np.savez(
        chunk_path,
        **{key: arrays[key] for key in array_keys},
        # slow_rows 存为 object array 以支持 dict 序列化
        slow_rows=np.array(arrays["slow_rows"], dtype=object),
    )
    return {
        "chunk_index": chunk_index,
        "start_sequence_index": start_sequence_index,
        "sequence_count": len(conditions),
        "path": str(chunk_path),
    }


def _merge_chunk_files(
    results: list[dict[str, object]],
    *,
    sequence_count: int,
    temp_dir: Path,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    ordered = sorted(results, key=lambda item: int(item["chunk_index"]))
    if not ordered:
        raise ValueError("no chunk files were generated")
    arrays: dict[str, object] = {}
    # allow_pickle=True 因为 slow_rows 存为 object array
    with np.load(str(ordered[0]["path"]), allow_pickle=True) as first_payload:
        for key in array_keys:
            sample = first_payload[key]
            target = np.lib.format.open_memmap(
                temp_dir / f"merged_{key}.npy",
                mode="w+",
                dtype=sample.dtype,
                shape=(sequence_count, *sample.shape[1:]),
            )
            arrays[key] = target

    slow_rows: list[dict[str, str]] = []
    for result in ordered:
        start = int(result["start_sequence_index"])
        count = int(result["sequence_count"])
        # allow_pickle=True 因为 slow_rows 存为 object array
        with np.load(str(result["path"]), allow_pickle=True) as payload:
            end = start + count
            for key in array_keys:
                arrays[key][start:end] = payload[key]
            slow_rows.extend(dict(row) for row in payload["slow_rows"].tolist())
    for key in array_keys:
        arrays[key].flush()
    arrays["slow_rows"] = slow_rows
    return arrays

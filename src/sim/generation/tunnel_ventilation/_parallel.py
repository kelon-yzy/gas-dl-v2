"""掘进通风 benchmark 并行 chunk 生成与合并。

从 benchmark.py 拆出以保持单文件 <400 行。接口与 syngas `_parallel` 对齐。

内存优化（2026-07-06）：chunk 生成不再通过 npz 中转，而是让 build_sequence_arrays
直接以 memmap 写入 chunk 临时目录，合并阶段从 chunk memmap 拷贝到 merged memmap。
这样每个 worker 的 RAM 占用量仅为当前 timestep 的波形中间结果（<100 MiB），
不再受 chunk 序列数 × timesteps × 5000 点的大数组支配。
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from sim.generation.tunnel_ventilation.slow import build_sequence_arrays

# benchmark.py 定义的 ARRAY_KEYS 由调用方传入，避免循环 import。
ArrayKeys = tuple[str, ...]

# 大波形数组 key 列表 —— 这些数组通过 memmap 直接落盘，不走 npz
_WAVEFORM_ARRAY_KEYS = ("ultrasonic", "fiber_mic")


def build_arrays_parallel(
    *,
    conditions: list[dict[str, str]],
    spec: object,
    phase_schedule: object,
    ultrasonic_spec: object,
    fiber_mic_spec: object | None,
    staging_dir: Path,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    """多进程 chunk 生成 + 合并。

    spec 需要有 workers / chunk_size / temp_dir / keep_chunks / timesteps /
    dt_s / seed / multi_path_phase / path_lms / stage_jitter /
    optical_absorption_backend / hitran_cache_root 属性。
    """
    from sim.generation.tunnel_ventilation.benchmark import default_chunk_size

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
    # 关闭 merged memmap（防止 Windows 下文件占用）
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
    phase_schedule: object,
    ultrasonic_spec: object,
    fiber_mic_spec: object | None,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    """生成单个 chunk 的序列数组，大波形数组通过 memmap 直接写入临时目录。

    slow_rows（Python dict 列表）单独序列化到 .npz，
    其余数组由 build_sequence_arrays 以 memmap 写入 chunk 子目录。
    """
    chunk_dir = temp_dir / f"chunk-{chunk_index:05d}"
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
        temp_dir=chunk_dir,
    )
    # 小数组（非波形）仍在 RAM 中，与 slow_rows 一起序列化到 .npz
    small_keys = [k for k in array_keys if k not in _WAVEFORM_ARRAY_KEYS]
    small_arrays = {k: arrays[k] for k in small_keys}
    # slow_rows 存为 object array 以支持 dict 序列化
    small_arrays["slow_rows"] = np.array(arrays["slow_rows"], dtype=object)

    # 关闭 chunk memmap 对象，让子进程释放文件句柄
    for key in _WAVEFORM_ARRAY_KEYS:
        arr = arrays.get(key)
        if arr is not None:
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()

    np.savez(chunk_dir / "small.npz", **small_arrays)
    return {
        "chunk_index": chunk_index,
        "start_sequence_index": start_sequence_index,
        "sequence_count": len(conditions),
        "chunk_dir": str(chunk_dir),
    }


def _merge_chunk_files(
    results: list[dict[str, object]],
    *,
    sequence_count: int,
    temp_dir: Path,
    array_keys: ArrayKeys,
) -> dict[str, object]:
    """合并所有 chunk：大波形数组从 chunk memmap 拷贝，小数组从 npz 加载。"""
    ordered = sorted(results, key=lambda item: int(item["chunk_index"]))
    if not ordered:
        raise ValueError("no chunk files were generated")

    arrays: dict[str, object] = {}

    # 用第一个 chunk 确定各数组的 dtype / shape
    first_dir = Path(str(ordered[0]["chunk_dir"]))
    # 大波形数组：从 chunk memmap 获取 shape
    for key in _WAVEFORM_ARRAY_KEYS:
        waveform_path = first_dir / f"{key}.npy"
        if waveform_path.exists():
            sample = np.lib.format.open_memmap(str(waveform_path), mode="r")
            arrays[key] = np.lib.format.open_memmap(
                temp_dir / f"merged_{key}.npy",
                mode="w+",
                dtype=sample.dtype,
                shape=(sequence_count, *sample.shape[1:]),
            )
            sample._mmap.close()
        # key 不存在时不写入 arrays dict，避免下游代码访问 None

    # 小数组：从第一个 chunk 的 npz 获取 shape
    with np.load(str(first_dir / "small.npz"), allow_pickle=True) as first_small:
        for key in array_keys:
            if key in _WAVEFORM_ARRAY_KEYS:
                continue
            sample = first_small[key]
            arrays[key] = np.lib.format.open_memmap(
                temp_dir / f"merged_{key}.npy",
                mode="w+",
                dtype=sample.dtype,
                shape=(sequence_count, *sample.shape[1:]),
            )

    slow_rows: list[dict[str, str]] = []
    for result in ordered:
        start = int(result["start_sequence_index"])
        count = int(result["sequence_count"])
        chunk_dir = Path(str(result["chunk_dir"]))
        end = start + count

        # 大波形数组：从 chunk memmap 拷贝
        for key in _WAVEFORM_ARRAY_KEYS:
            waveform_path = chunk_dir / f"{key}.npy"
            if not waveform_path.exists():
                continue
            src = np.lib.format.open_memmap(str(waveform_path), mode="r")
            arrays[key][start:end] = src
            src._mmap.close()

        # 小数组：从 npz 加载
        with np.load(str(chunk_dir / "small.npz"), allow_pickle=True) as small:
            for key in array_keys:
                if key in _WAVEFORM_ARRAY_KEYS:
                    continue
                arrays[key][start:end] = small[key]
            slow_rows.extend(dict(row) for row in small["slow_rows"].tolist())

    for key in array_keys:
        arr = arrays.get(key)
        if arr is not None:
            arr.flush()
    arrays["slow_rows"] = slow_rows
    return arrays

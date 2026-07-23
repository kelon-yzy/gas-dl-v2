from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np

from tv3.common.waveform import waveform_array_filename


def _memmap_source_path(array: object) -> Path | None:
    """Return on-disk path for a NumPy memmap, else None."""
    filename = getattr(array, "filename", None)
    if not filename:
        # Some numpy builds expose the path only on the underlying mmap.
        mmap = getattr(array, "_mmap", None)
        filename = getattr(mmap, "filename", None) if mmap is not None else None
    if not filename:
        return None
    # mmap.filename may be bytes on some platforms.
    if isinstance(filename, bytes):
        filename = filename.decode("utf-8", errors="surrogateescape")
    path = Path(str(filename))
    return path if path.is_file() else None


def _close_array_mmap(array: object) -> None:
    mmap = getattr(array, "_mmap", None)
    if mmap is not None:
        try:
            mmap.close()
        except ValueError:
            pass


def _disk_free_bytes(path: Path) -> int:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _array_nbytes(array: object) -> int:
    try:
        return int(np.asarray(array).nbytes)
    except Exception:
        shape = getattr(array, "shape", None)
        dtype = getattr(array, "dtype", None)
        if shape is None or dtype is None:
            return 0
        return int(np.prod(shape) * np.dtype(dtype).itemsize)


def write_arrays(output_dir: Path, arrays: dict[str, object], labels: np.ndarray, sequence_ids: list[str], slow_channel_names: tuple[str, ...], label_names: tuple[str, ...], storage: str, *, ultrasonic_dtype: str = "int16", fiber_dtype: str = "int16") -> dict[str, list[int]]:
    if arrays.get("bidirectional"):
        return write_bidirectional_arrays(
            output_dir,
            arrays,
            labels,
            sequence_ids,
            slow_channel_names,
            label_names,
            storage,
            ultrasonic_dtype=ultrasonic_dtype,
            fiber_dtype=fiber_dtype,
        )
    sequences_dir = output_dir / "sequences"
    labels_dir = output_dir / "labels"
    metadata_dir = output_dir / "metadata"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    slow = arrays["slow"]
    ultrasonic = arrays["ultrasonic"]
    ultrasonic_scale = arrays["ultrasonic_scale"]
    ultrasonic_tof_s = arrays["ultrasonic_tof_s"]
    ultrasonic_tof_observed_s = arrays["ultrasonic_tof_observed_s"]
    ultrasonic_peak_index = arrays["ultrasonic_peak_index"]
    ultrasonic_sound_speed = arrays["ultrasonic_sound_speed_m_per_s"]
    ultrasonic_sound_speed_estimated = arrays["ultrasonic_sound_speed_estimated_m_per_s"]
    ultrasonic_alpha = arrays["ultrasonic_alpha_true_npm"]
    ultrasonic_tof_quality = arrays["ultrasonic_tof_quality"]
    ultrasonic_tof_accepted = arrays["ultrasonic_tof_accepted"]
    fiber_mic = arrays.get("fiber_mic")
    fiber_mic_scale = arrays.get("fiber_mic_scale")

    # Capture shapes before publish: relocating memmaps closes source handles.
    shapes = {
        "slow": list(slow.shape),
        "ultrasonic": list(ultrasonic.shape),
        "ultrasonic_scale": list(ultrasonic_scale.shape),
        "ultrasonic_tof_s": list(ultrasonic_tof_s.shape),
        "ultrasonic_tof_observed_s": list(ultrasonic_tof_observed_s.shape),
        "ultrasonic_peak_index": list(ultrasonic_peak_index.shape),
        "ultrasonic_sound_speed_m_per_s": list(ultrasonic_sound_speed.shape),
        "ultrasonic_sound_speed_estimated_m_per_s": list(ultrasonic_sound_speed_estimated.shape),
        "ultrasonic_alpha_true_npm": list(ultrasonic_alpha.shape),
        "ultrasonic_tof_quality": list(ultrasonic_tof_quality.shape),
        "ultrasonic_tof_accepted": list(ultrasonic_tof_accepted.shape),
        "y": list(labels.shape),
    }
    if fiber_mic is not None:
        shapes["fiber_mic"] = list(fiber_mic.shape)
        shapes["fiber_mic_scale"] = list(fiber_mic_scale.shape)

    # Pure memmap storage: relocate on-disk temps (avoids 2x disk for ~28GiB waves).
    # storage=both keeps copy so in-memory handles remain readable for npz.
    use_memmap = storage in {"memmap", "both"}
    relocate = storage == "memmap"
    _write_npy(sequences_dir / "slow.npy", slow, use_memmap=use_memmap, relocate=relocate)
    _write_npy(
        sequences_dir / waveform_array_filename("ultrasonic", ultrasonic_dtype),
        ultrasonic,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(sequences_dir / "ultrasonic_scale.npy", ultrasonic_scale, use_memmap=use_memmap, relocate=relocate)
    _write_npy(sequences_dir / "ultrasonic_tof_s.npy", ultrasonic_tof_s, use_memmap=use_memmap, relocate=relocate)
    _write_npy(
        sequences_dir / "ultrasonic_tof_observed_s.npy",
        ultrasonic_tof_observed_s,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_peak_index.npy",
        ultrasonic_peak_index,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_sound_speed_m_per_s.npy",
        ultrasonic_sound_speed,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_sound_speed_estimated_m_per_s.npy",
        ultrasonic_sound_speed_estimated,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_alpha_true_npm.npy",
        ultrasonic_alpha,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_tof_quality.npy",
        ultrasonic_tof_quality,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / "ultrasonic_tof_accepted.npy",
        ultrasonic_tof_accepted,
        use_memmap=use_memmap,
        relocate=relocate,
    )
    if fiber_mic is not None:
        _write_npy(
            sequences_dir / waveform_array_filename("fiber_mic", fiber_dtype),
            fiber_mic,
            use_memmap=use_memmap,
            relocate=relocate,
        )
        _write_npy(
            sequences_dir / "fiber_mic_scale.npy",
            fiber_mic_scale,
            use_memmap=use_memmap,
            relocate=relocate,
        )
    np.save(labels_dir / "y.npy", labels)
    np.save(metadata_dir / "sequence_ids.npy", np.array(sequence_ids))
    np.save(metadata_dir / "slow_channel_names.npy", np.array(slow_channel_names))
    np.save(metadata_dir / "label_names.npy", np.array(label_names))

    if storage in {"npz", "both"}:
        npz_payload = {
            "ultrasonic": ultrasonic,
            "ultrasonic_scale": ultrasonic_scale,
            "ultrasonic_tof_s": ultrasonic_tof_s,
            "ultrasonic_tof_observed_s": ultrasonic_tof_observed_s,
            "ultrasonic_peak_index": ultrasonic_peak_index,
            "ultrasonic_sound_speed_m_per_s": ultrasonic_sound_speed,
            "ultrasonic_sound_speed_estimated_m_per_s": ultrasonic_sound_speed_estimated,
            "ultrasonic_alpha_true_npm": ultrasonic_alpha,
            "ultrasonic_tof_quality": ultrasonic_tof_quality,
            "ultrasonic_tof_accepted": ultrasonic_tof_accepted,
            "slow": slow,
            "y": labels,
            "sequence_ids": np.array(sequence_ids),
            "slow_channel_names": np.array(slow_channel_names),
            "label_names": np.array(label_names),
        }
        if fiber_mic is not None:
            npz_payload["fiber_mic"] = fiber_mic
            npz_payload["fiber_mic_scale"] = fiber_mic_scale
        np.savez_compressed(sequences_dir / "waveform_sequence.npz", **npz_payload)

    return shapes


def write_bidirectional_arrays(
    output_dir: Path,
    arrays: dict[str, object],
    labels: np.ndarray,
    sequence_ids: list[str],
    slow_channel_names: tuple[str, ...],
    label_names: tuple[str, ...],
    storage: str,
    *,
    ultrasonic_dtype: str = "int16",
    fiber_dtype: str = "int16",
) -> dict[str, list[int]]:
    """Write tunnel-ventilation-bidir-1 ultrasonic AB/BA arrays.

    Does not write legacy unidirectional ``ultrasonic*.npy`` keys.
    Oracle arrays are written for audit only; deploy loaders must ignore them.
    """
    sequences_dir = output_dir / "sequences"
    labels_dir = output_dir / "labels"
    metadata_dir = output_dir / "metadata"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    use_memmap = storage in {"memmap", "both"}
    slow = arrays["slow"]
    required = (
        "ultrasonic_ab",
        "ultrasonic_ba",
        "ultrasonic_ab_scale",
        "ultrasonic_ba_scale",
        "ultrasonic_tof_observed_ab_s",
        "ultrasonic_tof_observed_ba_s",
        "ultrasonic_peak_index_ab",
        "ultrasonic_peak_index_ba",
        "ultrasonic_tof_quality_ab",
        "ultrasonic_tof_quality_ba",
        "ultrasonic_tof_accepted_ab",
        "ultrasonic_tof_accepted_ba",
        "ultrasonic_tof_true_ab_s",
        "ultrasonic_tof_true_ba_s",
        "ultrasonic_v_path_true_m_per_s",
        "ultrasonic_sound_speed_m_per_s",
        "ultrasonic_alpha_true_npm",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"bidirectional arrays missing keys: {missing}")

    fiber_mic = arrays.get("fiber_mic")
    fiber_mic_scale = arrays.get("fiber_mic_scale")
    # Shapes before publish: relocate closes source memmap handles.
    shapes = {key: list(np.asarray(arrays[key]).shape) for key in ("slow", *required)}
    shapes["y"] = list(labels.shape)
    if fiber_mic is not None:
        shapes["fiber_mic"] = list(fiber_mic.shape)
        shapes["fiber_mic_scale"] = list(fiber_mic_scale.shape)

    relocate = storage == "memmap"
    _write_npy(sequences_dir / "slow.npy", slow, use_memmap=use_memmap, relocate=relocate)
    _write_npy(
        sequences_dir / waveform_array_filename("ultrasonic_ab", ultrasonic_dtype),
        arrays["ultrasonic_ab"],
        use_memmap=use_memmap,
        relocate=relocate,
    )
    _write_npy(
        sequences_dir / waveform_array_filename("ultrasonic_ba", ultrasonic_dtype),
        arrays["ultrasonic_ba"],
        use_memmap=use_memmap,
        relocate=relocate,
    )
    for key in required[2:]:
        _write_npy(
            sequences_dir / f"{key}.npy",
            arrays[key],
            use_memmap=use_memmap,
            relocate=relocate,
        )

    if fiber_mic is not None:
        _write_npy(
            sequences_dir / waveform_array_filename("fiber_mic", fiber_dtype),
            fiber_mic,
            use_memmap=use_memmap,
            relocate=relocate,
        )
        _write_npy(
            sequences_dir / "fiber_mic_scale.npy",
            fiber_mic_scale,
            use_memmap=use_memmap,
            relocate=relocate,
        )

    np.save(labels_dir / "y.npy", labels)
    np.save(metadata_dir / "sequence_ids.npy", np.array(sequence_ids))
    np.save(metadata_dir / "slow_channel_names.npy", np.array(slow_channel_names))
    np.save(metadata_dir / "label_names.npy", np.array(label_names))
    return shapes


def _write_npy(path: Path, array, *, use_memmap: bool, relocate: bool = False) -> None:
    """Publish ``array`` to ``path``.

    When ``relocate`` is true and ``array`` is already an on-disk memmap on the
    same filesystem, rename/replace instead of copying. This keeps peak disk near
    one copy of AB/BA (~57 GiB) instead of two (~115 GiB) during bidir formal write.
    Cross-device rename falls back to a streaming memmap copy **only when free
    space is sufficient**; otherwise raise a clear OSError instead of SIGBUS.
    """
    if use_memmap:
        src_path = _memmap_source_path(array) if relocate else None
        nbytes = _array_nbytes(array)
        if src_path is not None and src_path.resolve() != path.resolve():
            _close_array_mmap(array)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(src_path, path)
                if nbytes >= (1 << 30):
                    print(
                        f"[tv3-gen] published via rename "
                        f"{src_path.name} -> {path.name} ({nbytes / (1 << 30):.1f} GiB)",
                        flush=True,
                    )
                return
            except OSError as exc:
                free = _disk_free_bytes(path.parent)
                if free < nbytes + (1 << 30):
                    raise OSError(
                        f"cannot publish {path.name}: rename failed ({exc}); "
                        f"copy fallback needs ~{nbytes / (1 << 30):.1f} GiB but only "
                        f"{free / (1 << 30):.1f} GiB free under {path.parent}. "
                        f"Free disk or keep temp+output on the same filesystem."
                    ) from exc
                print(
                    f"[tv3-gen] rename failed ({exc}); streaming copy "
                    f"{src_path.name} -> {path.name} ({nbytes / (1 << 30):.1f} GiB, "
                    f"free={free / (1 << 30):.1f} GiB)",
                    flush=True,
                )
                src = np.lib.format.open_memmap(str(src_path), mode="r")
                try:
                    target = np.lib.format.open_memmap(
                        path, mode="w+", dtype=src.dtype, shape=src.shape
                    )
                    target[:] = src
                    target.flush()
                finally:
                    _close_array_mmap(src)
                try:
                    src_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return
        if nbytes >= (1 << 30):
            free = _disk_free_bytes(path.parent)
            if free < nbytes + (1 << 30):
                src_desc = str(src_path) if src_path is not None else "in-memory/non-memmap"
                raise OSError(
                    f"cannot publish {path.name}: need ~{nbytes / (1 << 30):.1f} GiB "
                    f"to copy from {src_desc}, but only {free / (1 << 30):.1f} GiB free. "
                    f"Ensure memmap rename publish is active and merged_*.npy stays on "
                    f"the same filesystem as sequences/."
                )
            print(
                f"[tv3-gen] publishing via copy {path.name} "
                f"({nbytes / (1 << 30):.1f} GiB, free={free / (1 << 30):.1f} GiB)",
                flush=True,
            )
        target = np.lib.format.open_memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
        target[:] = array
        target.flush()
        return
    np.save(path, array)

#!/usr/bin/env python3
"""F2 audit: validate tv3-bidir-smoke contract, int16 storage self-consistency, size."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.sim.core.tunnel_ventilation_bidir_schema import (  # noqa: E402
    BIDIR_ORACLE_ARRAYS,
    COMPOSITION_SCHEME,
    SCHEMA_VERSION,
    SIM_REVISION_TAG,
    SLOW_CHANNELS,
)
from tv3.common.waveform import waveform_array_filename  # noqa: E402
from tv3.sim.generation.tunnel_ventilation.bidir_registry import (  # noqa: E402
    default_config_dir,
    sha256_file,
)


REQUIRED_FILES = (
    "manifest.json",
    "condition_grid_sequence.csv",
    "sequence_index.csv",
    "sequence_labels.csv",
    "quality/validation_summary.json",
    "metadata/waveform_spec.json",
    "metadata/slow_channel_names.npy",
    "metadata/label_names.npy",
    "labels/y.npy",
    "sequences/slow.npy",
    f"sequences/{waveform_array_filename('ultrasonic_ab', 'int16')}",
    f"sequences/{waveform_array_filename('ultrasonic_ba', 'int16')}",
    "sequences/ultrasonic_ab_scale.npy",
    "sequences/ultrasonic_ba_scale.npy",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-dir",
        type=Path,
        default=_TV3_ROOT / "data" / "tv3-bidir-smoke",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_TV3_ROOT / "outputs" / "tv3_bidir" / "benchmark_audit",
    )
    p.add_argument(
        "--f0-verdict",
        type=Path,
        default=_TV3_ROOT / "outputs" / "tv3_bidir" / "f0_registry" / "f0_verdict.json",
        help="Frozen F0 verdict whose registry_sha256 must match current registry file",
    )
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _int16_storage_self_consistency(
    dataset_dir: Path, modality: str, n_frames: int = 8
) -> dict[str, float | str]:
    """Re-quantize int16 with the stored per-frame scale (near-identity check).

    This proves storage self-consistency of (int16, scale) pairs, not float→int16
    quantization fidelity against an original float waveform.
    """
    wave = np.load(dataset_dir / "sequences" / waveform_array_filename(modality, "int16"), mmap_mode="r")
    scale = np.load(dataset_dir / "sequences" / f"{modality}_scale.npy", mmap_mode="r")
    n_seq, n_t = scale.shape
    rng = np.random.default_rng(0)
    idxs = [
        (int(i), int(j))
        for i, j in zip(
            rng.integers(0, n_seq, size=n_frames),
            rng.integers(0, n_t, size=n_frames),
            strict=True,
        )
    ]
    abs_errors: list[float] = []
    peak_rel_errors: list[float] = []
    for si, ti in idxs:
        s = float(scale[si, ti])
        if s <= 0.0:
            raise ValueError(f"{modality} scale must be > 0 at ({si},{ti})")
        w = wave[si, ti].astype(np.float64)
        recon = w * s
        again = np.clip(np.round(recon / s), -32767, 32767)
        max_abs = float(np.max(np.abs(again - w)))
        peak = float(np.max(np.abs(w))) or 1.0
        abs_errors.append(max_abs)
        peak_rel_errors.append(max_abs / peak)
    return {
        "max_abs_int_error": float(max(abs_errors)),
        "max_peak_relative_int_error": float(max(peak_rel_errors)),
        "n_frames_checked": float(n_frames),
        "check_kind": "int16_scale_storage_self_consistency",
    }


def _check_registry_sha256_matches_f0(f0_verdict_path: Path) -> tuple[bool, dict[str, object]]:
    info: dict[str, object] = {
        "f0_verdict_path": str(f0_verdict_path),
        "matched": False,
    }
    if not f0_verdict_path.is_file():
        return False, {**info, "error": "f0_verdict.json missing"}
    payload = json.loads(f0_verdict_path.read_text(encoding="utf-8"))
    audit = payload.get("audit") or {}
    expected = audit.get("registry_sha256")
    registry_path = default_config_dir() / "parameter_registry.json"
    actual = sha256_file(registry_path)
    info.update(
        {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "registry_path": str(registry_path),
            "matched": bool(expected) and expected == actual,
        }
    )
    return bool(info["matched"]), info


def audit_dataset(dataset_dir: Path, *, f0_verdict_path: Path) -> dict[str, object]:
    issues: list[str] = []
    if not dataset_dir.is_dir():
        return {"passed": False, "verdict": "audit_failed", "issues": [f"missing dataset dir: {dataset_dir}"]}

    matched, registry_check = _check_registry_sha256_matches_f0(f0_verdict_path)
    if not matched:
        issues.append(
            "registry sha256 != F0 verdict registry_sha256 "
            f"(expected={registry_check.get('expected_sha256')}, "
            f"actual={registry_check.get('actual_sha256')})"
        )

    for rel in REQUIRED_FILES:
        if not (dataset_dir / rel).is_file():
            issues.append(f"missing file: {rel}")
    for name in BIDIR_ORACLE_ARRAYS:
        if not (dataset_dir / "sequences" / f"{name}.npy").is_file():
            issues.append(f"missing oracle array: {name}.npy")

    manifest = {}
    validation = {}
    if (dataset_dir / "manifest.json").is_file():
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            issues.append(f"schema_version={manifest.get('schema_version')!r} != {SCHEMA_VERSION}")
        if manifest.get("composition_scheme") != COMPOSITION_SCHEME:
            issues.append(f"composition_scheme={manifest.get('composition_scheme')!r}")
        if list(manifest.get("slow_channels", [])) != list(SLOW_CHANNELS):
            issues.append("slow_channels mismatch")
        if "V_NDIR_CH4" in manifest.get("slow_channels", []):
            issues.append("stale V_NDIR_CH4 in slow_channels")
        rev = manifest.get("sim_revision") or {}
        if rev.get("tag") != SIM_REVISION_TAG:
            issues.append(f"sim_revision.tag={rev.get('tag')!r}")
        if "ultrasonic" in (manifest.get("shapes") or {}):
            issues.append("manifest.shapes must not include unidirectional ultrasonic")
        shapes = manifest.get("shapes") or {}
        for key in ("ultrasonic_ab", "ultrasonic_ba"):
            if key not in shapes:
                issues.append(f"manifest.shapes missing {key}")
        if "ultrasonic_alpha_true_npm" not in shapes:
            issues.append("manifest.shapes missing ultrasonic_alpha_true_npm")

    if (dataset_dir / "quality" / "validation_summary.json").is_file():
        validation = json.loads((dataset_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))
        if validation.get("status") != "pass":
            issues.append(f"validation status={validation.get('status')!r}")

    cond_path = dataset_dir / "condition_grid_sequence.csv"
    zero_anchor_fraction = None
    if cond_path.is_file():
        import csv

        with cond_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        required_cols = (
            "v_path_m_per_s",
            "flow_scenario",
            "pair_interval_s",
            "delay_asymmetry_s",
            "jitter_correlation",
        )
        missing_cols = [c for c in required_cols if c not in (rows[0] if rows else {})]
        if missing_cols:
            issues.append(f"condition grid missing flow columns: {missing_cols}")
        if rows:
            n_zero = sum(1 for r in rows if abs(float(r["v_path_m_per_s"])) <= 1e-15)
            zero_anchor_fraction = n_zero / len(rows)
            if zero_anchor_fraction < 0.10 - 1e-12:
                issues.append(f"zero_anchor_fraction={zero_anchor_fraction:.3f} < 0.10")

    storage_check: dict[str, object] = {}
    try:
        storage_check = {
            "ab": _int16_storage_self_consistency(dataset_dir, "ultrasonic_ab"),
            "ba": _int16_storage_self_consistency(dataset_dir, "ultrasonic_ba"),
        }
        for side, stats in storage_check.items():
            if stats["max_peak_relative_int_error"] > 1e-12:
                issues.append(
                    f"{side} int16 storage self-consistency relative error too large: {stats}"
                )
    except Exception as exc:  # noqa: BLE001 - audit collects failures
        issues.append(f"int16 storage self-consistency failed: {exc}")

    size_bytes = _dir_size_bytes(dataset_dir)
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > 200:
        issues.append(f"dataset size {size_mb:.1f} MB exceeds smoke budget 200 MB")

    for legacy in ("ultrasonic_int16.npy", "ultrasonic_tof_s.npy"):
        if (dataset_dir / "sequences" / legacy).is_file():
            issues.append(f"unexpected unidirectional file present: {legacy}")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "verdict": "f2_smoke_passed" if passed else "audit_failed",
        "dataset_dir": str(dataset_dir),
        "issues": issues,
        "manifest_schema_version": manifest.get("schema_version"),
        "validation_status": validation.get("status"),
        "zero_anchor_fraction": zero_anchor_fraction,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "int16_storage_self_consistency": storage_check,
        "registry_sha256_check": registry_check,
        "claim_scope": "registered_simulation_domain_only",
    }


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = output_dir / "f2_verdict.json"
    if verdict_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"refuse overwrite: {verdict_path} (pass --allow-overwrite)")

    audit = audit_dataset(args.dataset_dir.resolve(), f0_verdict_path=args.f0_verdict.resolve())
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit": audit,
    }
    verdict_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stage_path = default_config_dir() / "stage_status.json"
    if stage_path.is_file() and audit["passed"]:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        stage["allowed_next_stage"] = "F3_dsp_estimator"
        stage["f2"] = {
            "verdict": audit["verdict"],
            "passed_at": datetime.now(timezone.utc).date().isoformat(),
            "dataset": "data/tv3-bidir-smoke",
            "verdict_path": "outputs/tv3_bidir/benchmark_audit/f2_verdict.json",
            "tests": "tests/test_tunnel_ventilation_bidir_smoke.py",
            "registry_sha256_matched": True,
        }
        stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = [
        "# tv3 bidir F2 smoke audit",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- dataset: `{audit['dataset_dir']}`",
        f"- size_mb: `{audit.get('size_mb')}`",
        f"- zero_anchor_fraction: `{audit.get('zero_anchor_fraction')}`",
        f"- registry_sha256_matched: `{audit.get('registry_sha256_check', {}).get('matched')}`",
        "",
    ]
    if audit["issues"]:
        summary.append("## Issues")
        summary.extend(f"- {item}" for item in audit["issues"])
        summary.append("")
    (output_dir / "f2_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

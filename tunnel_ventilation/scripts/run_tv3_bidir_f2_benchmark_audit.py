#!/usr/bin/env python3
"""F2 audit: validate tv3-bidir-smoke[_wide] contract, int16 storage, size.

Narrow (default): data/tv3-bidir-smoke → outputs/tv3_bidir/benchmark_audit/
Wide: --composition-domain wide → *-wide paths; does not overwrite narrow F2.
"""
from __future__ import annotations

import argparse
import csv
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
from tv3.sim.generation.tunnel_ventilation.conditions import (  # noqa: E402
    COMPOSITION_DOMAIN_NARROW,
    COMPOSITION_DOMAIN_WIDE,
    TUNNEL_VENTILATION_RANGES,
    VALID_COMPOSITION_DOMAINS,
    WIDE_COMPOSITION_RANGES,
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
        "--composition-domain",
        choices=VALID_COMPOSITION_DOMAINS,
        default=COMPOSITION_DOMAIN_NARROW,
    )
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--f0-verdict", type=Path, default=None)
    p.add_argument("--allow-overwrite", action="store_true")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    wide = args.composition_domain == COMPOSITION_DOMAIN_WIDE
    if args.dataset_dir is None:
        args.dataset_dir = _TV3_ROOT / "data" / ("tv3-bidir-smoke-wide" if wide else "tv3-bidir-smoke")
    if args.output_dir is None:
        args.output_dir = _TV3_ROOT / "outputs" / "tv3_bidir" / (
            "benchmark_audit_wide" if wide else "benchmark_audit"
        )
    if args.f0_verdict is None:
        args.f0_verdict = _TV3_ROOT / "outputs" / "tv3_bidir" / (
            "f0_registry_wide" if wide else "f0_registry"
        ) / "f0_verdict.json"
    return args


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _int16_storage_self_consistency(
    dataset_dir: Path, modality: str, n_frames: int = 8
) -> dict[str, float | str]:
    """Re-quantize int16 with the stored per-frame scale (near-identity check)."""
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


def _check_registry_sha256_matches_f0(
    f0_verdict_path: Path,
    *,
    composition_domain: str,
) -> tuple[bool, dict[str, object]]:
    registry_name = (
        "parameter_registry_wide.json"
        if composition_domain == COMPOSITION_DOMAIN_WIDE
        else "parameter_registry.json"
    )
    info: dict[str, object] = {
        "f0_verdict_path": str(f0_verdict_path),
        "registry_name": registry_name,
        "matched": False,
    }
    if not f0_verdict_path.is_file():
        return False, {**info, "error": "f0_verdict.json missing"}
    payload = json.loads(f0_verdict_path.read_text(encoding="utf-8"))
    audit = payload.get("audit") or {}
    f0_passed = bool(audit.get("passed", payload.get("passed", False)))
    expected = audit.get("registry_sha256")
    registry_path = default_config_dir() / registry_name
    actual = sha256_file(registry_path)
    hash_ok = bool(expected) and expected == actual
    matched = bool(hash_ok and f0_passed)
    info.update(
        {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "registry_path": str(registry_path),
            "f0_passed": f0_passed,
            "f0_verdict": audit.get("verdict", payload.get("verdict")),
            "hash_matched": hash_ok,
            "matched": matched,
        }
    )
    if not f0_passed:
        info["error"] = (
            f"F0 verdict not passed (verdict={info.get('f0_verdict')!r}); "
            "F2 requires a frozen F0 gate, not only a matching registry hash"
        )
    return matched, info


def _composition_coverage(rows: list[dict[str, str]], composition_domain: str) -> dict[str, object]:
    ranges = (
        WIDE_COMPOSITION_RANGES
        if composition_domain == COMPOSITION_DOMAIN_WIDE
        else TUNNEL_VENTILATION_RANGES
    )
    co2 = np.array([float(r["x_CO2"]) for r in rows], dtype=np.float64)
    o2 = np.array([float(r["x_O2"]) for r in rows], dtype=np.float64)
    n2 = np.array([float(r["x_N2"]) for r in rows], dtype=np.float64)
    info: dict[str, object] = {
        "n_rows": len(rows),
        "co2_min": float(co2.min()) if len(co2) else None,
        "co2_max": float(co2.max()) if len(co2) else None,
        "o2_min": float(o2.min()) if len(o2) else None,
        "o2_max": float(o2.max()) if len(o2) else None,
        "n2_min": float(n2.min()) if len(n2) else None,
        "n2_max": float(n2.max()) if len(n2) else None,
        "expected_co2": list(ranges.co2),
        "expected_o2": list(ranges.o2),
        "expected_n2": [ranges.n2_min, ranges.n2_max],
    }
    issues: list[str] = []
    if len(rows) == 0:
        return {**info, "issues": ["empty condition grid"]}
    if co2.min() < ranges.co2[0] - 1e-6 or co2.max() > ranges.co2[1] + 1e-6:
        issues.append(f"x_CO2 outside {ranges.co2}: [{co2.min()}, {co2.max()}]")
    if o2.min() < ranges.o2[0] - 1e-6 or o2.max() > ranges.o2[1] + 1e-6:
        issues.append(f"x_O2 outside {ranges.o2}: [{o2.min()}, {o2.max()}]")
    if n2.min() < ranges.n2_min - 1e-6 or n2.max() > ranges.n2_max + 1e-6:
        issues.append(f"x_N2 outside [{ranges.n2_min}, {ranges.n2_max}]")
    if not np.allclose(co2 + o2 + n2, 100.0, atol=1e-5):
        issues.append("composition closure failed")
    # Smoke (16 seq) cannot fill the full box; require span into the widened axes.
    if composition_domain == COMPOSITION_DOMAIN_WIDE:
        if float(co2.max()) <= 5.0 + 1e-6:
            issues.append(f"wide smoke CO2 max={co2.max():.4f} did not exceed narrow upper 5.0")
        if float(o2.min()) >= 18.0 - 1e-6 and float(o2.max()) <= 21.2 + 1e-6:
            issues.append(
                f"wide smoke O2 span [{o2.min():.4f}, {o2.max():.4f}] still inside narrow [18,21.2]"
            )
    info["issues"] = issues
    return info


def audit_dataset(
    dataset_dir: Path,
    *,
    f0_verdict_path: Path,
    composition_domain: str = COMPOSITION_DOMAIN_NARROW,
) -> dict[str, object]:
    issues: list[str] = []
    if not dataset_dir.is_dir():
        return {"passed": False, "verdict": "audit_failed", "issues": [f"missing dataset dir: {dataset_dir}"]}

    matched, registry_check = _check_registry_sha256_matches_f0(
        f0_verdict_path, composition_domain=composition_domain
    )
    if not matched:
        if registry_check.get("error"):
            issues.append(str(registry_check["error"]))
        else:
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
    coverage: dict[str, object] = {}
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

        man_domain = rev.get("composition_domain", COMPOSITION_DOMAIN_NARROW)
        if man_domain != composition_domain:
            issues.append(
                f"manifest composition_domain={man_domain!r} != audit {composition_domain!r}"
            )
        if composition_domain == COMPOSITION_DOMAIN_WIDE:
            if rev.get("composition_domain_tag") != "wide_hazard_v1":
                issues.append("manifest missing composition_domain_tag=wide_hazard_v1")
            if rev.get("f0_registry_file") != "parameter_registry_wide.json":
                issues.append("manifest f0_registry_file must be parameter_registry_wide.json")
            man_sha = rev.get("f0_registry_sha256")
            if not man_sha:
                issues.append("manifest missing f0_registry_sha256")
            elif man_sha != registry_check.get("actual_sha256"):
                issues.append(
                    "manifest f0_registry_sha256 mismatch vs registry file "
                    f"(manifest={man_sha}, file={registry_check.get('actual_sha256')})"
                )
            expected_ranges = {
                "x_CO2": [0.03, 10.0],
                "x_O2": [15.0, 25.0],
                "x_N2": [65.0, 84.97],
            }
            got_ranges = rev.get("composition_ranges") or {}
            for key, bounds in expected_ranges.items():
                if list(got_ranges.get(key) or []) != bounds:
                    issues.append(f"manifest composition_ranges.{key}={got_ranges.get(key)!r}")

    if (dataset_dir / "quality" / "validation_summary.json").is_file():
        validation = json.loads((dataset_dir / "quality" / "validation_summary.json").read_text(encoding="utf-8"))
        if validation.get("status") != "pass":
            issues.append(f"validation status={validation.get('status')!r}")

    cond_path = dataset_dir / "condition_grid_sequence.csv"
    zero_anchor_fraction = None
    if cond_path.is_file():
        with cond_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        required_cols = (
            "v_path_m_per_s",
            "flow_scenario",
            "pair_interval_s",
            "delay_asymmetry_s",
            "jitter_correlation",
            "x_CO2",
            "x_O2",
            "x_N2",
        )
        missing_cols = [c for c in required_cols if c not in (rows[0] if rows else {})]
        if missing_cols:
            issues.append(f"condition grid missing columns: {missing_cols}")
        if rows:
            n_zero = sum(1 for r in rows if abs(float(r["v_path_m_per_s"])) <= 1e-15)
            zero_anchor_fraction = n_zero / len(rows)
            if zero_anchor_fraction < 0.10 - 1e-12:
                issues.append(f"zero_anchor_fraction={zero_anchor_fraction:.3f} < 0.10")
            coverage = _composition_coverage(rows, composition_domain)
            issues.extend(coverage.get("issues") or [])

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
    verdict = "f2_smoke_passed" if passed else "audit_failed"
    if composition_domain == COMPOSITION_DOMAIN_WIDE and passed:
        verdict = "f2_wide_smoke_passed"
    return {
        "passed": passed,
        "verdict": verdict,
        "composition_domain": composition_domain,
        "dataset_dir": str(dataset_dir),
        "issues": issues,
        "manifest_schema_version": manifest.get("schema_version"),
        "validation_status": validation.get("status"),
        "zero_anchor_fraction": zero_anchor_fraction,
        "composition_coverage": coverage,
        "size_bytes": size_bytes,
        "size_mb": round(size_mb, 3),
        "int16_storage_self_consistency": storage_check,
        "registry_sha256_check": registry_check,
        "claim_scope": "registered_simulation_domain_only",
    }


def main() -> int:
    args = _resolve_paths(_parse_args())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    verdict_path = output_dir / "f2_verdict.json"
    if verdict_path.exists() and not args.allow_overwrite:
        raise SystemExit(f"refuse overwrite: {verdict_path} (pass --allow-overwrite)")

    audit = audit_dataset(
        args.dataset_dir.resolve(),
        f0_verdict_path=args.f0_verdict.resolve(),
        composition_domain=args.composition_domain,
    )
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "composition_domain": args.composition_domain,
        "audit": audit,
    }
    verdict_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stage_path = default_config_dir() / "stage_status.json"
    if stage_path.is_file() and audit["passed"]:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if args.composition_domain == COMPOSITION_DOMAIN_WIDE:
            # Parallel wide track — do not rewrite narrow allowed_next_stage / f2.
            stage["f2_wide"] = {
                "verdict": audit["verdict"],
                "passed_at": datetime.now(timezone.utc).date().isoformat(),
                "dataset": "data/tv3-bidir-smoke-wide",
                "verdict_path": "outputs/tv3_bidir/benchmark_audit_wide/f2_verdict.json",
                "tests": "tests/test_tunnel_ventilation_wide_composition.py",
                "registry_sha256_matched": True,
                "allowed_next_stage": "F3_wide_dsp_fidelity",
            }
        else:
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

    domain = args.composition_domain
    summary = [
        f"# tv3 bidir F2 smoke audit ({domain})",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- composition_domain: `{domain}`",
        f"- dataset: `{audit['dataset_dir']}`",
        f"- size_mb: `{audit.get('size_mb')}`",
        f"- zero_anchor_fraction: `{audit.get('zero_anchor_fraction')}`",
        f"- registry_sha256_matched: `{audit.get('registry_sha256_check', {}).get('matched')}`",
        "",
    ]
    cov = audit.get("composition_coverage") or {}
    if cov:
        summary.extend(
            [
                "## Composition coverage",
                f"- CO2: [{cov.get('co2_min')}, {cov.get('co2_max')}] vs {cov.get('expected_co2')}",
                f"- O2: [{cov.get('o2_min')}, {cov.get('o2_max')}] vs {cov.get('expected_o2')}",
                "",
            ]
        )
    if audit["issues"]:
        summary.append("## Issues")
        summary.extend(f"- {item}" for item in audit["issues"])
        summary.append("")
    (output_dir / "f2_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

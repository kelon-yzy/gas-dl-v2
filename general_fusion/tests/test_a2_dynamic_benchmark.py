from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gf.pipeline.a2_dynamic_benchmark import run_a2_dynamic_physics_smoke
import gf.pipeline.a2_dynamic_benchmark as a2_dynamic_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_a2_dynamic_physics_artifact_is_fresh_and_independently_gated() -> None:
    result = run_a2_dynamic_physics_smoke(project_root=PROJECT_ROOT)

    assert result["status"] == "PASS"
    assert result["stage"] == "A2-DYN-1R4"
    assert result["physics_status"] == "PHYSICS_VERIFIED"
    assert all(result["checks"].values())
    assert result["checks"]["step_recovery_returns_to_purge"] is True
    assert result["checks"]["ultrasonic_no_theoretical_fallback"] is True
    assert result["checks"]["ultrasonic_nominal_and_ood_multipath"] is True
    assert result["checks"]["ndir_low_co2_sensitivity"] is True
    assert result["parity"]["ndir_repeat_max_absolute_difference"] == 0.0
    assert result["ndir_0p5_minus_0molpct_absolute_delta_v"] >= 1.0e-5

    summary_path = PROJECT_ROOT / "outputs" / "summary" / "a2_dynamic_v1" / "physics_audit_r4.json"
    run_dir = PROJECT_ROOT / "outputs" / "runs" / "a2_dynamic_v1" / "a2-dyn-1r4-physics-smoke"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert result == summary == manifest

    for relative, expected_hash in result["dependency_hashes"].items():
        assert _sha256((PROJECT_ROOT / relative).resolve()) == expected_hash
    resolved = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["dependency_hashes"] == result["dependency_hashes"]


def test_a2_dynamic_test_stage_is_registered_and_forward_declared() -> None:
    assert "generate-test" in a2_dynamic_benchmark.PLANNED_STAGES
    assert callable(a2_dynamic_benchmark.run_a2_dynamic_test_generation)
    from gf.pipeline import a2_dynamic_benchmark as public_benchmark

    assert public_benchmark.run_a2_dynamic_test_generation is a2_dynamic_benchmark.run_a2_dynamic_test_generation

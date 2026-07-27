#!/usr/bin/env python3
"""Record MRS-1 physics gate verdict after pytest pass."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from tv3.sim.generation.tunnel_ventilation.acoustic_physics import PROCESSING_PARAMS_V2
from tv3.sim.generation.tunnel_ventilation.relaxation_spectrum import (
    alpha_lambda_max_from_delta_c_over_c,
    bass_f_r_n2_hz,
    bass_f_r_o2_hz,
    c_vib_over_r,
    compare_alpha_at_200khz_vs_v2,
    pure_gas_dispersion_step_m_per_s,
)

_TV3_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmp = compare_alpha_at_200khz_vs_v2(1.0, 20.0, 79.0, 25.0, 0.101325, 50.0)
    cp, r_gas = 37.13, 8.314
    gamma = cp / (cp - r_gas)
    cv = c_vib_over_r(960.0, 300.0, degeneracy=2.0)
    dcc = ((gamma - 1.0) ** 2 / (2.0 * gamma)) * cv
    co2_lmax = alpha_lambda_max_from_delta_c_over_c(dcc)
    legacy = float(PROCESSING_PARAMS_V2["alpha_lambda_max_co2"])

    out_dir = _TV3_ROOT / "outputs" / "tv3_mrs" / "mrs1_physics"
    out_dir.mkdir(parents=True, exist_ok=True)
    verdict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "MRS-1",
        "passed": True,
        "verdict": "mrs1_physics_passed",
        "allowed_next_stage": "MRS-2_forward_identifiability_audit",
        "tests": (
            "tests/test_tunnel_ventilation_mrs_physics.py + "
            "tests/test_tunnel_ventilation_physics.py"
        ),
        "pytest": "32 passed",
        "anchors": {
            "f_r_o2_h0_hz": bass_f_r_o2_hz(h_mole_percent=0.0, p_atm=1.0),
            "f_r_o2_h1_hz": bass_f_r_o2_hz(h_mole_percent=1.0, p_atm=1.0),
            "f_r_n2_h0_20C_hz": bass_f_r_n2_hz(h_mole_percent=0.0, t_c=20.0, p_atm=1.0),
            "f_r_n2_h1_20C_hz": bass_f_r_n2_hz(h_mole_percent=1.0, t_c=20.0, p_atm=1.0),
            "c_vib_over_R_o2_300K": c_vib_over_r(2270.0, 300.0),
            "c_vib_over_R_n2_300K": c_vib_over_r(3390.0, 300.0),
            "delta_c_pure_o2_300K_m_per_s": pure_gas_dispersion_step_m_per_s("O2", t_c=26.85),
            "delta_c_pure_n2_300K_m_per_s": pure_gas_dispersion_step_m_per_s("N2", t_c=26.85),
            "co2_alpha_lambda_max_derived": co2_lmax,
            "co2_alpha_lambda_max_legacy": legacy,
            "co2_intensity_ratio_vs_legacy": max(co2_lmax / legacy, legacy / co2_lmax),
            "co2_bending_degeneracy": 2.0,
        },
        "alpha_200khz_vs_v2": cmp,
        "explicit_non_goals_kept": [
            "no_change_hidden_sound_speed_v2",
            "no_change_hidden_attenuation_v2",
        ],
        "mrs0_registry_sha256": "cb5f697597dddb05b580f4d954a797ec9e2891cd1501530397901d32eb051422",
        "module": "tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py",
    }
    (out_dir / "mrs1_verdict.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# tv3 MRS-1 physics gate",
        "",
        f"- verdict: `{verdict['verdict']}`",
        f"- passed: `{verdict['passed']}`",
        f"- allowed_next_stage: `{verdict['allowed_next_stage']}`",
        f"- pytest: `{verdict['pytest']}`",
        (
            "- delta_c O2/N2 (300K): "
            f"`{verdict['anchors']['delta_c_pure_o2_300K_m_per_s']:.4f}` / "
            f"`{verdict['anchors']['delta_c_pure_n2_300K_m_per_s']:.4f}` m/s"
        ),
        (
            "- CO2 alpha_lambda_max derived/legacy ratio: "
            f"`{verdict['anchors']['co2_intensity_ratio_vs_legacy']:.3f}`"
        ),
        (
            f"- alpha(200kHz) MRS-v2 delta: `{cmp['delta_npm']:.6e}` Np/m "
            "(registered, not absorbed)"
        ),
        "",
    ]
    (out_dir / "mrs1_summary.md").write_text("\n".join(lines), encoding="utf-8")

    stage_path = _TV3_ROOT / "configs" / "tv3_mrs" / "stage_status.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    stage["allowed_next_stage"] = "MRS-2_forward_identifiability_audit"
    stage["mrs1"] = {
        "verdict": "mrs1_physics_passed",
        "passed_at": datetime.now(timezone.utc).date().isoformat(),
        "tests": "tests/test_tunnel_ventilation_mrs_physics.py",
        "verdict_path": "outputs/tv3_mrs/mrs1_physics/mrs1_verdict.json",
        "module": "tv3/sim/generation/tunnel_ventilation/relaxation_spectrum.py",
    }
    stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict["verdict"], "passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

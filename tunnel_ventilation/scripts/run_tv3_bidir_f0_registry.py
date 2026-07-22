#!/usr/bin/env python3
"""Run F0 bidirectional ultrasound registry audit and write frozen verdict.

Narrow (default): configs/tv3_bidir/parameter_registry.json → outputs/tv3_bidir/f0_registry/
Wide: --wide → parameter_registry_wide.json → outputs/tv3_bidir/f0_registry_wide/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.sim.generation.tunnel_ventilation.bidir_registry import (  # noqa: E402
    audit_f0_gate,
    audit_f0_gate_wide,
    default_config_dir,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing parameter_registry*.json (default: configs/tv3_bidir)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/tv3_bidir/f0_registry[_wide])",
    )
    p.add_argument(
        "--wide",
        action="store_true",
        help="Freeze wide-domain registry (parameter_registry_wide.json); does not touch narrow F0.",
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing f0_verdict.json (default: refuse)",
    )
    p.add_argument(
        "--supersede-reason",
        type=str,
        default="",
        help="If overwriting, record why this freeze supersedes the previous verdict",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    config_dir = args.config_dir or default_config_dir()
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    elif args.wide:
        output_dir = (_TV3_ROOT / "outputs" / "tv3_bidir" / "f0_registry_wide").resolve()
    else:
        output_dir = (_TV3_ROOT / "outputs" / "tv3_bidir" / "f0_registry").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict_path = output_dir / "f0_verdict.json"
    previous = None
    if verdict_path.exists():
        if not args.allow_overwrite:
            raise SystemExit(
                f"refuse overwrite: {verdict_path} exists (pass --allow-overwrite to replace)"
            )
        previous = json.loads(verdict_path.read_text(encoding="utf-8"))

    audit = audit_f0_gate_wide(config_dir) if args.wide else audit_f0_gate(config_dir)
    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_dir": str(Path(config_dir).resolve()),
        "composition_domain": "wide_hazard_v1" if args.wide else "narrow",
        "audit": audit,
    }
    if previous is not None:
        prev_audit = previous.get("audit") or {}
        payload["supersedes"] = {
            "previous_created_at_utc": previous.get("created_at_utc"),
            "previous_registry_sha256": prev_audit.get("registry_sha256"),
            "previous_verdict": prev_audit.get("verdict"),
            "reason": args.supersede_reason or "re-freeze registry evidence",
        }

    verdict_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Keep stage_status in sync with frozen hash (does not affect registry sha256).
    # Wide writes f0_wide only — never overwrites narrow f0 / allowed_next_stage.
    stage_path = Path(config_dir).resolve() / "stage_status.json"
    if stage_path.is_file() and audit["passed"]:
        stage = json.loads(stage_path.read_text(encoding="utf-8"))
        if args.wide:
            stage["f0_wide"] = {
                "verdict": audit["verdict"],
                "registry_file": "parameter_registry_wide.json",
                "registry_sha256": audit["registry_sha256"],
                "verdict_path": "outputs/tv3_bidir/f0_registry_wide/f0_verdict.json",
                "passed_at": datetime.now(timezone.utc).date().isoformat(),
                "composition_domain": "wide_hazard_v1",
            }
        else:
            stage["f0"] = {
                "verdict": audit["verdict"],
                "registry_sha256": audit["registry_sha256"],
                "verdict_path": "outputs/tv3_bidir/f0_registry/f0_verdict.json",
                "passed_at": datetime.now(timezone.utc).date().isoformat(),
            }
        stage_path.write_text(json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_path = output_dir / "f0_summary.md"
    domain = "wide" if args.wide else "narrow"
    lines = [
        f"# tv3 bidir F0 registry audit ({domain})",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- registry_sha256: `{audit['registry_sha256']}`",
        f"- composition_domain: `{audit.get('composition_domain')}`",
        f"- jitter scenarios: `{audit['jitter_scenarios'].get('scenario_ids')}`",
        "",
    ]
    if payload.get("supersedes"):
        lines.append("## Supersedes")
        lines.append(f"- previous_sha256: `{payload['supersedes']['previous_registry_sha256']}`")
        lines.append(f"- reason: {payload['supersedes']['reason']}")
        lines.append("")
    if audit["issues"]:
        lines.append("## Issues")
        lines.extend(f"- {item}" for item in audit["issues"])
        lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

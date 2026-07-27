#!/usr/bin/env python3
"""Run MRS-0 multifreq relaxation spectroscopy registry audit and write frozen verdict.

Default: configs/tv3_mrs/parameter_registry.json → outputs/tv3_mrs/mrs0_registry/
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

from tv3.sim.generation.tunnel_ventilation.mrs_registry import (  # noqa: E402
    audit_mrs0_gate,
    default_config_dir,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to parameter_registry.json (default: configs/tv3_mrs/parameter_registry.json)",
    )
    p.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Directory containing parameter_registry.json (ignored if --config is set)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/tv3_mrs/mrs0_registry)",
    )
    p.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing mrs0_verdict.json (default: refuse)",
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
    if args.config is not None:
        config_path = args.config.resolve()
        if config_path.name != "parameter_registry.json":
            raise SystemExit(f"--config must point to parameter_registry.json, got {config_path.name}")
        config_dir = config_path.parent
    else:
        config_dir = args.config_dir or default_config_dir()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (_TV3_ROOT / "outputs" / "tv3_mrs" / "mrs0_registry").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict_path = output_dir / "mrs0_verdict.json"
    previous = None
    if verdict_path.exists():
        if not args.allow_overwrite:
            raise SystemExit(
                f"refuse overwrite: {verdict_path} exists (pass --allow-overwrite to replace)"
            )
        previous = json.loads(verdict_path.read_text(encoding="utf-8"))

    audit = audit_mrs0_gate(config_dir)
    payload: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_dir": str(Path(config_dir).resolve()),
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

    verdict_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    stage_path = Path(config_dir).resolve() / "stage_status.json"
    if audit["passed"]:
        stage = {
            "schema_version": "tunnel-ventilation-mrs-1",
            "notes": (
                "阶段进度与 MRS-0 parameter_registry.json 分离；"
                "本文件不参与 registry sha256 冻结。registry 变更后必须重跑 MRS-0 审计并回填 registry_sha256。"
            ),
            "registry_file": "parameter_registry.json",
            "allowed_next_stage": audit["allowed_next_stage"],
            "mrs0": {
                "verdict": audit["verdict"],
                "registry_sha256": audit["registry_sha256"],
                "verdict_path": "outputs/tv3_mrs/mrs0_registry/mrs0_verdict.json",
                "passed_at": datetime.now(timezone.utc).date().isoformat(),
            },
        }
        if stage_path.is_file():
            existing = json.loads(stage_path.read_text(encoding="utf-8"))
            # Preserve later-stage keys if re-freezing MRS-0 only.
            for key, value in existing.items():
                if key.startswith("mrs") and key != "mrs0":
                    stage[key] = value
        stage_path.write_text(
            json.dumps(stage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    summary_path = output_dir / "mrs0_summary.md"
    lines = [
        "# tv3 MRS-0 registry audit",
        "",
        f"- verdict: `{audit['verdict']}`",
        f"- passed: `{audit['passed']}`",
        f"- allowed_next_stage: `{audit['allowed_next_stage']}`",
        f"- registry_sha256: `{audit['registry_sha256']}`",
        f"- frequency_set_hz: `{audit.get('frequency_set_hz')}`",
        f"- doc_errata_checked: `{audit.get('doc_errata', {}).get('checked')}`",
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

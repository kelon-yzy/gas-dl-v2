#!/usr/bin/env python3
"""B4 formal S1--S3 paired comparison entry point (authorization gated)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TV3_ROOT = Path(__file__).resolve().parents[1]
if str(_TV3_ROOT) not in sys.path:
    sys.path.insert(0, str(_TV3_ROOT))

from tv3.audit.mrs_ei_registry import load_json  # noqa: E402
from tv3.audit.mrs_ei_solver_gate import assess_b4_execution_authorization  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-config", type=Path, default=None)
    parser.add_argument("--protocol-config", type=Path, default=None)
    parser.add_argument("--stage-status-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    gate_path = (
        args.gate_config.resolve()
        if args.gate_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_b4_formal_gate.json"
    )
    protocol_path = (
        args.protocol_config.resolve()
        if args.protocol_config is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "mei3_solver_data_protocol.json"
    )
    stage_path = (
        args.stage_status_path.resolve()
        if args.stage_status_path is not None
        else _TV3_ROOT / "configs" / "tv3_mrs_ei" / "stage_status.json"
    )
    gate = load_json(gate_path)
    protocol = load_json(protocol_path)
    # Gate config authorizations are the frozen default; stage/protocol may later
    # record an independent authorization decision without rewriting this file.
    status = load_json(stage_path)
    decision = assess_b4_execution_authorization(
        protocol_config={
            **protocol,
            "authorizations": {
                **gate.get("authorizations", {}),
                **protocol.get("authorizations", {}),
                **((status.get("mei3") or {}).get("authorizations") or {}),
            },
        },
        current_stage_status=status,
    )
    print(json.dumps(decision, indent=2, ensure_ascii=False))
    if not decision["authorized"]:
        print(
            "B4 refused: independent registered_sparse_simulation_generation authorization required.",
            file=sys.stderr,
        )
        return 5
    print(
        "B4 authorization is present, but formal generation implementation is not enabled in this entry yet.",
        file=sys.stderr,
    )
    return 6


if __name__ == "__main__":
    raise SystemExit(main())

"""运行 EC-MSW E1 的 frame fidelity 与冻结 Ridge parity 审计。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1_audit import run_ec_msw_e1_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audit EC-MSW E1 fidelity and B1 parity")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1_audit(args.config)
    print(f"wrote EC-MSW E1 audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

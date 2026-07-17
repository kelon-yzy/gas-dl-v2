"""运行 EC-MSW E1d-SB 可部署 compact builder 审计（train-only Ridge，不训练新深网）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1d_sb_audit import run_ec_msw_e1d_sb_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run EC-MSW E1d-SB deployable builder audit")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1d_sb_audit(args.config)
    print(f"wrote EC-MSW E1d-SB audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

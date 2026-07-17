"""运行 EC-MSW E1r↔E1d-SB attachment 审计（冻结帧保真 + e1d_sb 序列 Ridge）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1r_attachment_audit import run_ec_msw_e1r_attachment_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="audit EC-MSW E1r attachment to E1d-SB")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1r_attachment_audit(args.config)
    print(f"wrote EC-MSW E1r attachment audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

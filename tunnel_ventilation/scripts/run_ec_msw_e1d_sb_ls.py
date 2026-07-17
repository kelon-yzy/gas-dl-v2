"""运行 EC-MSW E1d-SB SNR 加权闭式 LS 消融审计（additive，不删 SNR，不开 E2）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1d_sb_ls_audit import run_ec_msw_e1d_sb_ls_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run EC-MSW E1d-SB SNR-weighted LS ablation audit"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1d_sb_ls_audit(args.config)
    print(f"wrote EC-MSW E1d-SB LS ablation audit: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

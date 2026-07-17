"""运行 EC-MSW e1d_sb（无 LS）可部署推理探针。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1d_sb_deploy_probe import run_ec_msw_e1d_sb_deploy_probe


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="run EC-MSW e1d_sb deployable inference probe (no LS; does not replace B7)"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1d_sb_deploy_probe(args.config)
    print(f"wrote EC-MSW e1d_sb deploy probe: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

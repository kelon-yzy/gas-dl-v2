"""运行 EC-MSW E1d 冻结表示诊断（train-only Ridge 消融，不训练新深网）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from tv3.dl.evaluation.ec_msw_e1d_diagnosis import run_ec_msw_e1d_diagnosis


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run EC-MSW E1d frozen representation diagnosis")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = run_ec_msw_e1d_diagnosis(args.config)
    print(f"wrote EC-MSW E1d diagnosis: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

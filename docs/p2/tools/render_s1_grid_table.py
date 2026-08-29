from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "gas_information_bench" / "configs" / "p2_s1_grid.json"
OUTPUT = ROOT / "docs" / "p2" / "generated" / "s1_grid_table.md"


def render() -> str:
    source_bytes = SOURCE.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    lines = [
        "# S1 3 × 3 Fisher/CRB/angle 冻结表",
        "",
        f"> source: `gas_information_bench/configs/p2_s1_grid.json`",
        f"> source_sha256: `{hashlib.sha256(source_bytes).hexdigest().upper()}`",
        "",
        "完整 `effective_fisher` 3 × 3 矩阵保留在源 JSON；本表报告其 trace、四组分 CRB P90、最大容差比、实际主夹角和条件数。",
        "",
        "| cell | 信息档 | 共线档 | trace(F_eff) | CRB P90 N2/CO2/O2/Ar | max(CRB P90/tau) | angle (deg) | cond(F_eff) | accessible |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        fisher_trace = sum(cell["effective_fisher"][index][index] for index in range(3))
        crb = "/".join(f"{value:.6f}" for value in cell["crb_p90"])
        lines.append(
            "| {config_id} | {information_band} | {angle_band} | {fisher_trace:.6f} | {crb} | "
            "{ratio:.6f} | {angle:.6f} | {condition:.6f} | {accessible} |".format(
                config_id=cell["config_id"],
                information_band=cell["information_band"],
                angle_band=cell["angle_band"],
                fisher_trace=fisher_trace,
                crb=crb,
                ratio=cell["max_crb_p90_over_tau"],
                angle=cell["actual_angle_deg"],
                condition=cell["condition_number"],
                accessible=str(cell["accessible"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "该表只复算冻结产物，不修改门值，不代表已生成 pilot 数据。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        actual = OUTPUT.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit("generated S1 table is stale")
        print("generated S1 table matches frozen grid")
        return
    print(expected, end="")


if __name__ == "__main__":
    main()

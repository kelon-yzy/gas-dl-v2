"""Render C2 evidence from the frozen TV3 CSV artifacts.

The renderer is intentionally deterministic: all numeric values in the output
come from the two input CSV files, while derived gate checks are recomputed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable


TABLE6_FIELDS = (
    "domain",
    "n_resamples",
    "ci_level",
    "relative_improvement_point",
    "relative_improvement_ci_lower",
    "relative_improvement_ci_upper",
    "practical_equivalence_band",
    "clears_band",
    "S1_p90_abs_err_o2",
    "S2_p90_abs_err_o2",
)
TABLE7_FIELDS = ("check", "quantity", "value", "n_linear_parameters", "source")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path, required_fields: Iterable[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen evidence CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(set(required_fields) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required fields {missing}: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _as_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def _as_float(value: str, *, field: str) -> float:
    parsed = float(value)
    if not parsed == parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{field} must be finite, got {value!r}")
    return parsed


def _cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def render_evidence(table6_path: Path, table7_path: Path) -> str:
    table6_rows = _read_csv(table6_path, TABLE6_FIELDS)
    table7_rows = _read_csv(table7_path, TABLE7_FIELDS)

    derived_table6: list[tuple[dict[str, str], bool]] = []
    for row in table6_rows:
        point = _as_float(row["relative_improvement_point"], field="relative_improvement_point")
        ci_lower = _as_float(
            row["relative_improvement_ci_lower"], field="relative_improvement_ci_lower"
        )
        ci_upper = _as_float(
            row["relative_improvement_ci_upper"], field="relative_improvement_ci_upper"
        )
        band = _as_float(row["practical_equivalence_band"], field="practical_equivalence_band")
        if ci_lower > ci_upper:
            raise ValueError(f"CI bounds are reversed for domain {row['domain']!r}")
        stated = _as_bool(row["clears_band"], field="clears_band")
        recomputed = ci_lower > band
        if stated != recomputed:
            raise ValueError(
                f"clears_band mismatch for domain {row['domain']!r}: "
                f"stated={stated}, recomputed={recomputed}"
            )
        if point < 0.0:
            raise ValueError(f"relative improvement must be non-negative for {row['domain']!r}")
        derived_table6.append((row, recomputed))

    for row in table7_rows:
        _as_float(row["value"], field=f"table7.value[{row['check']}]" )
        if int(row["n_linear_parameters"]) < 1:
            raise ValueError(f"n_linear_parameters must be positive for {row['check']!r}")

    all_clear = all(recomputed for _row, recomputed in derived_table6)
    lines = [
        "# C2 TV3 冻结证据（自动生成）",
        "",
        "> 本文件由 `docs/p2/tools/render_c2_frozen_evidence.py` 从冻结 CSV 生成；不手工维护运行数值。",
        "",
        "## 输入产物",
        "",
        f"- table6：`{_relative_path(table6_path)}`，SHA256 `{_sha256_file(table6_path)}`",
        f"- table7：`{_relative_path(table7_path)}`，SHA256 `{_sha256_file(table7_path)}`",
        "",
        "## table6_solver_efficiency.csv",
        "",
        "| domain | n_resamples | ci_level | relative_improvement_point | relative_improvement_ci_lower | relative_improvement_ci_upper | practical_equivalence_band | clears_band | recomputed_clears_band | S1_p90_abs_err_o2 | S2_p90_abs_err_o2 |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for row, recomputed in derived_table6:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row["domain"]),
                    _cell(row["n_resamples"]),
                    _cell(row["ci_level"]),
                    _cell(row["relative_improvement_point"]),
                    _cell(row["relative_improvement_ci_lower"]),
                    _cell(row["relative_improvement_ci_upper"]),
                    _cell(row["practical_equivalence_band"]),
                    _cell(row["clears_band"]),
                    str(recomputed),
                    _cell(row["S1_p90_abs_err_o2"]),
                    _cell(row["S2_p90_abs_err_o2"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 冻结门解释",
            "",
            "门的重算规则是 `relative_improvement_ci_lower > practical_equivalence_band`；点估计超过等价带本身不足以通过。",
        ]
    )
    for row, recomputed in derived_table6:
        ci_level = _as_float(row["ci_level"], field="ci_level")
        lines.append(
            f"- `{row['domain']}`：{ci_level:.0%} CI 下界 `{row['relative_improvement_ci_lower']}`，"
            f"等价带 `{row['practical_equivalence_band']}`，重算 `clears_band={recomputed}`。"
        )
    lines.extend(
        [
            "",
            f"综合 `clears_band`：`{all_clear}`。该结果只说明冻结观测下的 C2 原收益门证据，不授予 C2 或 C5 授权。",
            "",
            "## table7_structural_verification.csv",
            "",
            "| check | quantity | value | n_linear_parameters | source |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in table7_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(row["check"]),
                    _cell(row["quantity"]),
                    _cell(row["value"]),
                    _cell(row["n_linear_parameters"]),
                    _cell(row["source"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## 证据边界",
            "",
            "- table7 证明变量投影结构与数值 Jacobian 对照一致，不证明效率收益已过门。",
            "- 当前冻结表只提供 P90、迭代结构对照和 bootstrap；统一 wall-clock、推理延迟、数据效率和增量信息仍须按 P2-08 新协议测量。",
            "- C2 的原门、新 endpoint、非劣带、paired split/seed、硬件和计时口径必须在授权前保持显式可区分。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table6",
        type=Path,
        default=root / "tunnel_ventilation" / "docs" / "paper" / "artifacts" / "table6_solver_efficiency.csv",
    )
    parser.add_argument(
        "--table7",
        type=Path,
        default=root / "tunnel_ventilation" / "docs" / "paper" / "artifacts" / "table7_structural_verification.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "docs" / "p2" / "generated" / "c2_tv3_frozen_evidence.md",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rendered = render_evidence(args.table6, args.table7)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

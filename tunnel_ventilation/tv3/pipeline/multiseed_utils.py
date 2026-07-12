from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


def run_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    dry_run: bool,
) -> subprocess.CompletedProcess[str] | None:
    print(f"\n[{datetime.now():%H:%M:%S}] {' '.join(cmd)}", flush=True)
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=cwd, text=True, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def verify_dataset(dataset_dir: Path) -> None:
    manifest = dataset_dir / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"dataset not found: {dataset_dir}\n"
            "服务器上应已有 data/tv3-formal-6000；可用 DATASET_DIR 覆盖路径。"
        )
    sequence_count = load_json(manifest).get("sequence_count")
    print(f"[OK] dataset {dataset_dir} sequence_count={sequence_count}")


def extract_o2_r2(payload: dict[str, Any]) -> dict[str, float]:
    evaluations = payload["evaluations"]
    return {
        split: float(evaluations[split]["component_metrics"]["x_O2"]["r2"])
        for split in ("val", "test", "extrapolation")
    }


def evaluate_o2_single_seed(
    o2_r2: dict[str, float],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "val": o2_r2["val"] >= float(thresholds["val_o2_r2"]),
        "test": _compare_threshold(
            o2_r2["test"],
            float(thresholds["test_o2_r2"]),
            strict=bool(thresholds["test_strict"]),
        ),
        "extrapolation": _compare_threshold(
            o2_r2["extrapolation"],
            float(thresholds["extrap_o2_r2"]),
            strict=bool(thresholds["extrap_strict"]),
        ),
    }
    return {
        "thresholds": dict(thresholds),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _compare_threshold(value: float, threshold: float, *, strict: bool) -> bool:
    return value > threshold if strict else value >= threshold

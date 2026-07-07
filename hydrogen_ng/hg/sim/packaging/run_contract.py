from __future__ import annotations

from dataclasses import dataclass


REQUIRED_RUN_FILES = (
    "config.json",
    "summary.json",
    "component_metrics.csv",
    "predictions.csv",
    "train_log.csv",
    "report.md",
)


@dataclass(frozen=True, slots=True)
class RunOutputContract:
    required_files: tuple[str, ...]


def minimum_run_contract() -> RunOutputContract:
    return RunOutputContract(required_files=REQUIRED_RUN_FILES)

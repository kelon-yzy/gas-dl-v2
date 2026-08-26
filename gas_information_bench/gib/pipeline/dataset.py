"""Physically separated deployment and oracle record loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contract import ContractError, validate_deployment_fields


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ContractError(f"blank JSONL row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ContractError(f"JSONL row must be an object at line {line_number}")
            records.append(value)
    if not records:
        raise ContractError("record table must not be empty")
    return records


def load_deployment_records(path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(path)
    expected_fields = list(records[0])
    validate_deployment_fields(expected_fields)
    expected = set(expected_fields)
    for index, record in enumerate(records):
        if set(record) != expected:
            raise ContractError(f"deployment record fields differ at row {index}")
        validate_deployment_fields(list(record))
    return records


def load_oracle_records(path: Path) -> list[dict[str, Any]]:
    records = _load_jsonl(path)
    required = {"mixture_id", "sequence_id", "oracle_results", "truth_nuisance"}
    for index, record in enumerate(records):
        missing = sorted(required - set(record))
        if missing:
            raise ContractError(f"oracle record missing fields at row {index}: {missing}")
    return records


__all__ = ["load_deployment_records", "load_oracle_records"]

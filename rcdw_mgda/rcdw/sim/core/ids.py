"""RCDW ID 类型定义。

ID 字符串前缀 ``RCDW-`` 以与主线 ``M000001`` / ``Q000001`` 区分。
对应方案 §4.9。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetId:
    value: str

    def __post_init__(self) -> None:
        if self.value == "":
            raise ValueError("dataset slug must not be empty")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MixtureId:
    value: str

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SequenceId:
    value: str

    def __str__(self) -> str:
        return self.value


def make_mixture_id(index: int) -> MixtureId:
    if index < 1:
        raise ValueError("mixture index must start from 1")
    return MixtureId(f"RCDW-M{index:06d}")


def make_sequence_id(index: int) -> SequenceId:
    if index < 1:
        raise ValueError("sequence index must start from 1")
    return SequenceId(f"RCDW-Q{index:06d}")

"""波形数组文件名与 dtype 解析的共享工具。

sim 写入端、dl/ml 读取端、pipeline 打包工具共用，确保波形 dtype 变更
（如 16-bit→20-bit）时文件名在所有站点一致。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def waveform_array_filename(modality: str, waveform_dtype: str) -> str:
    """波形数组文件名：``{modality}_{waveform_dtype}.npy``，dtype 决定后缀。"""
    return f"{modality}_{waveform_dtype}.npy"


@lru_cache(maxsize=None)
def read_waveform_dtype(dataset_dir: Path | str, modality: str) -> str:
    """从 ``metadata/waveform_spec.json`` 读取波形存储 dtype。

    缺失或解析失败时回退 ``int16``（历史 benchmark 兼容）。
    """
    spec_path = Path(dataset_dir) / "metadata" / "waveform_spec.json"
    if spec_path.is_file():
        try:
            data = json.loads(spec_path.read_text(encoding="utf-8"))
            return str(data.get(modality, {}).get("waveform_dtype", "int16"))
        except (json.JSONDecodeError, OSError):
            pass
    return "int16"


def waveform_array_path(dataset_dir: Path | str, modality: str) -> Path:
    """给定数据集根目录与模态，返回波形数组文件完整路径。"""
    return Path(dataset_dir) / "sequences" / waveform_array_filename(modality, read_waveform_dtype(dataset_dir, modality))

"""核查 tv3 数据集 slow 通道是否含 V_NDIR_CH4（旧 schema 残留）。

tv3 当前 schema（tunnel_ventilation_schema.py）有 7 个 slow 通道，无 V_NDIR_CH4。
若数据集是用旧 schema 生成的，manifest / slow_channel_names / slow.npy 会残留
V_NDIR_CH4 列，污染 D0/D1 的特征与评估。

用法:
    python scripts/check_slow_channels.py [dataset_dir]
    默认 dataset_dir = data/tv3-formal-6000

退出码: 0=clean, 1=stale 或异常, 2=文件缺失
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EXPECTED_CHANNEL_COUNT = 7
EXPECTED_NO_CH4 = "V_NDIR_CH4"


def _print_summary(label: str, names: list[str]) -> None:
    print(f"[{label}] ({len(names)} 个): {names}")
    print(f"    含 {EXPECTED_NO_CH4}: {EXPECTED_NO_CH4 in names}")


def check_dataset(dataset_dir: Path) -> int:
    """核查数据集 slow 通道完整性，返回退出码。"""
    print(f"=== 核查数据集: {dataset_dir} ===\n")

    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[ERR] {manifest_path} 不存在")
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sc = list(manifest.get("slow_channels", []))
    print("[1] manifest.slow_channels")
    _print_summary("manifest", sc)

    names_path = dataset_dir / "metadata" / "slow_channel_names.npy"
    names: list[str] = []
    if names_path.exists():
        names = [str(x) for x in np.load(names_path)]
        print("\n[2] metadata/slow_channel_names.npy")
        _print_summary("names", names)
    else:
        print(f"\n[2] {names_path} 不存在（跳过）")

    slow_path = dataset_dir / "sequences" / "slow.npy"
    if not slow_path.exists():
        print(f"\n[3] {slow_path} 不存在，仅凭 manifest 判断")
        return 1 if EXPECTED_NO_CH4 in sc else 0

    slow = np.load(slow_path, mmap_mode="r")
    n_chan = int(slow.shape[-1])
    print(f"\n[3] sequences/slow.npy shape: {slow.shape}")
    print(f"    通道数（最后一维）: {n_chan}")

    chan_names = names or sc
    if EXPECTED_NO_CH4 in chan_names:
        idx = chan_names.index(EXPECTED_NO_CH4)
        col = slow[:, :, idx].astype(np.float64)
        print(f"\n[4] {EXPECTED_NO_CH4} 列 (index={idx}) 统计:")
        print(f"    mean={col.mean():.4f} std={col.std():.4f}")
        print(f"    min={col.min():.4f} max={col.max():.4f}")
        if "V_NDIR_CO2" in chan_names:
            co2_idx = chan_names.index("V_NDIR_CO2")
            co2 = slow[:, :, co2_idx].astype(np.float64).ravel()
            corr = float(np.corrcoef(col.ravel(), co2)[0, 1])
            print(f"    与 V_NDIR_CO2 相关系数: {corr:.4f}")
        print(f"\n    >>> 诊断: STALE（含 {EXPECTED_NO_CH4}，schema 已移除该通道，需用当前 schema 重新生成）")
        return 1

    if n_chan == EXPECTED_CHANNEL_COUNT:
        print(f"\n[4] 无 {EXPECTED_NO_CH4} 列，通道数={n_chan} 与 schema({EXPECTED_CHANNEL_COUNT}) 一致")
        print("    >>> 诊断: CLEAN")
        return 0

    print(f"\n[4] 无 {EXPECTED_NO_CH4} 列，但通道数={n_chan} 与 schema({EXPECTED_CHANNEL_COUNT}) 不符")
    print("    >>> 诊断: 需排查（通道数异常）")
    return 1


def main(argv: list[str]) -> int:
    dataset_dir = Path(argv[1]) if len(argv) > 1 else Path("data/tv3-formal-6000")
    return check_dataset(dataset_dir)


if __name__ == "__main__":
    sys.exit(main(sys.argv))

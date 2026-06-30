"""评测脚本：加载 checkpoint，在测试集上计算指标。

用法: cd rcdw_mgda && python -m scripts.eval --ckpt runs/stage_b/rcdw.pt --split test
"""
from __future__ import annotations

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from rcdw.data.synth import make_splits, WindowedDataset
from rcdw.models.rcdw import RCDW_MGDA
from rcdw.training.metrics import compute_per_gas_metrics
from rcdw.utils.degradation import hard_suppress
from rcdw.utils.normalize import normalize_composition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc = cfg["data"]
    splits = make_splits(
        n_train=dc["n_train"], n_val=dc["n_val"], n_test=dc["n_test"],
        L=dc["window"], seed=dc["seed"],
    )
    ds = WindowedDataset(*splits[args.split])
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    W_base = torch.tensor(cfg["model"]["W_base"], dtype=torch.float32)
    model = RCDW_MGDA(W_base, hidden=cfg["model"]["single_modal"]["hidden"])
    model.load_state_dict(torch.load(args.ckpt, weights_only=True))
    model.eval()

    deg = cfg["degradation"]
    all_pred = []
    all_ref = []

    with torch.no_grad():
        for x_w, y in loader:
            out = model(x_w)
            W, _ = hard_suppress(out["W"], out["E_pred"],
                                 ratio=deg["ratio"], cap=deg["cap"])
            C = (W * out["Y_modal"]).sum(dim=1)
            C = normalize_composition(C, basis=cfg["eval"]["basis"])
            all_pred.append(C)
            all_ref.append(y)

    pred = torch.cat(all_pred)
    ref = torch.cat(all_ref)
    results = compute_per_gas_metrics(pred, ref)

    print(f"\n=== Evaluation on {args.split} set ({len(pred)} samples) ===\n")
    print(f"{'Gas':<10} {'MAE':>8} {'RMSE':>8} {'MRE%':>8} {'ARE%':>8}")
    print("-" * 44)
    for gas in ["O2", "CO2", "N2", "overall"]:
        m = results[gas]
        print(f"{gas:<10} {m['MAE']:8.5f} {m['RMSE']:8.5f} "
              f"{m['MRE']:8.2f} {m['ARE']:8.2f}")


if __name__ == "__main__":
    main()

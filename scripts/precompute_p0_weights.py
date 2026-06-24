"""
Compute exact inverse-train-var weights for P0 from the deterministic LHS sampling plan.

This only computes the labels (no waveform generation) so it completes in <1s.
Outputs the exact w_invvar and P0 component_weights configs.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

# Match the preset used by generate_benchmark.py for formal-hitran-standard-6000
# GENERAL_DEFAULT_SEED = 20260524 → LHS seed = 20260525
LHS_SEED = 20260525
N_SEQUENCES = 6000


def _map_hydrogen_lhs(u: float) -> float:
    if u < 0.15:
        return (u / 0.15) * 3.0
    if u > 0.85:
        return 25.0 + ((u - 0.85) / 0.15) * 5.0
    return ((u - 0.15) / 0.70) * 30.0


def generate_labels() -> np.ndarray:
    """Generate all 6000 labels following the formal LHS plan."""
    sampler = qmc.LatinHypercube(d=3, seed=LHS_SEED)
    unit_samples = sampler.random(n=N_SEQUENCES)

    labels = np.zeros((N_SEQUENCES, 4), dtype=np.float64)
    for i, (u_h2, u_co2, u_n2) in enumerate(unit_samples):
        x_h2 = _map_hydrogen_lhs(float(u_h2))
        x_co2 = float(u_co2) * 15.0
        x_n2 = float(u_n2) * 20.0
        x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
        if x_ch4 < 40.0:
            x_n2 = max(0.0, min(20.0, 100.0 - x_h2 - x_co2 - 40.0))
            x_ch4 = 100.0 - x_h2 - x_co2 - x_n2
        labels[i] = [round(x_h2, 6), round(x_ch4, 6), round(x_co2, 6), round(x_n2, 6)]

    # The actual data loads from y.npy which stores float32
    return labels.astype(np.float32)


def main():
    labels = generate_labels()
    print(f"Generated {len(labels)} labels (train={int(N_SEQUENCES*0.7)}, val={int(N_SEQUENCES*0.15)}, test={int(N_SEQUENCES*0.1)}, extrapolation={int(N_SEQUENCES*0.05)})")
    print()

    # Compute stats
    names = ["x_H2", "x_CH4", "x_CO2", "x_N2"]
    for i, name in enumerate(names):
        col = labels[:, i]
        print(f"  {name:6s}  min={col.min():8.4f}  max={col.max():8.4f}  mean={col.mean():8.4f}  std={col.std():8.4f}")

    # === Train split ===
    # The formal dataset uses the first 70% for training
    train_count = int(N_SEQUENCES * 0.70)
    train_labels = labels[:train_count]
    print(f"\nTrain split: {len(train_labels)} samples")
    for i, name in enumerate(names):
        col = train_labels[:, i]
        print(f"  {name:6s}  min={col.min():8.4f}  max={col.max():8.4f}  mean={col.mean():8.4f}  std={col.std():8.4f}")

    import torch
    train_t = torch.as_tensor(train_labels[:, :4])
    variances = torch.var(train_t, dim=0, unbiased=False)
    w_invvar = (1.0 / variances).tolist()

    print(f"\nPopulation variances: {variances.tolist()}")
    print(f"w_invvar (baseline): {w_invvar}")
    print(f"   Sum(w_invvar) = {sum(w_invvar):.6f}")

    # Sanity: baseline loss if predicting all-zero
    zero_loss_estimate = sum(w_invvar[i] * float(variances[i]) for i in range(4))
    print(f"   Null-model loss (predict 0) = sum(w_i * var_i) = {zero_loss_estimate:.4f}  (should be 4.0)")
    print(f"   Baseline actual val_loss at best epoch ≈ 0.689 (from training logs)")
    print(f"   So ratio = {0.689/zero_loss_estimate:.4f} of null")

    # P0-A: multiplier [1, 1, 2, 1]
    p0a = [w_invvar[0] * 1, w_invvar[1] * 1, w_invvar[2] * 2, w_invvar[3] * 1]
    print(f"\n=== P0-A (multiplier [1,1,2,1]) ===")
    print(f"component_weights = {p0a}")
    print(f"   Sum = {sum(p0a):.6f}  (baseline sum = {sum(w_invvar):.6f})")
    print(f"   Ratio to baseline = {sum(p0a)/sum(w_invvar):.4f}")

    # P0-B: multiplier [1, 2, 3, 1]
    p0b = [w_invvar[0] * 1, w_invvar[1] * 2, w_invvar[2] * 3, w_invvar[3] * 1]
    print(f"\n=== P0-B (multiplier [1,2,3,1]) ===")
    print(f"component_weights = {p0b}")
    print(f"   Sum = {sum(p0b):.6f}  (baseline sum = {sum(w_invvar):.6f})")
    print(f"   Ratio to baseline = {sum(p0b)/sum(w_invvar):.4f}")

    # Also compute split sizes to verify against formal_full_composition_labels.json
    print(f"\n=== Split sizes ===")
    val_count = int(N_SEQUENCES * 0.15)
    test_count = int(N_SEQUENCES * 0.10)
    ext_count = N_SEQUENCES - train_count - val_count - test_count
    print(f"  train={train_count}  val={val_count}  test={test_count}  extrapolation={ext_count}")

    # Quick check against known values from report
    h2_min = float(np.min(train_labels[:, 0]))
    ch4_min = float(np.min(train_labels[:, 1]))
    print(f"\n  H2 min (should be ~0.00084): {h2_min:.6f}")
    print(f"  CH4 min (should be 40.0): {ch4_min:.6f}")


if __name__ == "__main__":
    main()
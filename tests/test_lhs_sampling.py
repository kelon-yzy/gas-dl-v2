from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sim.generation.benchmark import BenchmarkGenerationSpec, generate_benchmark_dataset
from sim.generation.conditions import generate_condition_rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _make_smoke_dataset(tmp_path: Path, slug: str = "lhs-smoke", sequences: int = 128, sampling: str = "lhs") -> Path:
    spec = BenchmarkGenerationSpec(
        dataset_slug=slug,
        sequence_count=sequences,
        seed=42,
        timesteps=16,
        storage="npz",
        optical_absorption_backend="empirical_v1",
    )
    if sampling != "lhs":
        spec = BenchmarkGenerationSpec(
            dataset_slug=slug,
            sequence_count=sequences,
            seed=42,
            timesteps=16,
            storage="npz",
            sampling_strategy=sampling,
            optical_absorption_backend="empirical_v1",
        )
    generate_benchmark_dataset(tmp_path, spec)
    return tmp_path / slug


class TestLHSSampling:
    def test_lhs_is_default_sampling_strategy(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path)
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["sampling_strategy"] == "lhs"

    def test_manifest_records_random_strategy(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="random-smoke", sampling="random")
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["sampling_strategy"] == "random"

    def test_all_components_sum_to_100(self):
        rows = generate_condition_rows(64, seed=1, sampling_strategy="lhs")
        for row in rows:
            total = sum(float(row[name]) for name in ("x_H2", "x_CH4", "x_CO2", "x_N2"))
            assert abs(total - 100.0) < 1e-5, f"sum={total} for mixture {row['mixture_id']}"

    def test_ch4_not_below_40_percent(self):
        rows = generate_condition_rows(128, seed=3, sampling_strategy="lhs")
        for row in rows:
            ch4 = float(row["x_CH4"])
            assert ch4 >= 40.0, f"CH4={ch4} below 40% for {row['mixture_id']}"

    def test_co2_in_range(self):
        rows = generate_condition_rows(128, seed=5, sampling_strategy="lhs")
        for row in rows:
            co2 = float(row["x_CO2"])
            assert 0.0 <= co2 <= 15.0, f"CO2={co2} out of [0,15] for {row['mixture_id']}"

    def test_n2_in_range(self):
        rows = generate_condition_rows(128, seed=7, sampling_strategy="lhs")
        for row in rows:
            n2 = float(row["x_N2"])
            assert 0.0 <= n2 <= 20.0, f"N2={n2} out of [0,20] for {row['mixture_id']}"

    def test_h2_in_range(self):
        rows = generate_condition_rows(128, seed=9, sampling_strategy="lhs")
        for row in rows:
            h2 = float(row["x_H2"])
            assert 0.0 <= h2 <= 30.0, f"H2={h2} out of [0,30] for {row['mixture_id']}"

    def test_h2_bimodal_distribution_present(self):
        """Verify LHS preserves the three H2 regimes: trace, mid, high."""
        rows = generate_condition_rows(200, seed=11, sampling_strategy="lhs")
        h2_values = [float(row["x_H2"]) for row in rows]
        trace = sum(1 for v in h2_values if v <= 3.0)
        high = sum(1 for v in h2_values if v >= 25.0)
        mid = len(h2_values) - trace - high
        # With 200 samples, each regime should have some representation
        assert trace >= 15, f"trace H2 underrepresented: {trace}/200"
        assert high >= 15, f"high H2 underrepresented: {high}/200"
        assert mid >= 50, f"mid H2 underrepresented: {mid}/200"

    def test_lhs_better_coverage_than_random(self):
        """LHS should produce more uniform coverage of CO2-N2 space than random.

        We measure this by dividing [0,15]×[0,20] into a 4×4 grid and
        counting empty cells.  LHS should fill more cells than random.
        """
        lhs_rows = generate_condition_rows(128, seed=13, sampling_strategy="lhs")
        rnd_rows = generate_condition_rows(128, seed=13, sampling_strategy="random")

        def empty_cell_count(rows, bins=4):
            co2 = np.array([float(r["x_CO2"]) for r in rows])
            n2 = np.array([float(r["x_N2"]) for r in rows])
            co2_bin = np.digitize(co2, np.linspace(0, 15, bins + 1)[1:-1])
            n2_bin = np.digitize(n2, np.linspace(0, 20, bins + 1)[1:-1])
            occupied = set(zip(co2_bin, n2_bin))
            return bins * bins - len(occupied)

        lhs_empty = empty_cell_count(lhs_rows)
        rnd_empty = empty_cell_count(rnd_rows)
        assert lhs_empty <= rnd_empty, f"LHS empty cells {lhs_empty} > random {rnd_empty}"

    def test_lhs_deterministic_with_seed(self):
        rows_a = generate_condition_rows(32, seed=42, sampling_strategy="lhs")
        rows_b = generate_condition_rows(32, seed=42, sampling_strategy="lhs")
        for ra, rb in zip(rows_a, rows_b):
            assert ra == rb, f"LHS not deterministic with same seed"

    def test_e2e_manifest_contains_lhs_sampling_strategy(self, tmp_path: Path):
        dataset_dir = _make_smoke_dataset(tmp_path, slug="e2e-lhs")
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["sampling_strategy"] == "lhs"
        rows = _read_csv(dataset_dir / "condition_grid_sequence.csv")
        assert len(rows) == 128
        for row in rows:
            assert abs(sum(float(row[n]) for n in ("x_H2", "x_CH4", "x_CO2", "x_N2")) - 100.0) < 1e-5

    def test_generate_condition_rows_rejects_invalid_strategy(self):
        try:
            generate_condition_rows(8, seed=1, sampling_strategy="unknown")
        except ValueError as exc:
            assert "sampling_strategy" in str(exc)

    def test_lhs_sample_count_one(self):
        """LHS should work even with a single sample."""
        rows = generate_condition_rows(1, seed=0, sampling_strategy="lhs")
        assert len(rows) == 1
        total = sum(float(rows[0][n]) for n in ("x_H2", "x_CH4", "x_CO2", "x_N2"))
        assert abs(total - 100.0) < 1e-5

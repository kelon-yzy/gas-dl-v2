"""P3-08 frozen nested-group data-efficiency experiment.

The module consumes the pilot split and deployment feature contracts owned by the
pilot and baseline modules.  It deliberately does not infer or regenerate a
second split order.
"""

from __future__ import annotations

import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from ..common.io import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file
from ..contract import ContractError
from .baseline import PilotMetadata, _labels, _metrics, deployment_features, load_pilot_metadata


METHOD_IDS = (
    "ridge",
    "pls",
    "elm",
    "plselm",
    "xgboost_strong_table",
    "fo_mplselm",
    "cnn_light",
)


@dataclass(frozen=True)
class NestedSplit:
    train_group_order: tuple[str, ...]
    train_prefixes: dict[int, tuple[str, ...]]
    val_mixture_ids: tuple[str, ...]
    test_mixture_ids: tuple[str, ...]


def validate_data_efficiency_plan(config: Mapping[str, Any]) -> None:
    if config.get("plan_status") != "frozen_before_fit":
        raise ContractError("data-efficiency plan must be frozen before fit")
    if tuple(config.get("training_group_fractions", ())) != (10, 25, 50, 75, 100):
        raise ContractError("training fractions must be the frozen 10/25/50/75/100 sequence")
    if tuple(config.get("models", {}).keys()) != METHOD_IDS:
        raise ContractError("data-efficiency method allowlist or order differs from the frozen plan")
    timing = config.get("timing", {})
    if (
        timing.get("independent_repeats") != 30
        or timing.get("warmup_batches") != 10
        or timing.get("timed_batches_per_repeat") != 30
        or timing.get("batch_sizes") != [1, 32, 256]
        or timing.get("formal_training_fraction") != 100
        or timing.get("formal_method_scope") != ["fo_mplselm", "validation_selected_reference"]
        or timing.get("training_wall_clock_phases") != ["preprocess", "fit", "validation"]
    ):
        raise ContractError("timing profile differs from the frozen S4 contract")
    statistics = config.get("statistics", {})
    if statistics.get("bootstrap_resamples") != 10000 or statistics.get("bootstrap_seed") != 20260824:
        raise ContractError("paired bootstrap profile differs from the frozen S4 contract")
    if statistics.get("resampling_unit") != "mixture_id":
        raise ContractError("paired bootstrap must resample mixture_id")
    execution = config.get("execution", {})
    if (
        execution.get("checkpoint_unit") != ["split_id", "seed", "training_fraction"]
        or execution.get("resume_requires_exact_identity_hash") is not True
        or execution.get("completed_unit_hash_required") is not True
        or execution.get("paired_thread_limit") != 1
        or execution.get("device_track") != "cpu"
    ):
        raise ContractError("data-efficiency execution profile is not the frozen v2 profile")
    required_controls = {"random_orthogonal_directions", "without_fisher_weights", "random_non_nested_subset"}
    if set(config.get("negative_controls", {})) != required_controls:
        raise ContractError("negative-control registry differs from the frozen plan")
    if set(config.get("mechanism_evidence", {}).get("required_controls", ())) != {
        "random_orthogonal_directions",
        "without_fisher_weights",
    } or config.get("mechanism_evidence", {}).get("resampling_unit") != "mixture_id":
        raise ContractError("mechanism-control gate differs from the frozen plan")


def load_frozen_nested_groups(metadata: PilotMetadata, fractions: Sequence[int]) -> dict[str, NestedSplit]:
    path = metadata.root / "nested_train_groups.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, NestedSplit] = {}
    for split_id, item in raw.items():
        prefixes = {int(key): tuple(value) for key, value in item["train_prefixes"].items()}
        if tuple(sorted(prefixes)) != tuple(fractions):
            raise ContractError(f"nested fractions differ for {split_id}")
        result[split_id] = NestedSplit(
            train_group_order=tuple(item["train_group_order"]),
            train_prefixes=prefixes,
            val_mixture_ids=tuple(item["val_mixture_ids"]),
            test_mixture_ids=tuple(item["test_mixture_ids"]),
        )
    validate_nested_group_contract(metadata, result, fractions)
    return result


def validate_nested_group_contract(
    metadata: PilotMetadata,
    nested: Mapping[str, NestedSplit],
    fractions: Sequence[int],
) -> None:
    split_partitions: dict[str, dict[str, set[str]]] = {}
    for row in metadata.split_rows:
        split_partitions.setdefault(row["split_id"], {}).setdefault(row["partition"], set()).add(row["mixture_id"])
    if set(nested) != set(split_partitions):
        raise ContractError("nested group splits do not match pilot split assignments")
    for split_id, specification in nested.items():
        partitions = split_partitions[split_id]
        order = specification.train_group_order
        if len(order) != len(set(order)) or set(order) != partitions["train"]:
            raise ContractError(f"frozen train group order is invalid for {split_id}")
        if set(specification.val_mixture_ids) != partitions["val"]:
            raise ContractError(f"validation groups differ from pilot split for {split_id}")
        if set(specification.test_mixture_ids) != partitions["test"]:
            raise ContractError(f"test groups differ from pilot split for {split_id}")
        if partitions["train"] & partitions["val"] or partitions["train"] & partitions["test"] or partitions["val"] & partitions["test"]:
            raise ContractError(f"mixture group leakage detected for {split_id}")
        previous: tuple[str, ...] = ()
        for fraction in fractions:
            prefix = specification.train_prefixes[int(fraction)]
            if prefix != order[: len(prefix)] or prefix[: len(previous)] != previous:
                raise ContractError(f"training groups are not frozen nested prefixes for {split_id} at {fraction}%")
            previous = prefix
        if previous != order:
            raise ContractError(f"100% prefix does not contain all train groups for {split_id}")


def indices_for_groups(metadata: PilotMetadata, mixture_ids: Sequence[str]) -> np.ndarray:
    allowed = set(mixture_ids)
    return np.asarray([index for index, mixture_id in enumerate(metadata.mixture_ids) if mixture_id in allowed], dtype=int)


def _labels_with_cache(
    metadata: PilotMetadata,
    indices: np.ndarray,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    missing = np.asarray([int(index) for index in indices if int(index) not in cache], dtype=int)
    if missing.size:
        values = _labels(metadata, missing)
        for index, value in zip(missing, values):
            cache[int(index)] = np.asarray(value, dtype=np.float64)
    return np.asarray([cache[int(index)] for index in indices], dtype=np.float64)


def random_non_nested_groups(specification: NestedSplit, fraction: int, seed: int) -> tuple[str, ...]:
    """Leakage control only: deterministic draw with the same group count."""
    count = len(specification.train_prefixes[fraction])
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.asarray(specification.train_group_order), size=count, replace=False)
    return tuple(str(item) for item in chosen)


class _ELM:
    def __init__(self, hidden_units: int, alpha: float, seed: int, *, orthogonal: bool):
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.seed = seed
        self.orthogonal = orthogonal

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ELM":
        rng = np.random.default_rng(self.seed)
        weights = rng.normal(size=(x.shape[1], self.hidden_units))
        if self.orthogonal:
            if x.shape[1] >= self.hidden_units:
                weights = np.linalg.qr(weights)[0][:, : self.hidden_units]
            else:
                weights = np.linalg.qr(weights.T)[0].T
        self.weights_ = weights
        self.bias_ = rng.uniform(-1.0, 1.0, size=self.hidden_units)
        hidden = np.tanh(x @ self.weights_ + self.bias_)
        gram = hidden.T @ hidden + self.alpha * np.eye(hidden.shape[1])
        self.output_ = np.linalg.solve(gram, hidden.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x @ self.weights_ + self.bias_) @ self.output_


class _ScaledELM:
    def __init__(self, specification: Mapping[str, Any], seed: int, *, orthogonal: bool = False):
        self.specification = specification
        self.seed = seed
        self.orthogonal = orthogonal

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ScaledELM":
        from sklearn.preprocessing import StandardScaler

        self.scaler_ = StandardScaler().fit(x)
        self.elm_ = _ELM(
            int(self.specification["hidden_units"]),
            float(self.specification["alpha"]),
            self.seed,
            orthogonal=self.orthogonal,
        ).fit(self.scaler_.transform(x), y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.elm_.predict(self.scaler_.transform(x))


class _PLSELM:
    def __init__(
        self,
        specification: Mapping[str, Any],
        seed: int,
        *,
        target_weights: np.ndarray | None = None,
        orthogonal: bool = False,
    ):
        self.specification = specification
        self.seed = seed
        self.target_weights = target_weights
        self.orthogonal = orthogonal

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_PLSELM":
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.preprocessing import StandardScaler

        weights = np.ones(y.shape[1], dtype=np.float64) if self.target_weights is None else np.asarray(self.target_weights)
        if weights.shape != (y.shape[1],) or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ContractError("Fisher target weights must be finite positive component weights")
        self.target_weights_ = weights
        self.scaler_ = StandardScaler().fit(x)
        scaled_x = self.scaler_.transform(x)
        component_count = min(int(self.specification["pls_components"]), x.shape[1], y.shape[1], x.shape[0] - 1)
        if component_count < 1:
            raise ContractError("PLSELM requires at least two training samples")
        self.pls_ = PLSRegression(n_components=component_count, scale=False, max_iter=500).fit(scaled_x, y * weights)
        scores = self.pls_.transform(scaled_x)
        self.score_scaler_ = StandardScaler().fit(scores)
        self.elm_ = _ELM(
            int(self.specification["hidden_units"]),
            float(self.specification["alpha"]),
            self.seed,
            orthogonal=self.orthogonal,
        ).fit(self.score_scaler_.transform(scores), y * weights)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.pls_.transform(self.scaler_.transform(x))
        return self.elm_.predict(self.score_scaler_.transform(scores)) / self.target_weights_


class _LightCNN:
    def __init__(self, specification: Mapping[str, Any], seed: int):
        self.specification = specification
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_LightCNN":
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        torch.set_num_threads(int(self.specification["torch_threads"]))
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True)
        if np.any(self.std_ == 0.0):
            raise ContractError("CNN input contains a zero-variance feature")
        train = torch.as_tensor(((x - self.mean_) / self.std_)[:, None, :], dtype=torch.float32)
        targets = torch.as_tensor(y, dtype=torch.float32)
        channels = int(self.specification["channels"])
        self.model_ = nn.Sequential(
            nn.Conv1d(1, channels, int(self.specification["kernel_size"]), padding="same"),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, y.shape[1]),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=float(self.specification["learning_rate"]))
        generator = torch.Generator().manual_seed(self.seed)
        batch_size = int(self.specification["batch_size"])
        self.model_.train()
        for _ in range(int(self.specification["epochs"])):
            permutation = torch.randperm(len(train), generator=generator)
            for start in range(0, len(train), batch_size):
                batch = permutation[start : start + batch_size]
                loss = torch.mean((self.model_(train[batch]) - targets[batch]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        self.model_.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(((x - self.mean_) / self.std_)[:, None, :], dtype=torch.float32)
            return self.model_(tensor).cpu().numpy().astype(np.float64)


def _fisher_target_weights_from_matrices(matrices: Sequence[np.ndarray]) -> np.ndarray:
    physical_precisions = []
    simplex_transform = np.vstack([np.eye(3), -np.ones(3)])
    for fisher in matrices:
        physical_crb = simplex_transform @ np.linalg.inv(fisher) @ simplex_transform.T
        diagonal = np.diag(physical_crb)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
            raise ContractError("physical CRB diagonal must be finite and positive")
        physical_precisions.append(1.0 / np.sqrt(diagonal))
    weights = np.mean(np.asarray(physical_precisions, dtype=np.float64), axis=0)
    return weights / np.exp(np.mean(np.log(weights)))


def fisher_target_weights(
    metadata: PilotMetadata,
    train_indices: np.ndarray,
    *,
    matrix_cache: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    from ..sim.packaging.arrays import read_array_artifact

    matrices = []
    for index in train_indices:
        position = int(index)
        if matrix_cache is not None and position in matrix_cache:
            fisher = matrix_cache[position]
        else:
            record = metadata.records[position]
            fisher = read_array_artifact(metadata.root / record["arrays"]["effective_fisher"]["file_ref"])
            if matrix_cache is not None:
                matrix_cache[position] = fisher
        matrices.append(fisher)
    return _fisher_target_weights_from_matrices(matrices)


def make_model(
    method_id: str,
    config: Mapping[str, Any],
    seed: int,
    *,
    fisher_weights: np.ndarray,
    control: str = "main",
) -> Any:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import Ridge
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    specification = config["models"][method_id]
    if method_id == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(specification["alpha"])))
    if method_id == "pls":
        return PLSRegression(
            n_components=int(specification["n_components"]),
            scale=bool(specification["scale"]),
            max_iter=int(specification["max_iter"]),
        )
    if method_id == "elm":
        return _ScaledELM(specification, seed)
    if method_id == "plselm":
        return _PLSELM(specification, seed)
    if method_id == "xgboost_strong_table":
        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=int(specification["n_estimators"]),
                max_depth=int(specification["max_depth"]),
                learning_rate=float(specification["learning_rate"]),
                subsample=float(specification["subsample"]),
                colsample_bytree=float(specification["colsample_bytree"]),
                n_jobs=int(specification["n_jobs"]),
                tree_method=str(specification["tree_method"]),
                random_state=seed,
                objective="reg:squarederror",
            )
        )
    if method_id == "fo_mplselm":
        weights = np.ones_like(fisher_weights) if control == "without_fisher_weights" else fisher_weights
        direction_seed = seed + 10_000 if control == "random_orthogonal_directions" else seed
        return _PLSELM(specification, direction_seed, target_weights=weights, orthogonal=True)
    if method_id == "cnn_light":
        return _LightCNN(specification, seed)
    raise ContractError(f"unregistered method: {method_id}")


def paired_group_p90_ci(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    mixture_ids: Sequence[str],
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    groups = tuple(dict.fromkeys(str(item) for item in mixture_ids))
    if not groups:
        raise ContractError("paired bootstrap requires at least one mixture group")
    positions = {group: np.flatnonzero(np.asarray(mixture_ids) == group) for group in groups}

    def statistic(indices: np.ndarray) -> float:
        return float(np.quantile(np.abs(candidate[indices] - y_true[indices]), 0.9, method="higher") - np.quantile(np.abs(reference[indices] - y_true[indices]), 0.9, method="higher"))

    point = statistic(np.arange(len(y_true)))
    rng = np.random.default_rng(seed)
    group_positions = [positions[group] for group in groups]
    group_sizes = {len(indices) for indices in group_positions}
    if len(group_sizes) == 1:
        position_matrix = np.stack(group_positions)
        sampled_groups = rng.integers(0, len(groups), size=(resamples, len(groups)))
        indices = position_matrix[sampled_groups].reshape(resamples, -1)
        candidate_error = np.abs(candidate - y_true)
        reference_error = np.abs(reference - y_true)
        draws = np.quantile(candidate_error[indices], 0.9, axis=1, method="higher") - np.quantile(
            reference_error[indices],
            0.9,
            axis=1,
            method="higher",
        )
    else:
        maximum_size = max(group_sizes)
        padded_positions = np.full((len(groups), maximum_size), -1, dtype=np.int64)
        for group_index, indices in enumerate(group_positions):
            padded_positions[group_index, : len(indices)] = indices
        sampled_groups = rng.integers(0, len(groups), size=(resamples, len(groups)))
        candidate_error = np.abs(candidate - y_true)
        reference_error = np.abs(reference - y_true)
        draws = np.empty(resamples, dtype=np.float64)
        batch_size = 256
        for start in range(0, resamples, batch_size):
            stop = min(start + batch_size, resamples)
            selected = padded_positions[sampled_groups[start:stop]]
            valid = selected >= 0
            safe_indices = np.where(valid, selected, 0)
            candidate_batch = np.where(valid, candidate_error[safe_indices], np.nan).reshape(stop - start, -1)
            reference_batch = np.where(valid, reference_error[safe_indices], np.nan).reshape(stop - start, -1)
            draws[start:stop] = np.nanquantile(candidate_batch, 0.9, axis=1, method="higher") - np.nanquantile(
                reference_batch,
                0.9,
                axis=1,
                method="higher",
            )
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def paired_reduction_ci(candidate_ns: Sequence[int], reference_ns: Sequence[int], *, resamples: int, seed: int) -> tuple[float, float, float]:
    candidate = np.asarray(candidate_ns, dtype=np.float64)
    reference = np.asarray(reference_ns, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.size < 2 or np.any(reference <= 0.0):
        raise ContractError("paired timing requires aligned positive repeated observations")
    reductions = 1.0 - candidate / reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, reductions.size, size=(resamples, reductions.size))
    estimates = np.median(reductions[indices], axis=1)
    return float(np.median(reductions)), float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def evaluate_candidate_verdict(
    *,
    ni_rows: Sequence[Mapping[str, Any]],
    first_target_fraction: int | None,
    timing_rows: Sequence[Mapping[str, Any]],
    negative_controls: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    equivalent_to_plselm: bool,
) -> dict[str, Any]:
    required_controls = {"random_orthogonal_directions", "without_fisher_weights", "random_non_nested_subset"}
    if not ni_rows or not timing_rows or set(negative_controls) != required_controls:
        return {"candidate_verdict": "inconclusive", "reason": "required evidence is incomplete"}
    if any(row.get("status") != "complete" for row in ni_rows) or any(row.get("status") != "complete" for row in timing_rows):
        return {"candidate_verdict": "inconclusive", "reason": "failed precision or timing rows are retained"}
    ni_pass = all(float(row["ci_upper"]) <= float(config["non_inferiority_band"][row["component"]]) for row in ni_rows)
    minimum = float(config["efficiency"]["minimum_relative_reduction"])
    maximum_regression = float(config["efficiency"]["maximum_other_primary_regression"])
    timing_by_metric = {str(row["metric"]): row for row in timing_rows}
    timing_advantage = any(float(row["ci_lower"]) >= minimum for row in timing_rows)
    other_costs_pass = all(float(row["ci_lower"]) >= -maximum_regression for row in timing_rows)
    fraction_pass = first_target_fraction is not None and first_target_fraction <= int(config["efficiency"]["maximum_first_target_fraction"])
    controls_pass = all(bool(row.get("passed")) for row in negative_controls.values())
    passed = ni_pass and (fraction_pass or timing_advantage) and other_costs_pass and controls_pass and not equivalent_to_plselm
    if passed:
        verdict, reason = "enter_P4", "NI and the preregistered data or timing efficiency endpoint passed"
    else:
        verdict = "reject"
        if equivalent_to_plselm:
            reason = "FO-MPLSELM is equivalent to the registered PLSELM baseline"
        elif not ni_pass:
            reason = "precision non-inferiority failed"
        elif not controls_pass:
            reason = "one or more preregistered negative controls failed"
        else:
            reason = "no preregistered data or timing efficiency advantage"
    return {
        "candidate_verdict": verdict,
        "reason": reason,
        "precision_non_inferiority": ni_pass,
        "first_target_fraction_pass": fraction_pass,
        "timing_advantage": timing_advantage,
        "other_primary_costs_pass": other_costs_pass,
        "negative_controls_pass": controls_pass,
        "equivalent_to_plselm": equivalent_to_plselm,
        "timing_metrics": sorted(timing_by_metric),
    }


def _macro_validation_p90(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean([np.quantile(np.abs(prediction[:, index] - y_true[:, index]), 0.9, method="higher") for index in range(y_true.shape[1])]))


def _batch(x: np.ndarray, size: int) -> np.ndarray:
    if len(x) == 0:
        raise ContractError("inference timing requires non-empty test features")
    return x[np.arange(size) % len(x)]


def _actual_runtime() -> dict[str, str]:
    import sklearn
    import threadpoolctl
    import torch
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "torch": torch.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }


def _execution_metadata(
    config: Mapping[str, Any],
    execution_registry: Mapping[str, Any],
    *,
    execution_registry_sha256: str,
    git_commit: str,
) -> dict[str, Any]:
    if execution_registry.get("registry_id") != config.get("execution_registry_id"):
        raise RuntimeError("execution registry does not match the frozen data-efficiency plan")
    actual_runtime = _actual_runtime()
    if actual_runtime != config.get("required_runtime"):
        raise RuntimeError(f"runtime dependency lock mismatch: {actual_runtime}")
    hardware = execution_registry["hardware"]
    limit = int(config["execution"]["paired_thread_limit"])
    metadata = {
        "hardware_fingerprint": str(hardware["hardware_profile_id"]),
        "device_track": str(config["execution"]["device_track"]),
        "logical_cpu_count": int(hardware["cpu_logical_processors"]),
        "blas_threads": limit,
        "omp_threads": limit,
        "mkl_threads": limit,
        "framework_threads": limit,
        "os_version": str(hardware["os"]),
        "python_version": actual_runtime["python"],
        "numpy_version": actual_runtime["numpy"],
        "framework_version": actual_runtime["torch"],
        "method_package_versions": ";".join(
            f"{name}=={version}" for name, version in sorted(actual_runtime.items())
        ),
        "git_commit": str(git_commit),
        "execution_registry_sha256": str(execution_registry_sha256),
    }
    metadata["execution_fingerprint_sha256"] = sha256_bytes(canonical_json_bytes(metadata))
    return metadata


def _timed_fit(
    method_id: str,
    config: Mapping[str, Any],
    seed: int,
    train_x: np.ndarray,
    y_train: np.ndarray,
    val_x: np.ndarray,
    *,
    fisher_provider: Any,
    control: str = "main",
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    preprocess_start = time.perf_counter_ns()
    fisher = fisher_provider() if method_id == "fo_mplselm" else np.ones(y_train.shape[1], dtype=np.float64)
    model = make_model(method_id, config, seed, fisher_weights=fisher, control=control)
    preprocess_duration = time.perf_counter_ns() - preprocess_start
    fit_start = time.perf_counter_ns()
    model.fit(train_x, y_train)
    fit_duration = time.perf_counter_ns() - fit_start
    validation_start = time.perf_counter_ns()
    validation_prediction = np.asarray(model.predict(val_x), dtype=np.float64)
    validation_duration = time.perf_counter_ns() - validation_start
    return model, validation_prediction, {
        "preprocess_duration_ns": int(preprocess_duration),
        "fit_duration_ns": int(fit_duration),
        "validation_duration_ns": int(validation_duration),
        "training_total_duration_ns": int(preprocess_duration + fit_duration + validation_duration),
        "status": "complete",
    }


def _inference_timings(
    models: Mapping[str, Any],
    x: np.ndarray,
    config: Mapping[str, Any],
    order_seed: int,
    execution_fingerprint_sha256: str,
) -> list[dict[str, Any]]:
    timing = config["timing"]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(order_seed)
    batches = {int(batch_size): _batch(x, int(batch_size)) for batch_size in timing["batch_sizes"]}
    for repeat in range(int(timing["independent_repeats"])):
        for batch_size in timing["batch_sizes"]:
            batch = batches[int(batch_size)]
            order = list(models)
            rng.shuffle(order)
            for method_id in order:
                predictor = models[method_id].predict
                for _ in range(int(timing["warmup_batches"])):
                    predictor(batch)
                start = time.perf_counter_ns()
                for _ in range(int(timing["timed_batches_per_repeat"])):
                    predictor(batch)
                duration = time.perf_counter_ns() - start
                rows.append({
                    "method_id": method_id,
                    "repeat_index": repeat,
                    "batch_size": int(batch_size),
                    "duration_ns": int(duration),
                    "execution_fingerprint_sha256": execution_fingerprint_sha256,
                    "status": "complete",
                })
    return rows


def _unit_id(split_id: str, seed: int, fraction: int) -> str:
    return f"{split_id}__seed-{seed}__fraction-{fraction:03d}"


def _checkpoint_wrapper(unit_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "gib-benchmark-1",
        "unit_id": unit_id,
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
        "payload": payload,
    }


def _read_checkpoint(path: Path, expected_unit_id: str) -> dict[str, Any]:
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    if wrapper.get("unit_id") != expected_unit_id:
        raise ContractError(f"checkpoint unit identity mismatch: {path}")
    payload = wrapper.get("payload")
    if not isinstance(payload, dict):
        raise ContractError(f"checkpoint payload is invalid: {path}")
    if wrapper.get("payload_sha256") != sha256_bytes(canonical_json_bytes(payload)):
        raise ContractError(f"checkpoint payload hash mismatch: {path}")
    return payload


def _run_data_efficiency_unit(
    *,
    metadata: PilotMetadata,
    split_id: str,
    split: NestedSplit,
    seed: int,
    fraction: int,
    features: np.ndarray,
    config: Mapping[str, Any],
    execution_fingerprint_sha256: str,
    fisher_matrix_cache: dict[int, np.ndarray],
    label_cache: dict[int, np.ndarray],
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    components = list(config["components"])
    train_indices = indices_for_groups(metadata, split.train_prefixes[fraction])
    val_indices = indices_for_groups(metadata, split.val_mixture_ids)
    test_indices = indices_for_groups(metadata, split.test_mixture_ids)
    train_x = np.asarray(features[train_indices], dtype=np.float64)
    val_x = np.asarray(features[val_indices], dtype=np.float64)
    test_x = np.asarray(features[test_indices], dtype=np.float64)
    y_train = _labels_with_cache(metadata, train_indices, label_cache)
    y_val = _labels_with_cache(metadata, val_indices, label_cache)

    def fisher_provider(indices: np.ndarray = train_indices) -> np.ndarray:
        return fisher_target_weights(metadata, indices, matrix_cache=fisher_matrix_cache)

    fitted: dict[str, Any] = {}
    validation_scores: dict[str, float] = {}
    fit_rows: list[dict[str, Any]] = []
    order_rng = np.random.default_rng(seed + fraction)
    method_order = list(METHOD_IDS)
    order_rng.shuffle(method_order)
    thread_limit = int(config["execution"]["paired_thread_limit"])
    with threadpool_limits(limits=thread_limit):
        for method_id in method_order:
            model, val_prediction, timing_row = _timed_fit(
                method_id,
                config,
                seed,
                train_x,
                y_train,
                val_x,
                fisher_provider=fisher_provider,
            )
            validation_scores[method_id] = _macro_validation_p90(y_val, val_prediction)
            fitted[method_id] = model
            fit_rows.append({
                "method_id": method_id,
                "repeat_index": 0,
                "execution_fingerprint_sha256": execution_fingerprint_sha256,
                **timing_row,
            })
        reference_id = min(config["selection"]["reference_candidates"], key=validation_scores.__getitem__)

        if fraction == int(config["timing"]["formal_training_fraction"]):
            paired_methods = ["fo_mplselm", reference_id]
            for repeat_index in range(1, int(config["timing"]["independent_repeats"])):
                repeated_order = list(paired_methods)
                order_rng.shuffle(repeated_order)
                for method_id in repeated_order:
                    _, _, timing_row = _timed_fit(
                        method_id,
                        config,
                        seed,
                        train_x,
                        y_train,
                        val_x,
                        fisher_provider=fisher_provider,
                    )
                    fit_rows.append({
                        "method_id": method_id,
                        "repeat_index": repeat_index,
                        "execution_fingerprint_sha256": execution_fingerprint_sha256,
                        **timing_row,
                    })

        # Test labels are opened only after all scientific models are fit and the reference is selected.
        y_test = _labels_with_cache(metadata, test_indices, label_cache)
        predictions = {
            method_id: np.asarray(model.predict(test_x), dtype=np.float64)
            for method_id, model in fitted.items()
        }
        curve_rows: list[dict[str, Any]] = []
        prediction_records: list[dict[str, Any]] = []
        for method_id, prediction in predictions.items():
            for cell_id in sorted(set(metadata.cell_ids[test_indices])):
                local = np.flatnonzero(metadata.cell_ids[test_indices] == cell_id)
                for metric in _metrics(y_test[local], prediction[local], components):
                    curve_rows.append({
                        "grid_cell_id": str(cell_id),
                        "split_id": split_id,
                        "seed": seed,
                        "training_fraction": fraction,
                        "method_id": method_id,
                        "training_mixture_group_count": len(split.train_prefixes[fraction]),
                        "training_sample_count": len(train_indices),
                        **metric,
                    })
            for local_index, global_index in enumerate(test_indices):
                prediction_records.append({
                    "sequence_id": metadata.records[int(global_index)]["sequence_id"],
                    "mixture_id": metadata.records[int(global_index)]["mixture_id"],
                    "grid_cell_id": metadata.cell_ids[int(global_index)],
                    "split_id": split_id,
                    "seed": seed,
                    "training_fraction": fraction,
                    "method_id": method_id,
                    "reference_method_id": reference_id,
                    "truth": y_test[local_index].tolist(),
                    "prediction": prediction[local_index].tolist(),
                })

        timing_records = [
            {"split_id": split_id, "seed": seed, "training_fraction": fraction, "phase": "training", **row}
            for row in fit_rows
        ]
        if fraction == int(config["timing"]["formal_training_fraction"]):
            timed_models = {method_id: fitted[method_id] for method_id in ("fo_mplselm", reference_id)}
            for row in _inference_timings(
                timed_models,
                test_x,
                config,
                seed + fraction * 100,
                execution_fingerprint_sha256,
            ):
                timing_records.append({
                    "split_id": split_id,
                    "seed": seed,
                    "training_fraction": fraction,
                    "phase": "inference",
                    **row,
                })

        main_macro_p90 = _macro_validation_p90(y_test, predictions["fo_mplselm"])
        fisher = fisher_provider()
        random_indices = indices_for_groups(
            metadata,
            random_non_nested_groups(split, fraction, seed + fraction),
        )
        control_specs = {
            "random_orthogonal_directions": (train_indices, fisher, "random_orthogonal_directions"),
            "without_fisher_weights": (train_indices, fisher, "without_fisher_weights"),
            "random_non_nested_subset": (
                random_indices,
                fisher_target_weights(metadata, random_indices, matrix_cache=fisher_matrix_cache),
                "main",
            ),
        }
        control_records: list[dict[str, Any]] = []
        for control_id, (source_indices, control_fisher, control_mode) in control_specs.items():
            model = make_model(
                "fo_mplselm",
                config,
                seed,
                fisher_weights=control_fisher,
                control=control_mode,
            )
            model.fit(features[source_indices], _labels_with_cache(metadata, source_indices, label_cache))
            control_prediction = np.asarray(model.predict(test_x), dtype=np.float64)
            control_records.append({
                "control_id": control_id,
                "split_id": split_id,
                "seed": seed,
                "training_fraction": fraction,
                "candidate_macro_test_p90": main_macro_p90,
                "control_macro_test_p90": _macro_validation_p90(y_test, control_prediction),
                "mixture_ids": [str(metadata.records[int(index)]["mixture_id"]) for index in test_indices],
                "truth": y_test.tolist(),
                "candidate_prediction": predictions["fo_mplselm"].tolist(),
                "control_prediction": control_prediction.tolist(),
                "main_conclusion_allowed": control_id != "random_non_nested_subset",
                "status": "complete",
            })

    return {
        "unit_id": _unit_id(split_id, seed, fraction),
        "curve_rows": curve_rows,
        "prediction_records": prediction_records,
        "timing_records": timing_records,
        "control_records": control_records,
        "elapsed_ns": int(time.perf_counter_ns() - started),
    }


def run_data_efficiency(
    config: dict[str, Any],
    *,
    config_sha256: str,
    pilot_freeze: Path,
    execution_registry: Mapping[str, Any],
    execution_registry_sha256: str,
    git_commit: str,
    output_dir: Path,
    resume: bool = False,
) -> dict[str, Any]:
    """Run or resume the formal P3-08 append-only attempt."""
    validate_data_efficiency_plan(config)
    for name, expected in config["required_environment"].items():
        if os.environ.get(name) != expected:
            raise RuntimeError(f"required environment is not locked: {name}={expected}")
    metadata = load_pilot_metadata(pilot_freeze)
    if metadata.generation_summary.get("plan_id") != config["pilot_plan_id"]:
        raise ContractError("pilot plan differs from the frozen data-efficiency plan")
    execution_metadata = _execution_metadata(
        config,
        execution_registry,
        execution_registry_sha256=execution_registry_sha256,
        git_commit=git_commit,
    )
    pilot_manifest_sha256 = sha256_file(Path(pilot_freeze) / "evidence_manifest.json")
    identity = {
        "schema_version": "gib-benchmark-1",
        "task_id": "P3-08",
        "config_sha256": config_sha256,
        "runner_code_sha256": sha256_file(Path(__file__)),
        "pilot_freeze_id": Path(pilot_freeze).name,
        "pilot_evidence_manifest_sha256": pilot_manifest_sha256,
        "execution_metadata": execution_metadata,
    }
    identity["identity_sha256"] = sha256_bytes(canonical_json_bytes(identity))

    target = Path(output_dir)
    identity_path = target / "checkpoint_identity.json"
    manifest_path = target / "attempt_manifest.json"
    result_path = target / "data_efficiency_results.json"
    if resume:
        if not target.is_dir() or not identity_path.is_file():
            raise FileNotFoundError(f"resumable attempt does not exist: {target}")
        if manifest_path.exists():
            raise FileExistsError(f"attempt is already complete and cannot be resumed: {target}")
        recorded_identity = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded_identity != identity:
            raise ContractError("resume identity differs from the incomplete attempt")
    else:
        if target.exists():
            raise FileExistsError(f"attempt directory already exists: {target}")
        target.mkdir(parents=True)
        atomic_write_json(identity_path, identity)

    nested = load_frozen_nested_groups(metadata, config["training_group_fractions"])
    features, _ = deployment_features(metadata)
    units = [
        (str(split_id), int(seed), int(fraction))
        for split_id in config["split_ids"]
        for seed in config["seeds"]
        for fraction in config["training_group_fractions"]
    ]
    payloads: list[dict[str, Any]] = []
    fisher_matrix_cache: dict[int, np.ndarray] = {}
    label_caches = {str(split_id): {} for split_id in config["split_ids"]}
    units_root = target / "checkpoint_units"
    for ordinal, (split_id, seed, fraction) in enumerate(units, start=1):
        unit_id = _unit_id(split_id, seed, fraction)
        checkpoint_path = units_root / f"{unit_id}.json"
        if checkpoint_path.exists():
            payload = _read_checkpoint(checkpoint_path, unit_id)
        else:
            payload = _run_data_efficiency_unit(
                metadata=metadata,
                split_id=split_id,
                split=nested[split_id],
                seed=seed,
                fraction=fraction,
                features=features,
                config=config,
                execution_fingerprint_sha256=execution_metadata["execution_fingerprint_sha256"],
                fisher_matrix_cache=fisher_matrix_cache,
                label_cache=label_caches[split_id],
            )
            atomic_write_json(checkpoint_path, _checkpoint_wrapper(unit_id, payload))
        payloads.append(payload)
        elapsed_ns = sum(int(item["elapsed_ns"]) for item in payloads)
        eta_ns = int(elapsed_ns / ordinal * (len(units) - ordinal)) if ordinal else 0
        print(json.dumps({
            "task_id": "P3-08",
            "completed_units": ordinal,
            "total_units": len(units),
            "unit_id": unit_id,
            "elapsed_seconds": round(elapsed_ns / 1e9, 3),
            "eta_seconds": round(eta_ns / 1e9, 3),
        }, sort_keys=True))

    curve_rows = [row for payload in payloads for row in payload["curve_rows"]]
    prediction_records = [row for payload in payloads for row in payload["prediction_records"]]
    timing_records = [row for payload in payloads for row in payload["timing_records"]]
    control_records = [row for payload in payloads for row in payload["control_records"]]
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("config_sha256") != config_sha256:
            raise ContractError("existing final result does not match the resume identity")
    else:
        result = _summarize_result(
            config,
            metadata,
            config_sha256,
            Path(pilot_freeze).name,
            curve_rows,
            prediction_records,
            timing_records,
            control_records,
        )
        result["execution_metadata"] = execution_metadata
        result["execution_counts"] = {
            "checkpoint_units": len(payloads),
            "scientific_model_fits": len(units) * len(METHOD_IDS),
            "formal_timing_additional_fits": len(config["split_ids"])
            * len(config["seeds"])
            * (int(config["timing"]["independent_repeats"]) - 1)
            * 2,
            "negative_control_fits": len(units) * len(config["negative_controls"]),
            "formal_predict_calls": len(config["split_ids"])
            * len(config["seeds"])
            * int(config["timing"]["independent_repeats"])
            * len(config["timing"]["batch_sizes"])
            * 2
            * (int(config["timing"]["warmup_batches"]) + int(config["timing"]["timed_batches_per_repeat"])),
        }
        atomic_write_json(result_path, result)
    atomic_write_json(manifest_path, {
        "schema_version": "gib-benchmark-1",
        "attempt_id": target.name,
        "task_id": "P3-08",
        "status": "complete",
        "task_status": "completed",
        "candidate_verdict": result["candidate_verdict"],
        "claim_scope": config["claim_scope"],
        "next_allowed_task": "P3-13",
        "checkpoint_identity_sha256": identity["identity_sha256"],
    })
    return result


def _summarize_result(
    config: Mapping[str, Any], metadata: PilotMetadata, config_sha256: str, pilot_freeze_id: str,
    curve_rows: list[dict[str, Any]], prediction_records: list[dict[str, Any]],
    timing_records: list[dict[str, Any]], control_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ni_rows: list[dict[str, Any]] = []
    equivalent = True
    for split_id in config["split_ids"]:
        for seed in config["seeds"]:
            for fraction in config["training_group_fractions"]:
                scope = [row for row in prediction_records if row["split_id"] == split_id and row["seed"] == seed and row["training_fraction"] == fraction]
                reference_id = next(row["reference_method_id"] for row in scope)
                for cell_id in sorted(set(row["grid_cell_id"] for row in scope)):
                    candidate_rows = [row for row in scope if row["grid_cell_id"] == cell_id and row["method_id"] == "fo_mplselm"]
                    reference_rows = [row for row in scope if row["grid_cell_id"] == cell_id and row["method_id"] == reference_id]
                    plselm_rows = [row for row in scope if row["grid_cell_id"] == cell_id and row["method_id"] == "plselm"]
                    candidate = np.asarray([row["prediction"] for row in candidate_rows])
                    reference = np.asarray([row["prediction"] for row in reference_rows])
                    truth = np.asarray([row["truth"] for row in candidate_rows])
                    mixtures = [row["mixture_id"] for row in candidate_rows]
                    equivalence = config["plselm_equivalence"]
                    equivalent &= np.allclose(
                        candidate,
                        np.asarray([row["prediction"] for row in plselm_rows]),
                        rtol=float(equivalence["rtol"]),
                        atol=float(equivalence["atol"]),
                    )
                    for component_index, component in enumerate(config["components"]):
                        point, lower, upper = paired_group_p90_ci(
                            truth[:, component_index], candidate[:, component_index], reference[:, component_index], mixtures,
                            resamples=int(config["statistics"]["bootstrap_resamples"]),
                            seed=int(config["statistics"]["bootstrap_seed"]) + int(seed) + int(fraction) + component_index,
                        )
                        ni_rows.append({"grid_cell_id": cell_id, "split_id": split_id, "seed": int(seed), "training_fraction": int(fraction), "component": component, "reference_method_id": reference_id, "point": point, "ci_lower": lower, "ci_upper": upper, "status": "complete"})

    target = config["precision_target"]
    first_fractions = []
    for split_id in config["split_ids"]:
        for seed in config["seeds"]:
            reached = None
            for fraction in config["training_group_fractions"]:
                rows = [row for row in curve_rows if row["split_id"] == split_id and row["seed"] == seed and row["training_fraction"] == fraction and row["method_id"] == "fo_mplselm"]
                if rows and all(row["p90"] <= target[row["component"]] for row in rows):
                    reached = int(fraction)
                    break
            first_fractions.append(reached)
    first_target_fraction = max(item for item in first_fractions if item is not None) if all(item is not None for item in first_fractions) else None

    # Timing summaries use the exact paired repeat rows for FO and the validation-selected reference.
    timing_summary: list[dict[str, Any]] = []
    for metric, phase, batch_size in (("training_wall_clock", None, None), ("batch_size_1_latency", "inference", 1)):
        candidate_values: list[int] = []
        reference_values: list[int] = []
        timing_error: str | None = None
        expected_repeats = set(range(int(config["timing"]["independent_repeats"])))
        for split_id in config["split_ids"]:
            for seed in config["seeds"]:
                fraction = int(config["timing"]["formal_training_fraction"])
                scope_predictions = [row for row in prediction_records if row["split_id"] == split_id and row["seed"] == seed and row["training_fraction"] == fraction]
                reference_id = next(row["reference_method_id"] for row in scope_predictions)
                rows = [row for row in timing_records if row["split_id"] == split_id and row["seed"] == seed and row["training_fraction"] == fraction]
                if phase is None:
                    candidate_rows = [row for row in rows if row["method_id"] == "fo_mplselm" and "training_total_duration_ns" in row]
                    reference_rows = [row for row in rows if row["method_id"] == reference_id and "training_total_duration_ns" in row]
                    value_field = "training_total_duration_ns"
                else:
                    candidate_rows = [row for row in rows if row.get("phase") == phase and row["method_id"] == "fo_mplselm" and row["batch_size"] == batch_size]
                    reference_rows = [row for row in rows if row.get("phase") == phase and row["method_id"] == reference_id and row["batch_size"] == batch_size]
                    value_field = "duration_ns"
                candidate_by_repeat = {int(row["repeat_index"]): row for row in candidate_rows}
                reference_by_repeat = {int(row["repeat_index"]): row for row in reference_rows}
                fingerprints = {
                    str(row.get("execution_fingerprint_sha256"))
                    for row in [*candidate_rows, *reference_rows]
                }
                if (
                    len(candidate_rows) != len(expected_repeats)
                    or len(reference_rows) != len(expected_repeats)
                    or set(candidate_by_repeat) != expected_repeats
                    or set(reference_by_repeat) != expected_repeats
                ):
                    timing_error = f"paired timing repeats are incomplete for {split_id}/{seed}"
                    continue
                if len(fingerprints) != 1 or "None" in fingerprints:
                    timing_error = f"paired timing fingerprint mismatch for {split_id}/{seed}"
                    continue
                for repeat_index in sorted(expected_repeats):
                    candidate_values.append(int(candidate_by_repeat[repeat_index][value_field]))
                    reference_values.append(int(reference_by_repeat[repeat_index][value_field]))
        if timing_error is not None or len(candidate_values) < 2 or len(candidate_values) != len(reference_values):
            timing_summary.append({"metric": metric, "status": "failed", "error": timing_error or "paired timing observations are incomplete"})
        else:
            point, lower, upper = paired_reduction_ci(candidate_values, reference_values, resamples=int(config["statistics"]["bootstrap_resamples"]), seed=int(config["statistics"]["bootstrap_seed"]))
            timing_summary.append({"metric": metric, "point": point, "ci_lower": lower, "ci_upper": upper, "status": "complete"})

    controls: dict[str, dict[str, Any]] = {}
    expected_control_rows = len(config["split_ids"]) * len(config["seeds"]) * len(config["training_group_fractions"])
    required_mechanism_controls = set(config["mechanism_evidence"]["required_controls"])
    for control_index, control_id in enumerate(config["negative_controls"]):
        rows = [row for row in control_records if row["control_id"] == control_id]
        complete = len(rows) == expected_control_rows and all(row.get("status") == "complete" for row in rows)
        if control_id in required_mechanism_controls and complete:
            truth = np.concatenate([np.asarray(row["truth"], dtype=np.float64) for row in rows])
            candidate = np.concatenate([np.asarray(row["candidate_prediction"], dtype=np.float64) for row in rows])
            control = np.concatenate([np.asarray(row["control_prediction"], dtype=np.float64) for row in rows])
            mixture_ids = [mixture_id for row in rows for mixture_id in row["mixture_ids"]]
            component_evidence = []
            for component_index, component in enumerate(config["components"]):
                candidate_minus_control = paired_group_p90_ci(
                    truth[:, component_index],
                    candidate[:, component_index],
                    control[:, component_index],
                    mixture_ids,
                    resamples=int(config["statistics"]["bootstrap_resamples"]),
                    seed=int(config["statistics"]["bootstrap_seed"]) + control_index * 10 + component_index,
                )
                component_evidence.append({
                    "component": component,
                    "point": -candidate_minus_control[0],
                    "ci_lower": -candidate_minus_control[2],
                    "ci_upper": -candidate_minus_control[1],
                    "resampling_unit": "mixture_id",
                })
            controls[control_id] = {
                "passed": all(
                    row["ci_lower"] > float(config["mechanism_evidence"]["minimum_ci_lower"])
                    for row in component_evidence
                ),
                "component_evidence": component_evidence,
                "difference_direction": config["mechanism_evidence"]["difference_direction"],
                "main_conclusion_allowed": True,
            }
        else:
            controls[control_id] = {
                "passed": complete,
                "main_conclusion_allowed": False,
            }
    verdict = evaluate_candidate_verdict(
        ni_rows=ni_rows, first_target_fraction=first_target_fraction, timing_rows=timing_summary,
        negative_controls=controls, config=config, equivalent_to_plselm=equivalent,
    )
    return {
        "schema_version": "gib-benchmark-1", "task_id": "P3-08", "task_status": "completed",
        **verdict, "config_sha256": config_sha256, "pilot_freeze_id": pilot_freeze_id,
        "dataset_manifest_id": metadata.generation_summary["dataset_manifest_id"],
        "first_target_fraction": first_target_fraction, "first_target_fractions": first_fractions,
        "ni_rows": ni_rows, "timing_summary": timing_summary, "learning_curve_rows": curve_rows,
        "prediction_records": prediction_records, "timing_records": timing_records,
        "negative_controls": controls, "negative_control_rows": control_records,
        "claim_scope": config["claim_scope"], "next_allowed_task": "P3-13",
    }


__all__ = [
    "METHOD_IDS", "NestedSplit", "evaluate_candidate_verdict", "fisher_target_weights",
    "indices_for_groups", "load_frozen_nested_groups", "make_model", "paired_group_p90_ci",
    "paired_reduction_ci", "random_non_nested_groups", "run_data_efficiency",
    "validate_data_efficiency_plan", "validate_nested_group_contract",
]

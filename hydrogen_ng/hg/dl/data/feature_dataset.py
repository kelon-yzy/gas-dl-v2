from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from hg.common.windows import WINDOW_KIND_EARLY, WINDOW_KIND_PHASE, resolve_window_config
from hg.ml.features import MLFeatureConfig, load_feature_matrix


class V4FeatureMatrixDataset(Dataset):
    """DL-compatible dataset backed by the formal ML feature matrix."""

    def __init__(
        self,
        dataset_dir: Path | str,
        split: str,
        modalities: tuple[str, ...],
        scaler_path: Path | str | None = None,
        window: dict[str, object] | None = None,
        phase_windows: tuple[dict[str, object] | None, ...] | list[dict[str, object] | None] | None = None,
    ):
        if window is not None and phase_windows is not None:
            raise ValueError("window and phase_windows cannot be combined")
        resolved_window = resolve_window_config(window)
        feature_windows = _resolve_feature_windows(phase_windows)
        feature_config = MLFeatureConfig(
            modalities=modalities,
            slow_scaler_path=scaler_path,
            phase_filter=_phase_filter(resolved_window) if feature_windows is None else None,
            early_fraction=_early_fraction(resolved_window) if feature_windows is None else None,
            feature_windows=feature_windows,
        )
        matrix = load_feature_matrix(dataset_dir, split=split, config=feature_config)
        self.x = torch.from_numpy(matrix.x.copy())
        self.y = torch.from_numpy(matrix.y.copy())
        self.feature_names = matrix.feature_names
        self.label_names = matrix.label_names
        self.sequence_ids = matrix.sequence_ids

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _resolve_feature_windows(
    windows: tuple[dict[str, object] | None, ...] | list[dict[str, object] | None] | None,
):
    if windows is None:
        return None
    if not windows:
        raise ValueError("phase_windows must not be empty")
    return tuple(resolve_window_config(window) for window in windows)


def _phase_filter(window: object | None) -> str | None:
    if window is not None and window.kind == WINDOW_KIND_PHASE:
        return str(window.value)
    return None


def _early_fraction(window: object | None) -> float | None:
    if window is not None and window.kind == WINDOW_KIND_EARLY:
        return float(window.value)
    return None

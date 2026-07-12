from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class ScaledRidgeCVRegressor:
    """Train-only StandardScaler followed by multi-output RidgeCV."""

    def __init__(self, *, alphas: tuple[float, ...]):
        self.scaler = StandardScaler()
        self.model = RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: tuple[str, ...] | None = None,
    ) -> ScaledRidgeCVRegressor:
        del feature_names
        x_scaled = self.scaler.fit_transform(np.asarray(x, dtype=np.float64))
        self.model.fit(x_scaled, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(np.asarray(x, dtype=np.float64))
        return self.model.predict(x_scaled).astype(np.float32, copy=False)

    @property
    def selected_alpha(self) -> float:
        return float(self.model.alpha_)

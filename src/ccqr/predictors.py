"""Calibrated predictors used by the current CCQR experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .calibration import conformal_quantile


def _as_2d(predictions: np.ndarray) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=float)
    return predictions[:, None] if predictions.ndim == 1 else predictions


class StandardCQR:
    """Ordinary split CQR with fixed ``alpha/2`` endpoints."""

    def __init__(self, model: object, alpha: float = 0.1):
        self.model = model
        self.alpha = alpha
        self.quantiles = (alpha / 2, 1 - alpha / 2)

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> "StandardCQR":
        pred = _as_2d(self.model.predict(X, quantiles=list(self.quantiles)))
        y = np.asarray(y, dtype=float).reshape(-1)
        self.scores_ = np.maximum(pred[:, 0] - y, y - pred[:, 1])
        self.qhat_ = conformal_quantile(self.scores_, self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "qhat_"):
            raise RuntimeError("Call calibrate before predict.")
        pred = _as_2d(self.model.predict(X, quantiles=list(self.quantiles)))
        return np.column_stack([pred[:, 0] - self.qhat_, pred[:, 1] + self.qhat_])


@dataclass
class EndpointAveragedCQR:
    """Average quantile-pair endpoints and then apply ordinary split CQR."""

    model: object
    quantiles: Sequence[float] = (0.05, 0.95)
    bandwidth: float = 0.05
    n_quantiles: int = 9
    alpha: float | None = None

    def __post_init__(self) -> None:
        q_low, q_high = map(float, self.quantiles)
        if not 0 < q_low < q_high < 1:
            raise ValueError("quantiles must satisfy 0 < q_low < q_high < 1.")
        if self.n_quantiles < 1:
            raise ValueError("n_quantiles must be positive.")
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be nonnegative.")
        if q_low + self.bandwidth > q_high - self.bandwidth:
            raise ValueError("The lower and upper quantile grids cross.")
        if self.alpha is None:
            self.alpha = 1 - (q_high - q_low)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between 0 and 1.")
        grid_size = 1 if self.bandwidth == 0 else self.n_quantiles
        self.lower_grid_ = np.linspace(q_low, q_low + self.bandwidth, grid_size)
        self.upper_grid_ = np.linspace(q_high, q_high - self.bandwidth, grid_size)
        self.calibration_scores_: np.ndarray | None = None
        self.qhat_: float | None = None

    @property
    def prediction_quantiles(self) -> np.ndarray:
        return np.concatenate([self.lower_grid_, self.upper_grid_])

    def _average_endpoints(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.asarray(
            self.model.predict(X, quantiles=self.prediction_quantiles.tolist()),
            dtype=float,
        )
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        grid_size = self.lower_grid_.size
        expected = 2 * grid_size
        if predictions.shape[1] != expected:
            raise ValueError(
                f"The base model returned {predictions.shape[1]} columns; expected {expected}."
            )
        return predictions[:, :grid_size].mean(axis=1), predictions[:, grid_size:].mean(axis=1)

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray) -> "EndpointAveragedCQR":
        y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
        lower, upper = self._average_endpoints(X_calib)
        self.calibration_scores_ = np.maximum(lower - y_calib, y_calib - upper)
        self.qhat_ = conformal_quantile(self.calibration_scores_, float(self.alpha))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate before predict.")
        lower, upper = self._average_endpoints(X)
        return np.column_stack([lower - self.qhat_, upper + self.qhat_])


@dataclass
class ConformalLengthWeightedCQR:
    """Use fixed shared quantile-pair weights and apply ordinary split CQR."""

    model: object
    lower_grid: Sequence[float]
    upper_grid: Sequence[float]
    weights: Sequence[float]
    alpha: float = 0.1

    def __post_init__(self) -> None:
        self.lower_grid_ = np.asarray(self.lower_grid, dtype=float)
        self.upper_grid_ = np.asarray(self.upper_grid, dtype=float)
        self.weights_ = np.asarray(self.weights, dtype=float)
        if self.lower_grid_.shape != self.upper_grid_.shape:
            raise ValueError("lower_grid and upper_grid must define equal numbers of pairs.")
        if self.lower_grid_.shape != self.weights_.shape:
            raise ValueError("quantile grids and weights must have equal length.")
        if np.any(self.weights_ < 0) or not np.isclose(self.weights_.sum(), 1.0):
            raise ValueError("Pair weights must lie on the probability simplex.")
        self.qhat_: float | None = None
        self.calibration_scores_: np.ndarray | None = None

    @property
    def prediction_quantiles(self) -> np.ndarray:
        return np.concatenate([self.lower_grid_, self.upper_grid_])

    def _endpoints(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        predictions = np.asarray(
            self.model.predict(X, quantiles=self.prediction_quantiles.tolist()),
            dtype=float,
        )
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        split = self.lower_grid_.size
        expected = split + self.upper_grid_.size
        if predictions.shape[1] != expected:
            raise ValueError(
                f"The base model returned {predictions.shape[1]} columns; expected {expected}."
            )
        return predictions[:, :split] @ self.weights_, predictions[:, split:] @ self.weights_

    def calibrate(
        self, X_calib: np.ndarray, y_calib: np.ndarray
    ) -> "ConformalLengthWeightedCQR":
        y_calib = np.asarray(y_calib, dtype=float).reshape(-1)
        lower, upper = self._endpoints(X_calib)
        self.calibration_scores_ = np.maximum(lower - y_calib, y_calib - upper)
        self.qhat_ = conformal_quantile(self.calibration_scores_, self.alpha)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.qhat_ is None:
            raise RuntimeError("Call calibrate before predict.")
        lower, upper = self._endpoints(X)
        return np.column_stack([lower - self.qhat_, upper + self.qhat_])

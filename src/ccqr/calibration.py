"""Finite-sample split-conformal calibration utilities."""

from __future__ import annotations

import numpy as np


def conformal_quantile(
    scores: np.ndarray,
    alpha: float,
) -> float:
    """Return the finite-sample split-conformal calibration quantile."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("At least one calibration score is required.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    rank = min(int(np.ceil((scores.size + 1) * (1 - alpha))), scores.size)
    return float(np.partition(scores, rank - 1)[rank - 1])

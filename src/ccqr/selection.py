"""Leakage-free OOF selection for CCQR(d), CCQR(w), and CCQR(d,w)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .calibration import conformal_quantile


SelectionMode = Literal["d", "w", "both"]


def conformalized_mean_length(
    lower_predictions: np.ndarray,
    upper_predictions: np.ndarray,
    y: np.ndarray,
    weights: Sequence[float],
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Return OOF conformalized mean length and its CQR correction."""

    lower_predictions = np.asarray(lower_predictions, dtype=float)
    upper_predictions = np.asarray(upper_predictions, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if lower_predictions.ndim != 2 or lower_predictions.shape != upper_predictions.shape:
        raise ValueError("lower_predictions and upper_predictions must have equal 2D shapes.")
    if lower_predictions.shape[0] != y.size:
        raise ValueError("predictions and y have incompatible shapes.")
    if lower_predictions.shape[1] != weights.size:
        raise ValueError("The number of weights must match the number of quantile pairs.")
    if np.any(weights < -1e-10) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must lie on the probability simplex.")

    lower = lower_predictions @ weights
    upper = upper_predictions @ weights
    scores = np.maximum(lower - y, y - upper)
    qhat = conformal_quantile(scores, alpha)
    lengths = np.maximum(upper - lower + 2 * qhat, 0.0)
    return float(np.mean(lengths)), qhat


def conformal_length_weights(
    lower_predictions: np.ndarray,
    upper_predictions: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.1,
    random_state: int = 0,
    n_random_starts: int = 8,
    maxiter: int = 300,
) -> tuple[np.ndarray, float, float]:
    """Learn shared pair weights by minimizing OOF conformalized mean length."""

    lower_predictions = np.asarray(lower_predictions, dtype=float)
    upper_predictions = np.asarray(upper_predictions, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if lower_predictions.ndim != 2 or lower_predictions.shape != upper_predictions.shape:
        raise ValueError("lower_predictions and upper_predictions must have equal 2D shapes.")
    if lower_predictions.shape[0] != y.size:
        raise ValueError("predictions and y have incompatible shapes.")
    if lower_predictions.shape[1] == 0:
        raise ValueError("At least one quantile pair is required.")
    if n_random_starts < 0:
        raise ValueError("n_random_starts must be nonnegative.")

    n_pairs = lower_predictions.shape[1]

    def objective(candidate: np.ndarray) -> float:
        candidate = np.maximum(np.asarray(candidate, dtype=float), 0.0)
        total = candidate.sum()
        if not np.isfinite(total) or total <= 0:
            return np.inf
        candidate /= total
        return conformalized_mean_length(
            lower_predictions,
            upper_predictions,
            y,
            candidate,
            alpha,
        )[0]

    starts: list[np.ndarray] = []
    for size in range(1, n_pairs + 1):
        candidate = np.zeros(n_pairs, dtype=float)
        candidate[:size] = 1 / size
        starts.append(candidate)
    rng = np.random.default_rng(random_state)
    starts.extend(rng.dirichlet(np.ones(n_pairs), size=n_random_starts))

    best_weights = starts[0].copy()
    best_objective = objective(best_weights)
    constraints = {"type": "eq", "fun": lambda weights: weights.sum() - 1.0}
    bounds = [(0.0, 1.0)] * n_pairs
    for start in starts:
        start_objective = objective(start)
        if start_objective < best_objective:
            best_weights, best_objective = start.copy(), start_objective
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
        )
        if not np.all(np.isfinite(result.x)):
            continue
        candidate = np.maximum(result.x, 0.0)
        if candidate.sum() <= 0:
            continue
        candidate /= candidate.sum()
        candidate_objective = objective(candidate)
        if candidate_objective < best_objective:
            best_weights, best_objective = candidate, candidate_objective

    best_qhat = conformalized_mean_length(
        lower_predictions,
        upper_predictions,
        y,
        best_weights,
        alpha,
    )[1]
    return best_weights, best_objective, best_qhat


def bandwidth_grids(
    bandwidth: float,
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower and upper quantile grids for one bandwidth."""

    q_low, q_high = map(float, quantiles)
    bandwidth = float(bandwidth)
    if not 0 < q_low < q_high < 1:
        raise ValueError("quantiles must satisfy 0 < q_low < q_high < 1.")
    if n_quantiles < 1:
        raise ValueError("n_quantiles must be positive.")
    if bandwidth < 0:
        raise ValueError("bandwidth must be nonnegative.")
    if q_low + bandwidth > q_high - bandwidth:
        raise ValueError("The lower and upper bandwidth grids cross.")
    grid_size = 1 if bandwidth == 0 else n_quantiles
    lower = np.linspace(q_low, q_low + bandwidth, grid_size)
    upper = np.linspace(q_high, q_high - bandwidth, grid_size)
    return lower, upper


def candidate_quantile_union(
    bandwidths: Sequence[float],
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
    extra_quantiles: Sequence[float] = (),
) -> np.ndarray:
    """Return one sorted quantile union for all bandwidth candidates."""

    requested = [np.asarray(extra_quantiles, dtype=float).reshape(-1)]
    for bandwidth in bandwidths:
        lower, upper = bandwidth_grids(bandwidth, n_quantiles, quantiles)
        requested.extend([lower, upper])
    combined = np.concatenate(requested) if requested else np.empty(0)
    if combined.size == 0:
        raise ValueError("At least one candidate or extra quantile is required.")
    return np.unique(np.round(combined, 12))


def quantile_column_indices(
    available_quantiles: Sequence[float],
    requested_quantiles: Sequence[float],
) -> np.ndarray:
    """Locate requested levels in a cached quantile union."""

    available = np.asarray(available_quantiles, dtype=float).reshape(-1)
    requested = np.asarray(requested_quantiles, dtype=float).reshape(-1)
    indices = []
    for quantile in requested:
        matches = np.flatnonzero(
            np.isclose(available, quantile, rtol=0.0, atol=1e-12)
        )
        if matches.size != 1:
            raise KeyError(
                f"Quantile {quantile} is absent or duplicated in the OOF cache."
            )
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def select_bandwidth_from_oof(
    predictions: np.ndarray,
    prediction_quantiles: Sequence[float],
    y: np.ndarray,
    bandwidths: Sequence[float],
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
    alpha: float = 0.1,
) -> dict:
    """Select ``d`` by OOF conformalized mean interval length."""

    predictions = np.asarray(predictions, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[0] != y.size:
        raise ValueError("OOF predictions and responses have incompatible shapes.")
    if bandwidths.size == 0 or not np.isfinite(bandwidths).all():
        raise ValueError("At least one finite bandwidth candidate is required.")
    if np.unique(np.round(bandwidths, 12)).size != bandwidths.size:
        raise ValueError("Bandwidth candidates must be unique.")

    diagnostics = []
    for bandwidth in bandwidths:
        lower_grid, upper_grid = bandwidth_grids(
            bandwidth, n_quantiles, quantiles
        )
        uniform_weights = np.full(lower_grid.size, 1.0 / lower_grid.size)
        lower = predictions[
            :, quantile_column_indices(prediction_quantiles, lower_grid)
        ]
        upper = predictions[
            :, quantile_column_indices(prediction_quantiles, upper_grid)
        ]
        objective, qhat = conformalized_mean_length(
            lower,
            upper,
            y,
            uniform_weights,
            alpha,
        )
        diagnostics.append(
            {
                "bandwidth": float(bandwidth),
                "oof_conformal_mean_length": objective,
                "oof_conformal_qhat": qhat,
            }
        )

    best_index = int(
        np.argmin([candidate["oof_conformal_mean_length"] for candidate in diagnostics])
    )
    for index, candidate in enumerate(diagnostics):
        candidate["selected"] = index == best_index
    best = diagnostics[best_index]
    return {
        "bandwidth": best["bandwidth"],
        "objective": best["oof_conformal_mean_length"],
        "oof_qhat": best["oof_conformal_qhat"],
        "candidates": diagnostics,
    }

def select_joint_bandwidth_weights_from_oof(
    predictions: np.ndarray,
    prediction_quantiles: Sequence[float],
    y: np.ndarray,
    bandwidths: Sequence[float],
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
    alpha: float = 0.1,
    random_state: int = 0,
    n_random_starts: int = 8,
    maxiter: int = 300,
) -> dict:
    """Select ``(d, w)`` by OOF conformalized mean interval length."""

    predictions = np.asarray(predictions, dtype=float)
    prediction_quantiles = np.asarray(prediction_quantiles, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[0] != y.size:
        raise ValueError("OOF predictions and responses have incompatible shapes.")
    if predictions.shape[1] != prediction_quantiles.size:
        raise ValueError("OOF predictions and quantile levels have incompatible shapes.")
    if bandwidths.size == 0 or not np.isfinite(bandwidths).all():
        raise ValueError("At least one finite bandwidth candidate is required.")
    if np.unique(np.round(bandwidths, 12)).size != bandwidths.size:
        raise ValueError("Bandwidth candidates must be unique.")
    if n_random_starts < 0 or maxiter < 1:
        raise ValueError("n_random_starts must be nonnegative and maxiter positive.")

    diagnostics = []
    for index, bandwidth in enumerate(bandwidths):
        lower_grid, upper_grid = bandwidth_grids(
            bandwidth, n_quantiles=n_quantiles, quantiles=quantiles
        )
        lower = predictions[
            :, quantile_column_indices(prediction_quantiles, lower_grid)
        ]
        upper = predictions[
            :, quantile_column_indices(prediction_quantiles, upper_grid)
        ]
        if lower_grid.size == 1:
            weights = np.ones(1, dtype=float)
            objective, qhat = conformalized_mean_length(
                lower,
                upper,
                y,
                weights,
                alpha=alpha,
            )
        else:
            weights, objective, qhat = conformal_length_weights(
                lower,
                upper,
                y,
                alpha=alpha,
                random_state=int(random_state) + index,
                n_random_starts=n_random_starts,
                maxiter=maxiter,
            )
        diagnostics.append(
            {
                "bandwidth": float(bandwidth),
                "lower_grid": lower_grid,
                "upper_grid": upper_grid,
                "weights": weights,
                "oof_conformal_mean_length": float(objective),
                "oof_conformal_qhat": float(qhat),
            }
        )

    best_index = int(
        np.argmin(
            [candidate["oof_conformal_mean_length"] for candidate in diagnostics]
        )
    )
    for index, candidate in enumerate(diagnostics):
        candidate["selected"] = index == best_index
    best = diagnostics[best_index]
    return {
        "bandwidth": best["bandwidth"],
        "lower_grid": best["lower_grid"],
        "upper_grid": best["upper_grid"],
        "weights": best["weights"],
        "objective": best["oof_conformal_mean_length"],
        "oof_qhat": best["oof_conformal_qhat"],
        "candidates": diagnostics,
    }


def select_ccqr_from_oof(
    predictions: np.ndarray,
    prediction_quantiles: Sequence[float],
    y: np.ndarray,
    bandwidths: Sequence[float],
    *,
    selection: SelectionMode = "both",
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
    alpha: float = 0.1,
    d: float | None = None,
    random_state: int = 0,
    n_random_starts: int = 8,
    maxiter: int = 300,
) -> dict:
    """Select the CCQR construction requested by ``selection``.

    Parameters
    ----------
    predictions:
        OOF quantile predictions with shape ``(n_samples, n_levels)``.
    prediction_quantiles:
        Quantile level corresponding to each prediction column.
    y:
        Proper-training responses corresponding to the OOF rows.
    bandwidths:
        Candidate ``d`` values. ``d`` and ``both`` search this sequence.
        ``w`` uses its largest value unless ``d`` is supplied.
    selection:
        ``"d"`` for uniform weights with selected bandwidth, ``"w"`` for
        optimized weights on one fixed bandwidth, or ``"both"`` for joint
        profile selection of bandwidth and weights.

    Returns
    -------
    dict
        Every mode returns ``selection``, ``bandwidth``, ``lower_grid``,
        ``upper_grid``, ``weights``, ``objective``, ``oof_qhat``, and
        ``candidates``.
    """

    if selection not in {"d", "w", "both"}:
        raise ValueError("selection must be one of 'd', 'w', or 'both'.")
    if selection != "w" and d is not None:
        raise ValueError("d is used only when selection='w'.")

    predictions = np.asarray(predictions, dtype=float)
    prediction_quantiles = np.asarray(prediction_quantiles, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[0] != y.size:
        raise ValueError("OOF predictions and responses have incompatible shapes.")
    if predictions.shape[1] != prediction_quantiles.size:
        raise ValueError("OOF predictions and quantile levels have incompatible shapes.")
    if bandwidths.size == 0 or not np.isfinite(bandwidths).all():
        raise ValueError("At least one finite bandwidth candidate is required.")

    if selection == "d":
        result = select_bandwidth_from_oof(
            predictions,
            prediction_quantiles,
            y,
            bandwidths,
            n_quantiles=n_quantiles,
            quantiles=quantiles,
            alpha=alpha,
        )
        lower_grid, upper_grid = bandwidth_grids(
            result["bandwidth"],
            n_quantiles=n_quantiles,
            quantiles=quantiles,
        )
        weights = np.full(lower_grid.size, 1.0 / lower_grid.size)
        return {
            "selection": "d",
            "bandwidth": result["bandwidth"],
            "lower_grid": lower_grid,
            "upper_grid": upper_grid,
            "weights": weights,
            "objective": result["objective"],
            "oof_qhat": result["oof_qhat"],
            "candidates": result["candidates"],
        }

    if selection == "both":
        result = select_joint_bandwidth_weights_from_oof(
            predictions,
            prediction_quantiles,
            y,
            bandwidths,
            n_quantiles=n_quantiles,
            quantiles=quantiles,
            alpha=alpha,
            random_state=random_state,
            n_random_starts=n_random_starts,
            maxiter=maxiter,
        )
        return {"selection": "both", **result}

    fixed_bandwidth = (
        float(np.max(bandwidths))
        if d is None
        else float(d)
    )
    if not np.isfinite(fixed_bandwidth):
        raise ValueError("d must be finite.")
    lower_grid, upper_grid = bandwidth_grids(
        fixed_bandwidth,
        n_quantiles=n_quantiles,
        quantiles=quantiles,
    )
    lower = predictions[
        :, quantile_column_indices(prediction_quantiles, lower_grid)
    ]
    upper = predictions[
        :, quantile_column_indices(prediction_quantiles, upper_grid)
    ]
    if lower_grid.size == 1:
        weights = np.ones(1, dtype=float)
        objective, qhat = conformalized_mean_length(
            lower,
            upper,
            y,
            weights,
            alpha=alpha,
        )
    else:
        fixed_index = np.flatnonzero(
            np.isclose(bandwidths, fixed_bandwidth, rtol=0.0, atol=1e-12)
        )
        seed_offset = int(fixed_index[0]) if fixed_index.size else 0
        weights, objective, qhat = conformal_length_weights(
            lower,
            upper,
            y,
            alpha=alpha,
            random_state=int(random_state) + seed_offset,
            n_random_starts=n_random_starts,
            maxiter=maxiter,
        )
    candidate = {
        "bandwidth": fixed_bandwidth,
        "lower_grid": lower_grid,
        "upper_grid": upper_grid,
        "weights": weights,
        "oof_conformal_mean_length": float(objective),
        "oof_conformal_qhat": float(qhat),
        "selected": True,
    }
    return {
        "selection": "w",
        "bandwidth": fixed_bandwidth,
        "lower_grid": lower_grid,
        "upper_grid": upper_grid,
        "weights": weights,
        "objective": float(objective),
        "oof_qhat": float(qhat),
        "candidates": [candidate],
    }


def select_ccqr(
    model_factory: Callable[[int], object],
    X: np.ndarray,
    y: np.ndarray,
    bandwidths: Sequence[float],
    *,
    selection: SelectionMode = "both",
    d: float | None = None,
    K: int = 5,
    n_quantiles: int = 9,
    quantiles: Sequence[float] = (0.05, 0.95),
    alpha: float = 0.1,
    random_state: int = 0,
    n_random_starts: int = 8,
    maxiter: int = 300,
) -> dict:
    """Cross-fit a quantile learner and select CCQR(d), CCQR(w), or both.

    ``model_factory`` is called once per fold with a deterministic integer seed
    and must return a fresh, unfitted model implementing ``fit(X, y)`` and
    ``predict(X, quantiles=[...])``. ``K`` is the number of cross-fitting folds;
    ``n_quantiles`` is the number of symmetric quantile pairs and is a separate
    parameter.

    The fold construction and fold-model seed rule match the current synthetic
    experiment runner: shuffled K-fold splits use ``random_state``, and fold
    number ``f`` receives ``SeedSequence([random_state, f, 941])``.
    """

    if not callable(model_factory):
        raise TypeError("model_factory must be callable.")
    X = np.asarray(X)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional feature matrix.")
    if X.shape[0] != y.size:
        raise ValueError("X and y must contain the same number of observations.")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)):
        raise TypeError("K must be an integer number of cross-fitting folds.")
    K = int(K)
    if K < 2 or K > y.size:
        raise ValueError("K must lie between 2 and the number of observations.")

    bandwidths_array = np.asarray(bandwidths, dtype=float).reshape(-1)
    extra_quantiles: np.ndarray | Sequence[float] = ()
    if selection == "w" and d is not None:
        d_lower, d_upper = bandwidth_grids(
            d,
            n_quantiles=n_quantiles,
            quantiles=quantiles,
        )
        extra_quantiles = np.concatenate([d_lower, d_upper])
    prediction_quantiles = candidate_quantile_union(
        bandwidths_array,
        n_quantiles=n_quantiles,
        quantiles=quantiles,
        extra_quantiles=extra_quantiles,
    )
    oof_predictions = np.empty(
        (y.size, prediction_quantiles.size),
        dtype=float,
    )

    shuffled = np.arange(y.size)
    np.random.RandomState(random_state).shuffle(shuffled)
    fold_sizes = np.full(K, y.size // K, dtype=int)
    fold_sizes[: y.size % K] += 1
    start = 0
    for fold, fold_size in enumerate(fold_sizes):
        stop = start + int(fold_size)
        validation_index = shuffled[start:stop]
        train_mask = np.ones(y.size, dtype=bool)
        train_mask[validation_index] = False
        train_index = np.flatnonzero(train_mask)
        fold_seed = int(
            np.random.SeedSequence([random_state, fold, 941]).generate_state(1)[0]
        )
        model = model_factory(fold_seed)
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError(
                "model_factory must return an object implementing fit and predict."
            )
        model.fit(X[train_index], y[train_index])
        fold_predictions = np.asarray(
            model.predict(
                X[validation_index],
                quantiles=prediction_quantiles.tolist(),
            ),
            dtype=float,
        )
        if fold_predictions.ndim == 1:
            fold_predictions = fold_predictions[:, None]
        expected_shape = (validation_index.size, prediction_quantiles.size)
        if fold_predictions.shape != expected_shape:
            raise ValueError(
                "The base learner returned OOF predictions with shape "
                f"{fold_predictions.shape}; expected {expected_shape}."
            )
        oof_predictions[validation_index] = fold_predictions
        start = stop

    result = select_ccqr_from_oof(
        oof_predictions,
        prediction_quantiles,
        y,
        bandwidths_array,
        selection=selection,
        n_quantiles=n_quantiles,
        quantiles=quantiles,
        alpha=alpha,
        d=d,
        random_state=random_state,
        n_random_starts=n_random_starts,
        maxiter=maxiter,
    )
    return {
        **result,
        "K": K,
        "prediction_quantiles": prediction_quantiles,
    }

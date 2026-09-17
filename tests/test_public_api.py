import numpy as np

from ccqr import (
    ConformalLengthWeightedCQR,
    EndpointAveragedCQR,
    StandardCQR,
    bandwidth_grids,
    candidate_quantile_union,
    conformal_length_weights,
    conformal_quantile,
    select_ccqr,
    select_ccqr_from_oof,
    select_joint_bandwidth_weights_from_oof,
)


class LinearQuantileModel:
    def fit(self, X, y):
        return self

    def predict(self, X, quantiles):
        x = np.asarray(X, dtype=float).reshape(-1)
        q = np.asarray(quantiles, dtype=float)
        return x[:, None] + (q[None, :] - 0.5)


def test_finite_sample_quantile_uses_order_statistic():
    scores = np.arange(10, dtype=float)
    assert conformal_quantile(scores, 0.2) == 8.0


def test_bandwidth_zero_is_exactly_one_cqr_pair():
    lower, upper = bandwidth_grids(0.0, n_quantiles=9)
    np.testing.assert_array_equal(lower, [0.05])
    np.testing.assert_array_equal(upper, [0.95])


def test_weight_optimizer_is_deterministic_and_on_simplex():
    rng = np.random.default_rng(13)
    lower = rng.normal(size=(80, 4))
    upper = lower + rng.uniform(0.5, 1.5, size=(80, 4))
    y = rng.normal(size=80)
    first = conformal_length_weights(lower, upper, y, random_state=7, n_random_starts=2)
    second = conformal_length_weights(lower, upper, y, random_state=7, n_random_starts=2)
    np.testing.assert_allclose(first[0], second[0])
    assert np.isclose(first[0].sum(), 1.0)
    assert np.all(first[0] >= 0)


def test_joint_selection_and_predictors_use_public_api():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(90, 1))
    y = X[:, 0] + rng.normal(scale=0.2, size=90)
    model = LinearQuantileModel()
    d_values = np.array([0.0, 0.1])
    levels = candidate_quantile_union(d_values, n_quantiles=3)
    predictions = model.predict(X, levels)
    selected = select_joint_bandwidth_weights_from_oof(
        predictions,
        levels,
        y,
        d_values,
        n_quantiles=3,
        random_state=4,
        n_random_starts=1,
        maxiter=30,
    )
    assert selected["bandwidth"] in d_values
    assert np.isclose(selected["weights"].sum(), 1.0)

    predictors = [
        StandardCQR(model).calibrate(X[:30], y[:30]),
        EndpointAveragedCQR(model, bandwidth=0.1, n_quantiles=3).calibrate(
            X[:30], y[:30]
        ),
        ConformalLengthWeightedCQR(
            model,
            selected["lower_grid"],
            selected["upper_grid"],
            selected["weights"],
        ).calibrate(X[:30], y[:30]),
    ]
    for predictor in predictors:
        intervals = predictor.predict(X[30:])
        assert intervals.shape == (60, 2)
        assert np.isfinite(intervals).all()


def test_unified_selector_supports_d_w_and_both():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(70, 1))
    y = X[:, 0] + rng.normal(scale=0.25, size=70)
    model = LinearQuantileModel()
    bandwidths = np.array([0.0, 0.1, 0.2])
    levels = candidate_quantile_union(bandwidths, n_quantiles=3)
    predictions = model.predict(X, levels)

    results = {}
    for mode in ("d", "w", "both"):
        result = select_ccqr_from_oof(
            predictions,
            levels,
            y,
            bandwidths,
            selection=mode,
            n_quantiles=3,
            random_state=2,
            n_random_starts=1,
            maxiter=30,
        )
        results[mode] = result
        assert result["selection"] == mode
        assert set(
            [
                "bandwidth",
                "lower_grid",
                "upper_grid",
                "weights",
                "objective",
                "oof_qhat",
                "candidates",
            ]
        ).issubset(result)
        assert result["lower_grid"].shape == result["upper_grid"].shape
        assert result["weights"].shape == result["lower_grid"].shape
        assert np.isclose(result["weights"].sum(), 1.0)

    assert results["w"]["bandwidth"] == bandwidths.max()
    broad_joint_candidate = next(
        candidate
        for candidate in results["both"]["candidates"]
        if candidate["bandwidth"] == bandwidths.max()
    )
    np.testing.assert_allclose(
        results["w"]["weights"], broad_joint_candidate["weights"]
    )
    np.testing.assert_allclose(
        results["d"]["weights"],
        np.full(results["d"]["weights"].size, 1 / results["d"]["weights"].size),
    )

    fixed_d = select_ccqr_from_oof(
        predictions,
        levels,
        y,
        bandwidths,
        selection="w",
        d=0.1,
        n_quantiles=3,
        random_state=2,
        n_random_starts=1,
        maxiter=30,
    )
    assert fixed_d["bandwidth"] == 0.1


def test_high_level_selector_cross_fits_with_requested_K():
    rng = np.random.default_rng(202)
    X = rng.normal(size=(48, 1))
    y = X[:, 0] + rng.normal(scale=0.2, size=48)
    fold_seeds = []

    def model_factory(seed):
        fold_seeds.append(seed)
        return LinearQuantileModel()

    result = select_ccqr(
        model_factory,
        X,
        y,
        [0.0, 0.1],
        selection="d",
        K=4,
        n_quantiles=3,
    )
    assert result["selection"] == "d"
    assert result["K"] == 4
    assert len(fold_seeds) == 4
    assert len(set(fold_seeds)) == 4

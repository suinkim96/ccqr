# CCQR

`ccqr` is a small, learner-agnostic Python implementation of Composite
Conformalized Quantile Regression.  It contains the implementation used by the
current synthetic experiments:

- ordinary split CQR;
- `CCQR(d)`, which selects the quantile range from out-of-fold (OOF)
  predictions;
- `CCQR(w)`, which learns shared convex weights for symmetric quantile pairs;
- `CCQR(d,w)`, which jointly selects the range and pair weights.

Selection uses only proper-training OOF predictions.  The final conformal
calibration sample is kept untouched until the selected construction is fit.

## Installation

```bash
python -m pip install .
```

For the QRF reproduction example:

```bash
python -m pip install ".[examples]"
```

For development:

```bash
python -m pip install -e ".[test,examples]"
python -m pytest
```

## Base learner interface

CCQR accepts a model factory that returns a fresh quantile learner exposing

```python
model.fit(X, y)
model.predict(X, quantiles=[0.05, 0.10, 0.90, 0.95])
```

and returning an array of shape `(n_samples, n_quantiles)` in the requested
quantile order.

## Basic usage

```python
import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from ccqr import (
    ConformalLengthWeightedCQR,
    select_ccqr,
)

d_candidates = np.arange(0.0, 0.41, 0.05)

def model_factory(seed):
    return RandomForestQuantileRegressor(
        n_estimators=500,
        min_samples_leaf=10,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )

choice = select_ccqr(
    model_factory,
    X_proper,
    y_proper,
    d_candidates,
    selection="both",  # "d", "w", or "both"
    K=5,                # cross-fitting folds
    n_quantiles=9,
    alpha=0.1,
    random_state=0,
    n_random_starts=8,
    maxiter=300,
)

predictor = ConformalLengthWeightedCQR(
    final_model,
    lower_grid=choice["lower_grid"],
    upper_grid=choice["upper_grid"],
    weights=choice["weights"],
    alpha=0.1,
).calibrate(X_calibration, y_calibration)

intervals = predictor.predict(X_test)
```

The same selector covers all three CCQR variants:

```python
d_choice = select_ccqr(..., selection="d")
w_choice = select_ccqr(..., selection="w", d=0.4)
dw_choice = select_ccqr(..., selection="both")
```

All modes return the same core fields: `bandwidth`, `lower_grid`,
`upper_grid`, `weights`, `objective`, `oof_qhat`, and `candidates`.

The optimizer includes all prefix-uniform initializations and eight
`Dirichlet(1, ..., 1)` starts by default.  For joint selection, candidate
bandwidth number `i` uses RNG seed `random_state + i`.

If OOF predictions have already been computed, use the lower-level
`select_ccqr_from_oof` function. It accepts the OOF matrix and its quantile
levels directly and therefore has no `K` argument.

Calibration always uses the finite-sample split-conformal order statistic; no
calibration-rule option is exposed.

## Reproducibility check

The repository-level experiment runner in the accompanying research project
imports this package directly.  A one-seed QRF check is recorded in
`REPRODUCIBILITY.md`.

This repository intentionally contains the method and its tests, not datasets
or full experiment outputs.  Add an explicit open-source license before making
the repository public.

# Conformalized Composite Quantile Regression (CCQR)

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

For the QRF examples:

```bash
python -m pip install ".[examples]"
```

For development:

```bash
python -m pip install -e ".[test,examples]"
python -m pytest
```

## Toy notebook

[Run the complete toy example](examples/toy_ccqr.ipynb) to generate a seeded
one-dimensional heteroscedastic dataset, select CCQR(d,w), calibrate the
intervals, and compare them with ordinary CQR using the same QRF base learner.
The notebook includes executed results and a prediction-interval plot, so it
can also be read directly on GitHub.

From the repository root:

```bash
python -m pip install -e ".[examples,notebook]"
python -m jupyterlab examples/toy_ccqr.ipynb
```

Select **Restart Kernel and Run All Cells** to reproduce the example. It
generates all data locally with seed 0; no data download or research-project
files are required. Proper-training, calibration, and test sizes are
1,000, 1,000, and 5,000, respectively. Only proper-training data enter the
five-fold selection step. The QRF settings are fixed; its hyperparameters
are not tuned.

To execute the notebook without opening JupyterLab and refresh its saved
outputs and PNG:

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 examples/toy_ccqr.ipynb
```

![CQR and CCQR prediction intervals on the toy data](examples/toy_ccqr_intervals.png)

The metrics describe one simulation draw, not a repeated-simulation claim of
superiority. The 90% target is marginal coverage; realized coverage on one
test set and coverage conditional on a particular feature value can differ.

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

final_model = model_factory(seed=0)
final_model.fit(X_proper, y_proper)

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

This repository contains the method, its tests, and the self-contained toy
notebook, but not the full research datasets or experiment outputs.
Add an explicit open-source license before making
the repository public.

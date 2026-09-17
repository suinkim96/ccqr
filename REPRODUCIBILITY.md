# One-seed reproduction check

The packaged public API was checked against the stored main-simulation result
on 2026-09-17.

- package version: 0.4.0
- experiment-runner entry point: `select_ccqr_from_oof(selection="d" | "w" | "both")`
- high-level cross-fitting entry point: `select_ccqr(..., K=5)`

## Condition

- DGP: controlled normal location-scale model
- base learner: QRF
- repetition seed: 0
- proper-training/calibration/test sizes: 200/200/5000
- heteroscedasticity: `gamma=0`
- trees: 500
- minimum leaf size: 10
- maximum features: 1.0
- OOF folds: 5
- quantile pairs: 9
- weight optimizer: 8 random starts, 300 maximum iterations
- Python: 3.12.13

The verification run used `n_jobs=1` because the managed Windows test
environment does not permit the worker pipes created by `n_jobs=-1`. Forest
predictions and all statistical outputs were unchanged; only runtime was
excluded from the comparison.

## Result

| Method | Coverage | Mean length | qhat | Selected d |
|---|---:|---:|---:|---:|
| CQR | 0.8804 | 0.7290542481523714 | -0.03204588582315171 | -- |
| CCQR(d) | 0.8978 | 0.7334113922173078 | 0.08036576846561927 | 0.15 |
| CCQR(w) | 0.8648 | 0.6855766486850767 | 0.14079026520809879 | -- |
| CCQR(d,w) | 0.8850 | 0.7182282150621577 | 0.06803018347076328 | 0.10 |

The package-generated files were compared to the corresponding seed-0 rows
from `aistats_controlled_qrf_300reps_v6`:

| Artifact | Reference rows | Package rows | Maximum absolute numeric difference |
|---|---:|---:|---:|
| `results_long.csv` | 4 | 4 | 9.72e-17 |
| `d_selection.csv` | 9 | 9 | 8.33e-17 |
| `learned_weights.csv` | 9 | 9 | 8.33e-17 |
| `dw_selection.csv` | 9 | 9 | 8.33e-17 |
| `learned_joint_weights.csv` | 73 | 73 | 1.11e-16 |

There were no nonnumeric mismatches. These differences are machine-precision
roundoff, so the packaged implementation reproduces the current stored result.

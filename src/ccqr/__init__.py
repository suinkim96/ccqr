"""Public API for Composite Conformalized Quantile Regression."""

from ._version import __version__
from .calibration import conformal_quantile
from .predictors import ConformalLengthWeightedCQR, EndpointAveragedCQR, StandardCQR
from .selection import (
    SelectionMode,
    bandwidth_grids,
    candidate_quantile_union,
    conformal_length_weights,
    conformalized_mean_length,
    quantile_column_indices,
    select_ccqr,
    select_ccqr_from_oof,
    select_bandwidth_from_oof,
    select_joint_bandwidth_weights_from_oof,
)

__all__ = [
    "__version__",
    "conformal_quantile",
    "StandardCQR",
    "EndpointAveragedCQR",
    "ConformalLengthWeightedCQR",
    "SelectionMode",
    "conformalized_mean_length",
    "conformal_length_weights",
    "bandwidth_grids",
    "candidate_quantile_union",
    "quantile_column_indices",
    "select_ccqr",
    "select_ccqr_from_oof",
    "select_bandwidth_from_oof",
    "select_joint_bandwidth_weights_from_oof",
]

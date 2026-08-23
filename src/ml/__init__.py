"""Machine learning surrogate suite module."""

from src.ml.models import (
    BaseSurrogate,
    StandardScaler,
    LinearRidgeSurrogate,
    RandomForestSurrogate,
    GradientBoostingSurrogate,
    NeuralSurrogate,
)
from src.ml.evaluate import compute_regression_metrics, benchmark_speedup
from src.ml.train import SurrogateSuite, train_surrogate_suite

__all__ = [
    "BaseSurrogate",
    "StandardScaler",
    "LinearRidgeSurrogate",
    "RandomForestSurrogate",
    "GradientBoostingSurrogate",
    "NeuralSurrogate",
    "compute_regression_metrics",
    "benchmark_speedup",
    "SurrogateSuite",
    "train_surrogate_suite",
]

from __future__ import annotations

import warnings
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from models import NormalityReport, ShapiroResult


def _to_numeric_series(values: pd.Series | Sequence[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values, dtype="float64"), errors="coerce")


def _clean_numeric_values(values: pd.Series | Sequence[float]) -> tuple[pd.Series, int]:
    numeric = _to_numeric_series(values)
    finite_mask = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
    clean = numeric[finite_mask]
    excluded = int((~finite_mask).sum())
    return clean, excluded


def _downsample_evenly(values: np.ndarray, max_points: int | None) -> np.ndarray:
    if max_points is None or len(values) <= max_points:
        return values
    sample_idx = np.linspace(0, len(values) - 1, num=max_points, dtype=int)
    return values[sample_idx]


def center_series(values: pd.Series | Sequence[float]) -> pd.Series:
    numeric = _to_numeric_series(values)
    return numeric - numeric.mean()


def plot_histogram(
    values: pd.Series | Sequence[float],
    *,
    bins: int = 50,
    ax: plt.Axes | None = None,
    title: str = "Histogram",
    color: str = "#2E5EAA",
) -> tuple[plt.Figure, plt.Axes]:
    clean, _ = _clean_numeric_values(values)
    if clean.empty:
        raise ValueError("Histogram requires at least one finite numeric value.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4))
    else:
        figure = ax.figure

    ax.hist(clean.to_numpy(copy=False), bins=bins, color=color, edgecolor="white", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    figure.tight_layout()
    return figure, ax


def plot_qq(
    values: pd.Series | Sequence[float],
    *,
    ax: plt.Axes | None = None,
    title: str = "Q-Q Plot",
    max_points: int | None = 5_000,
    color: str = "#2E5EAA",
) -> tuple[plt.Figure, plt.Axes]:
    clean, _ = _clean_numeric_values(values)
    if clean.empty:
        raise ValueError("Q-Q plot requires at least one finite numeric value.")

    sampled = _downsample_evenly(clean.to_numpy(copy=False), max_points=max_points)
    (osm, osr), (slope, intercept, _) = stats.probplot(sampled, dist="norm", fit=True)

    if ax is None:
        figure, ax = plt.subplots(figsize=(6, 6))
    else:
        figure = ax.figure

    ax.scatter(osm, osr, s=12, alpha=0.7, color=color)
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, intercept + slope * x_line, color="#222222", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed quantiles")
    ax.grid(True, alpha=0.2)
    figure.tight_layout()
    return figure, ax


def shapiro_wilk_test(
    values: pd.Series | Sequence[float],
    *,
    alpha: float = 0.05,
) -> ShapiroResult:
    clean, _ = _clean_numeric_values(values)
    if len(clean) < 3:
        raise ValueError("Shapiro-Wilk requires at least three finite numeric values.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        statistic, p_value = stats.shapiro(clean.to_numpy(copy=False))

    notes = tuple(str(item.message) for item in caught)
    return ShapiroResult(
        statistic=float(statistic),
        p_value=float(p_value),
        sample_size=int(len(clean)),
        alpha=alpha,
        notes=notes,
    )


def normality_report(
    values: pd.Series | Sequence[float],
    *,
    alpha: float = 0.05,
) -> NormalityReport:
    clean, excluded = _clean_numeric_values(values)
    if len(clean) < 3:
        raise ValueError("Normality report requires at least three finite numeric values.")

    shapiro = shapiro_wilk_test(clean, alpha=alpha)
    notes = list(shapiro.notes)
    if excluded:
        notes.append(f"Excluded {excluded} non-finite values before statistical tests.")

    return NormalityReport(
        sample_size=int(len(clean)),
        excluded_non_finite=excluded,
        mean=float(clean.mean()),
        standard_deviation=float(clean.std(ddof=1)),
        shapiro=shapiro,
        notes=tuple(notes),
    )


__all__ = [
    "center_series",
    "normality_report",
    "plot_histogram",
    "plot_qq",
    "shapiro_wilk_test",
]

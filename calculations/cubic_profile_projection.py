"""
Cubic Profile Projection — Quasi-Steady-State CO₂ Fitting
===========================================================

Mathematical approach
---------------------
Unlike V1–V4 (which fit a *parameterised* PDE model by least-squares),
this script uses the **exact cubic family of solutions** to the steady-state
diffusion-reaction ODE.

Governing ODE (steady-state per 15-min snapshot):
    d/dz [ D_eff(z) · dC/dz ] + S(z) = 0

The cubic polynomial C(z) = az³ + bz² + cz + d has exactly four free
coefficients.  We constrain them with three direct measurements plus the
physical Neumann BC at the basalt floor:

    C(z₁) = C₁(t)      — sensor at 5 cm
    C(z₂) = C₂(t)      — sensor at 20 cm
    C(z₃) = C₃(t)      — sensor at deep (35 cm or 50 cm, vertical-specific)
    C'(L)  = 0          — zero-flux at base (z = L = 1.0 m)

This gives the system  M · [a, b, c, d]ᵀ = [C₁, C₂, C₃, 0]ᵀ  where M is the
constant geometry matrix.  M is **inverted once** per vertical and applied to
all 288 (or more) timesteps by a single matrix-vector product.

At every timestep t the coefficients immediately yield:
    C_s(t)  = d(t)           — inferred surface CO₂ at z = 0
    C'(0,t) = c(t)           — inferred surface flux gradient
    S_total(t) = −D_eff · C''(z̄, t)   — source term at mid-profile (§ see below)

Two output plots
-----------------
Figure 1 — Mean depth profiles (one per ideal period/vertical)
    For each of the 8 ideal-period verticals, compute the time-mean
    concentrations at each basalt depth, fit the cubic, and draw the
    smooth profile from z = 0 (surface) to z = L (1 m).  Observed mean
    data points are overlaid as scatter markers.

Figure 2 — Per-period time series  (4 × 2 subplot grid)
    For each vertical, plot:
    • Observed CO₂ at each basalt sensor depth (scatter, 3 colours)
    • Cubic-inferred surface concentration d(t) at z = 0  (dashed, black)
    • Actual air sensor CO₂ (dotted, grey) — **validation of the cubic fit**
    The gap between d(t) and the measured air CO₂ shows how well the
    model extrapolates from the basalt column to the surface.

Data source
-----------
    Oracle SensorDB via the project's `BasaltCO2Series` and `AirCO2Series`
    classes.  Requires Oracle Instant Client and project `.env` credentials.

Sensor inventory (all ideal periods)
--------------------------------------
See IDEAL_PERIODS catalogue below.  All basalt sensors use
    table: leo_west.datavalues, variable_id: 9
All air sensors use
    table: leo_west.datavalueslicor, variable_id: 56

Usage
-----
    python cubic_profile_projection.py

Outputs (saved in calculations/)
----------------------------------
    cubic_mean_profiles.png    — Figure 1
    cubic_timeseries.png       — Figure 2
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ── Add sensorDB to import path ────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SENSORDB_DIR = _REPO_ROOT / "Project_description" / "sensorDB"
if str(_SENSORDB_DIR) not in sys.path:
    sys.path.insert(0, str(_SENSORDB_DIR))

try:
    from basalt_co2_series import BasaltCO2Series
    from air_co2_series import AirCO2Series
    _ORACLE_AVAILABLE = True
except ImportError as _e:
    _ORACLE_AVAILABLE = False
    _IMPORT_ERROR = str(_e)


# ══════════════════════════════════════════════════════════════════════════════
# IDEAL PERIOD CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════

# Each entry fully describes one ideal-period 4-sensor vertical.
# "depth_deep_m" is either 0.35 or 0.50 depending on the vertical.

IDEAL_PERIODS: list[dict] = [
    {
        "label":        "x=−1, y=4",
        "ideal_start":  "2024-07-10 12:30:00",
        "ideal_end":    "2024-07-29 14:45:00",
        "sensor_5cm":   995,
        "sensor_20cm":  1011,
        "sensor_deep":  1034,
        "depth_deep_m": 0.50,
        "basalt_x":     -1, "basalt_y": 4,
        "air_sensor":   1275, "air_x": 0, "air_y": 4,
        "air_code":     "LEO-W_4_0_1_LI-7000",
    },
    {
        "label":        "x=−1, y=10",
        "ideal_start":  "2025-03-26 16:30:00",
        "ideal_end":    "2025-04-06 18:15:00",
        "sensor_5cm":   999,
        "sensor_20cm":  1015,
        "sensor_deep":  1028,
        "depth_deep_m": 0.35,
        "basalt_x":     -1, "basalt_y": 10,
        "air_sensor":   1279, "air_x": -2, "air_y": 10,
        "air_code":     "LEO-W_10_-2_1_LI-7000",
    },
    {
        "label":        "x=−1, y=18",
        "ideal_start":  "2024-07-10 13:00:00",
        "ideal_end":    "2024-07-29 14:15:00",
        "sensor_5cm":   1003,
        "sensor_20cm":  1019,
        "sensor_deep":  1030,
        "depth_deep_m": 0.35,
        "basalt_x":     -1, "basalt_y": 18,
        "air_sensor":   1289, "air_x": 0, "air_y": 17,
        "air_code":     "LEO-W_17_0_1_LI-7000",
    },
    {
        "label":        "x=−1, y=24",
        "ideal_start":  "2025-08-25 12:45:00",
        "ideal_end":    "2025-09-13 18:00:00",
        "sensor_5cm":   1007,
        "sensor_20cm":  1023,
        "sensor_deep":  1040,
        "depth_deep_m": 0.50,
        "basalt_x":     -1, "basalt_y": 24,
        "air_sensor":   1294, "air_x": 0, "air_y": 24,
        "air_code":     "LEO-W_24_0_1_LI-7000",
    },
    {
        "label":        "x=+1, y=4",
        "ideal_start":  "2025-09-30 03:15:00",
        "ideal_end":    "2025-10-03 10:45:00",
        "sensor_5cm":   996,
        "sensor_20cm":  1012,
        "sensor_deep":  1035,
        "depth_deep_m": 0.50,
        "basalt_x":      1, "basalt_y": 4,
        "air_sensor":   1275, "air_x": 0, "air_y": 4,
        "air_code":     "LEO-W_4_0_1_LI-7000",
    },
    {
        "label":        "x=+1, y=10",
        "ideal_start":  "2025-08-28 17:30:00",
        "ideal_end":    "2025-09-10 08:30:00",
        "sensor_5cm":   1000,
        "sensor_20cm":  1016,
        "sensor_deep":  1029,
        "depth_deep_m": 0.35,
        "basalt_x":      1, "basalt_y": 10,
        "air_sensor":   1284, "air_x": 2, "air_y": 10,
        "air_code":     "LEO-W_10_2_1_LI-7000",
    },
    {
        "label":        "x=+1, y=18",
        "ideal_start":  "2025-09-30 04:00:00",
        "ideal_end":    "2025-10-03 10:00:00",
        "sensor_5cm":   1004,
        "sensor_20cm":  1020,
        "sensor_deep":  1031,
        "depth_deep_m": 0.35,
        "basalt_x":      1, "basalt_y": 18,
        "air_sensor":   1289, "air_x": 0, "air_y": 17,
        "air_code":     "LEO-W_17_0_1_LI-7000",
    },
    {
        "label":        "x=+1, y=24",
        "ideal_start":  "2025-08-25 12:45:00",
        "ideal_end":    "2025-09-13 18:00:00",
        "sensor_5cm":   1008,
        "sensor_20cm":  1024,
        "sensor_deep":  1041,
        "depth_deep_m": 0.50,
        "basalt_x":      1, "basalt_y": 24,
        "air_sensor":   1294, "air_x": 0, "air_y": 24,
        "air_code":     "LEO-W_24_0_1_LI-7000",
    },
]

# ── Domain ────────────────────────────────────────────────────────────────────
DOMAIN_L: float = 1.0        # basalt column depth [m]
SLOT_FREQ:  str = "15min"    # aggregation frequency
AIR_FRESHNESS = pd.Timedelta(minutes=90)   # max age for air observation


# ══════════════════════════════════════════════════════════════════════════════
# CUBIC POLYNOMIAL MATHEMATICS
# ══════════════════════════════════════════════════════════════════════════════

def build_geometry_matrix(z1: float, z2: float, z3: float, L: float = DOMAIN_L) -> np.ndarray:
    """
    Constant 4×4 geometry matrix M for the cubic family.

        Row 0: C(z₁) = C₁     →  [z1³, z1², z1, 1 ]
        Row 1: C(z₂) = C₂     →  [z2³, z2², z2, 1 ]
        Row 2: C(z₃) = C₃     →  [z3³, z3², z3, 1 ]
        Row 3: C'(L)  = 0      →  [3L², 2L,  1,  0 ]

    The system  M · [a, b, c, d]ᵀ = [C₁, C₂, C₃, 0]ᵀ  has a unique solution
    whenever z₁, z₂, z₃ are distinct.
    """
    M = np.array([
        [z1**3, z1**2, z1, 1.0],
        [z2**3, z2**2, z2, 1.0],
        [z3**3, z3**2, z3, 1.0],
        [3.0 * L**2, 2.0 * L, 1.0, 0.0],
    ])
    return M


def invert_geometry_matrix(M: np.ndarray) -> np.ndarray:
    """Invert M once.  Condition number is printed as a sanity check."""
    cond = np.linalg.cond(M)
    if cond > 1e10:
        print(f"  [WARNING] Geometry matrix is ill-conditioned (κ = {cond:.2e})")
    return np.linalg.inv(M)


def fit_cubic_timeseries(
    C1: np.ndarray,
    C2: np.ndarray,
    C3: np.ndarray,
    M_inv: np.ndarray,
) -> np.ndarray:
    """
    Fit cubic at every timestep via a single batched matrix-vector multiply.

    Parameters
    ----------
    C1, C2, C3 : (n_t,)  observed concentrations at sensor depths
    M_inv      : (4, 4)  pre-inverted geometry matrix

    Returns
    -------
    coeffs : (n_t, 4)   columns are [a(t), b(t), c(t), d(t)]
    """
    n_t = len(C1)
    rhs = np.column_stack([C1, C2, C3, np.zeros(n_t)])   # (n_t, 4)
    # M_inv @ rhs[i] for each row i  →  (n_t, 4) via broadcasting
    return (M_inv @ rhs.T).T                              # (n_t, 4)


def eval_cubic(coeffs: np.ndarray, z: float | np.ndarray) -> np.ndarray:
    """
    Evaluate C(z) = a·z³ + b·z² + c·z + d at scalar or array z.

    Parameters
    ----------
    coeffs : (n_t, 4)      columns [a, b, c, d]
    z      : scalar or (n_z,)

    Returns
    -------
    (n_t,) if z is scalar, (n_t, n_z) if z is array
    """
    a, b, c, d = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    if np.ndim(z) == 0:
        return a * z**3 + b * z**2 + c * z + d
    z = np.asarray(z)
    # outer broadcast: (n_t, 1) * (1, n_z)
    return (
        a[:, None] * z**3
        + b[:, None] * z**2
        + c[:, None] * z
        + d[:, None]
    )


def mean_cubic_coeffs(
    C1_series: np.ndarray,
    C2_series: np.ndarray,
    C3_series: np.ndarray,
    M_inv: np.ndarray,
) -> np.ndarray:
    """
    Fit the cubic to the time-mean concentrations.

    By linearity of M⁻¹ this is identical to averaging the per-timestep
    coefficient arrays, i.e.:
        mean_coeffs = M_inv @ [mean(C1), mean(C2), mean(C3), 0]ᵀ
    """
    rhs_mean = np.array([
        float(np.nanmean(C1_series)),
        float(np.nanmean(C2_series)),
        float(np.nanmean(C3_series)),
        0.0,
    ])
    return M_inv @ rhs_mean   # (4,)


# ══════════════════════════════════════════════════════════════════════════════
# ORACLE DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

def _collapse_to_slot_mean(frame: pd.DataFrame, freq: str = SLOT_FREQ) -> pd.Series:
    """Collapse raw dataframe rows to slot (15-min) averages."""
    return (
        frame.assign(
            localdatetime=pd.to_datetime(frame["localdatetime"]),
            datavalue=pd.to_numeric(frame["datavalue"], errors="coerce"),
        )
        .assign(slot_ts=lambda df: df["localdatetime"].dt.floor(freq))
        .groupby("slot_ts")["datavalue"]
        .mean()
        .sort_index()
    )


def _align_air_last_known(
    air_raw: pd.Series,
    target_index: pd.DatetimeIndex,
    freshness: pd.Timedelta = AIR_FRESHNESS,
) -> pd.Series:
    """
    Align irregular air observations to a regular target index using the
    'last known value within freshness window' rule (same logic as notebooks).
    """
    air_frame = (
        air_raw.sort_index()
        .rename("datavalue")
        .rename_axis("air_ts")
        .reset_index()
    )
    target = pd.DataFrame({"ts": target_index})
    aligned = pd.merge_asof(
        target,
        air_frame,
        left_on="ts",
        right_on="air_ts",
        direction="backward",
        tolerance=freshness,
    )
    return aligned.set_index("ts")["datavalue"]


def fetch_ideal_period_data(period: dict) -> dict | None:
    """
    Fetch and align all sensor data for one ideal-period vertical.

    Returns a dict with keys:
        index       : pd.DatetimeIndex  (15-min, ideal period)
        C1          : np.ndarray  (n_t,)  — 5 cm
        C2          : np.ndarray  (n_t,)  — 20 cm
        C3          : np.ndarray  (n_t,)  — deep sensor
        C_air       : np.ndarray  (n_t,)  — surface air (freshness-limited)
        depth_deep_m: float
    Or None if fetch fails.
    """
    start = pd.Timestamp(period["ideal_start"])
    end   = pd.Timestamp(period["ideal_end"])
    label = period["label"]

    print(f"  Fetching {label}  [{start} → {end}]")

    ideal_index = pd.date_range(start, end, freq=SLOT_FREQ)

    try:
        # ── Basalt sensors ─────────────────────────────────────────────────
        def _fetch_basalt(sid: int, depth_cm: float) -> np.ndarray:
            sensor = BasaltCO2Series(
                table_name="leo_west.datavalues",
                sensor_id=sid,
                variable_id=9,
                slope="LEO West",
                x_coord_m=float(period["basalt_x"]),
                y_coord_m=float(period["basalt_y"]),
                depth_cm=depth_cm,
            )
            frame = sensor.fetch_dataframe(
                start_datetime=start, end_datetime=end
            )
            series = _collapse_to_slot_mean(frame).reindex(ideal_index)
            series = series.mask(series <= 0)
            return series.to_numpy(dtype=float)

        C1 = _fetch_basalt(period["sensor_5cm"],   5.0)
        C2 = _fetch_basalt(period["sensor_20cm"], 20.0)
        C3 = _fetch_basalt(period["sensor_deep"],
                           period["depth_deep_m"] * 100)

        # ── Air sensor ─────────────────────────────────────────────────────
        air = AirCO2Series(
            table_name="leo_west.datavalueslicor",
            sensor_id=period["air_sensor"],
            variable_id=56,
            slope="LEO West",
            x_coord_m=float(period["air_x"]),
            y_coord_m=float(period["air_y"]),
            height_m=0.25,
            sensor_code=period["air_code"],
        )
        # fetch a window wide enough to handle freshness at the start
        air_raw = air.fetch_series(
            start_datetime=start - AIR_FRESHNESS * 2,
            end_datetime=end,
        )
        air_raw = air_raw.mask(air_raw <= 0)
        C_air = _align_air_last_known(air_raw, ideal_index).to_numpy(dtype=float)

        print(f"      n_t={len(ideal_index)}, "
              f"valid_5cm={int(np.sum(~np.isnan(C1)))}, "
              f"valid_20cm={int(np.sum(~np.isnan(C2)))}, "
              f"valid_deep={int(np.sum(~np.isnan(C3)))}, "
              f"valid_air={int(np.sum(~np.isnan(C_air)))}")

        return {
            "index":        ideal_index,
            "C1":           C1,
            "C2":           C2,
            "C3":           C3,
            "C_air":        C_air,
            "depth_deep_m": period["depth_deep_m"],
        }

    except Exception as exc:
        print(f"  [ERROR] {label}: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MEAN DEPTH PROFILES
# ══════════════════════════════════════════════════════════════════════════════

def plot_mean_profiles(
    results: list[dict | None],
    periods: list[dict],
    out_path: Path,
) -> None:
    """
    All 8 time-averaged cubic profiles on a single depth-vs-concentration axis.

    Each vertical contributes:
      • A smooth cubic curve from z = 0 to z = L
      • Three scatter markers at the mean observed concentrations
      • A marker at the model-inferred surface concentration (z = 0)
    """
    n_periods = sum(r is not None for r in results)
    colours = cm.tab10(np.linspace(0, 0.9, len(periods)))
    z_fine   = np.linspace(0, DOMAIN_L, 400)

    fig, ax = plt.subplots(figsize=(9, 10))
    fig.suptitle(
        "Mean CO₂ Depth Profiles — Cubic Polynomial Fit\n"
        "All 8 Ideal Periods (quasi-steady-state)",
        fontsize=13, fontweight="bold",
    )

    for i, (period, result) in enumerate(zip(periods, results)):
        if result is None:
            continue

        col = colours[i]
        zd  = result["depth_deep_m"]
        z1, z2, z3 = 0.05, 0.20, zd

        M     = build_geometry_matrix(z1, z2, z3)
        M_inv = invert_geometry_matrix(M)
        mc    = mean_cubic_coeffs(result["C1"], result["C2"], result["C3"], M_inv)

        a, b, c, d = mc
        C_profile = a * z_fine**3 + b * z_fine**2 + c * z_fine + d

        # Smooth profile
        ax.plot(C_profile, z_fine, color=col, linewidth=1.8,
                label=period["label"])

        # Observed mean markers
        mean_C1 = float(np.nanmean(result["C1"]))
        mean_C2 = float(np.nanmean(result["C2"]))
        mean_C3 = float(np.nanmean(result["C3"]))
        ax.scatter([mean_C1, mean_C2, mean_C3],
                   [z1, z2, z3],
                   color=col, s=70, zorder=5, edgecolors="black", linewidths=0.5)

        # Model-inferred surface concentration
        ax.scatter([d], [0.0], color=col, s=90, marker="^", zorder=6,
                   edgecolors="black", linewidths=0.7)

        # Mean air measurement (validation dot on zero-depth line)
        mean_air = float(np.nanmean(result["C_air"]))
        if not np.isnan(mean_air):
            ax.scatter([mean_air], [0.0], color=col, s=90, marker="x",
                       linewidths=2.0, zorder=7)

    ax.set_xlabel("CO₂ concentration [ppm]", fontsize=12)
    ax.set_ylabel("Depth z [m]", fontsize=12)
    ax.invert_yaxis()
    ax.set_ylim(DOMAIN_L * 1.05, -0.05)
    ax.set_xlim(left=0)
    ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    ax.grid(linestyle="--", alpha=0.55)
    ax.legend(fontsize=9, loc="lower right", title="Vertical (basalt x, y)")

    # Legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], linestyle="-",  linewidth=2, color="k", label="Cubic profile fit"),
        Line2D([0], [0], linestyle="",   marker="o", markersize=7,
               markerfacecolor="k", markeredgecolor="k", label="Mean observed (basalt sensor)"),
        Line2D([0], [0], linestyle="",   marker="^", markersize=8,
               markerfacecolor="k", markeredgecolor="k",
               label="Inferred surface C_s = d(t) [z=0, model]"),
        Line2D([0], [0], linestyle="",   marker="x", markersize=8,
               markeredgecolor="k", markeredgewidth=2,
               label="Mean air CO₂ [z=0, measured]"),
    ]
    ax.legend(handles=legend_elements + [
        plt.Line2D([0], [0], color=colours[i], linewidth=2,
                   label=periods[i]["label"])
        for i, r in enumerate(results) if r is not None
    ], fontsize=8, loc="lower right", ncol=1, title="Legend")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — PER-PERIOD TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def plot_time_series(
    results: list[dict | None],
    periods: list[dict],
    out_path: Path,
) -> None:
    """
    4 × 2 subplot grid — one panel per ideal-period vertical.

    Each panel shows:
      • Scatter  — observed CO₂ at 5 cm, 20 cm, deep sensor
      • Dashed   — cubic-inferred surface concentration d(t) = C(0, t)
      • Dotted   — measured air CO₂ (validation of extrapolation to z = 0)
    """
    n_cols, n_rows = 2, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 18),
                             sharex=False, sharey=False)
    fig.suptitle(
        "CO₂ Time Series with Cubic Profile Projection per Ideal Period\n"
        "Scatter: observed   Dashed: inferred surface C_s = d(t)   Dotted: measured air",
        fontsize=12, fontweight="bold",
    )

    depth_colours = ["#2E5EAA", "#E87722", "#44AA44"]

    for idx, (period, result) in enumerate(zip(periods, results)):
        row, col_idx = divmod(idx, n_cols)
        ax = axes[row, col_idx]

        label = period["label"]
        start = pd.Timestamp(period["ideal_start"])
        end   = pd.Timestamp(period["ideal_end"])

        if result is None:
            ax.set_title(f"{label}\n(data unavailable)", fontsize=9)
            ax.text(0.5, 0.5, "Oracle fetch failed",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            continue

        z1, z2, z3 = 0.05, 0.20, result["depth_deep_m"]
        M     = build_geometry_matrix(z1, z2, z3)
        M_inv = invert_geometry_matrix(M)

        # Build per-timestep mask: only fit where ALL three sensors are valid
        valid = (~np.isnan(result["C1"])) & (~np.isnan(result["C2"])) & (~np.isnan(result["C3"]))
        dt_index = result["index"]
        time_h   = (dt_index - dt_index[0]).total_seconds() / 3600.0

        depth_label_deep = f"{int(z3 * 100)} cm"
        depth_labels = ["5 cm", "20 cm", depth_label_deep]

        for i, (C_vals, depth_lbl, dcol) in enumerate(
            zip([result["C1"], result["C2"], result["C3"]], depth_labels, depth_colours)
        ):
            ax.scatter(time_h, C_vals, s=4, alpha=0.55, color=dcol,
                       label=f"Observed {depth_lbl}", zorder=3)

        # Cubic fit where valid
        if valid.sum() > 0:
            C1v = result["C1"].copy(); C1v[~valid] = np.nan
            C2v = result["C2"].copy(); C2v[~valid] = np.nan
            C3v = result["C3"].copy(); C3v[~valid] = np.nan
            C1v_f = result["C1"][valid]
            C2v_f = result["C2"][valid]
            C3v_f = result["C3"][valid]
            t_valid = time_h[valid]

            coeffs = fit_cubic_timeseries(C1v_f, C2v_f, C3v_f, M_inv)
            d_surf = coeffs[:, 3]   # d(t) = C(z=0, t)

            ax.plot(t_valid, d_surf, color="black", linewidth=1.4,
                    linestyle="--", zorder=5, label="Inferred C_s = d(t) [z = 0]")

        # Measured air (validation)
        air_vals = result["C_air"]
        ax.plot(time_h, air_vals, color="gray", linewidth=0.9,
                linestyle=":", zorder=4, label="Measured air CO₂")

        ax.set_title(
            f"{label}\n{start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}",
            fontsize=9,
        )
        ax.set_xlabel("Time [hours]", fontsize=8)
        ax.set_ylabel("CO₂ [ppm]", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(linestyle="--", alpha=0.45, linewidth=0.6)
        ax.tick_params(labelsize=7)

        if row == 0 and col_idx == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    out_dir = Path(__file__).resolve().parent / "out" / "cubic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _ORACLE_AVAILABLE:
        print(f"[ERROR] Cannot import sensorDB module: {_IMPORT_ERROR}")
        print("  Ensure Oracle Instant Client is installed and project .env is present.")
        return

    print("=== CUBIC PROFILE PROJECTION ===")
    print(f"  Domain L = {DOMAIN_L} m,  slot freq = {SLOT_FREQ}")
    print(f"  {len(IDEAL_PERIODS)} ideal-period verticals to process\n")

    # ── Print geometry matrix for each unique sensor depth configuration ──────
    for z3 in (0.35, 0.50):
        M = build_geometry_matrix(0.05, 0.20, z3)
        M_inv = invert_geometry_matrix(M)
        cond  = np.linalg.cond(M)
        print(f"  Geometry matrix  (deep sensor at {z3} m):")
        print(f"    z = [0.05, 0.20, {z3}] m")
        print(f"    κ(M) = {cond:.4f}  (well-conditioned if < 1e6)")
        print()

    # ── Fetch all data ─────────────────────────────────────────────────────────
    print("Fetching Oracle data for all 8 ideal periods...")
    print("-" * 60)
    results: list[dict | None] = []
    for period in IDEAL_PERIODS:
        results.append(fetch_ideal_period_data(period))
    print("-" * 60)
    n_ok = sum(r is not None for r in results)
    print(f"\nSuccessfully fetched: {n_ok} / {len(IDEAL_PERIODS)} verticals\n")

    if n_ok == 0:
        print("[WARN] No data fetched.  Check Oracle connection and credentials.")
        return

    # ── Figure 1: mean depth profiles ─────────────────────────────────────────
    print("Generating Figure 1: mean depth profiles...")
    plot_mean_profiles(results, IDEAL_PERIODS, out_dir / "cubic_mean_profiles.png")

    # ── Figure 2: per-period time series ──────────────────────────────────────
    print("Generating Figure 2: per-period time series...")
    plot_time_series(results, IDEAL_PERIODS, out_dir / "cubic_timeseries.png")

    print("\nDone.  Both plots saved in calculations/out/cubic/")


if __name__ == "__main__":
    run()

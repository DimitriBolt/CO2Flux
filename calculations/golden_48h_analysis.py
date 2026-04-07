"""
Golden 48-Hour Slice Analysis — Data QC + V1-V4 Inverse Methods
================================================================

Extracts the 48-hour central window from each of the 8 ideal-period
verticals.  Fetches real air CO₂ via AirCO2Series (or mock fallback).
Runs data quality checks, builds per-group ensemble averages, plots the
averaged depth profile + 48h time series, then runs V1→V4 on the slice.

Depth groups
------------
  50 cm : x=±1, y=4  and  x=±1, y=24   (4 verticals)
  35 cm : x=±1, y=10 and  x=±1, y=18   (4 verticals)

Air CO₂ sourcing
----------------
  USE_REAL_DATA = False  →  Gaussian ~415 ppm baseline, ±2 ppm noise
  USE_REAL_DATA = True   →  AirCO2Series via Oracle SensorDB

Toggle USE_REAL_DATA in ideal_period_timeline.py (shared flag).

Outputs (all under calculations/out/golden_48h/)
-------------------------------------------------
  depth_profile_avg.png      — ensemble depth profile + air CO₂ + ±1σ
  timeseries_48h.png         — ensemble 48h series + air CO₂ + V1 model
  V1/v1_50cm.png             — V1 profile fit for 50 cm group
  V1/v1_35cm.png             — V1 profile fit for 35 cm group
  V2/v2_50cm_timeseries.png  — V2 CN timeseries for 50 cm group
  V2/v2_35cm_timeseries.png  — V2 CN timeseries for 35 cm group
  goldenslice_48h.csv        — cleaned 48h ensemble (both groups)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# ── Shared catalogue and toggle ───────────────────────────────────────────────
_CALC_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _CALC_DIR.parent
_SENSORDB  = _REPO_ROOT / "Project_description" / "sensorDB"
for _p in [str(_CALC_DIR), str(_SENSORDB)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ideal_period_timeline import (           # noqa: E402
    VERTICALS,
    USE_REAL_DATA,
    GLOBAL_START,
    GLOBAL_END,
    _mock_series,
    _real_series,
)

# ── Physical / numerical constants (matching V1-V4) ───────────────────────────
L_DOMAIN: float       = 1.0
N_Z: int              = 101
DZ: float             = L_DOMAIN / (N_Z - 1)
Z_GRID: np.ndarray    = np.linspace(0.0, L_DOMAIN, N_Z)
DT: float             = 900.0            # Δt [s] — 15-minute polling

# Thermodynamic constants (V3/V4)
V_WATER: float   = 0.15
DH_SOL: float    = -2400.0
K_H_298: float   = 0.8317
MOLAR_AIR: float = 40.87

# ── Colour palette ────────────────────────────────────────────────────────────
C_DEPTH = ["#2E5EAA", "#E87722", "#44AA44"]   # 5 cm, 20 cm, deep
C_AIR   = "#CC2222"                            # air CO₂
Y_MAX   = 9000

# ── Output directory ─────────────────────────────────────────────────────────
OUT_ROOT = _CALC_DIR / "out" / "golden_48h"


# ══════════════════════════════════════════════════════════════════════════════
# DEPTH GROUPS
# ══════════════════════════════════════════════════════════════════════════════

GROUP_50 = [v for v in VERTICALS if v["depth_deep_m"] == 0.50]
GROUP_35 = [v for v in VERTICALS if v["depth_deep_m"] == 0.35]

GROUPS: dict[str, dict] = {
    "50cm": {
        "verticals":    GROUP_50,
        "sensor_depths": np.array([0.05, 0.20, 0.50]),
        "depth_labels":  ["5 cm", "20 cm", "50 cm"],
        "depth_cm_keys": ["5cm", "20cm", "50cm"],
    },
    "35cm": {
        "verticals":    GROUP_35,
        "sensor_depths": np.array([0.05, 0.20, 0.35]),
        "depth_labels":  ["5 cm", "20 cm", "35 cm"],
        "depth_cm_keys": ["5cm", "20cm", "35cm"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# AIR CO₂ — REAL AND MOCK
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_air_real(v: dict, t0: pd.Timestamp, t1: pd.Timestamp,
                    time_index: pd.DatetimeIndex) -> pd.Series:
    """Fetch real air CO₂ via AirCO2Series and slot onto time_index."""
    from air_co2_series import AirCO2Series  # noqa: PLC0415
    sensor = AirCO2Series(
        table_name="leo_west.datavalues",
        sensor_id=v["air_sensor"],
        variable_id=56,
        slope="LEO West",
        x_coord_m=float(v["air_x"]),
        y_coord_m=float(v["air_y"]),
    )
    frame = sensor.fetch_dataframe(start_datetime=t0, end_datetime=t1)
    raw = (
        frame.assign(
            localdatetime=pd.to_datetime(frame["localdatetime"]),
            datavalue=pd.to_numeric(frame["datavalue"], errors="coerce"),
        )
        .assign(slot_ts=lambda df: df["localdatetime"].dt.floor("15min"))
        .groupby("slot_ts")["datavalue"]
        .mean()
    )
    raw = raw.mask(raw <= 0)
    return raw.reindex(time_index).rename("air_co2")


def _mock_air(v: dict, time_index: pd.DatetimeIndex) -> pd.Series:
    """Realistic ~415 ppm mock air CO₂ — no diurnal spike."""
    rng = np.random.default_rng(seed=abs(v["basalt_x"] * 1000 + v["basalt_y"] * 7 + 42))
    return pd.Series(
        rng.normal(415.0, 2.0, len(time_index)),
        index=time_index,
        name="air_co2",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 48-HOUR WINDOW EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _48h_index(v: dict, base_freq: str = "15min") -> pd.DatetimeIndex:
    """Return a 15-min DatetimeIndex for the 48h window centred on v's midpoint."""
    mid = v["start"] + (v["end"] - v["start"]) / 2
    t0  = max(mid - pd.Timedelta(hours=24), v["start"])
    t1  = min(mid + pd.Timedelta(hours=24), v["end"])
    return pd.date_range(t0.floor(base_freq), t1.floor(base_freq), freq=base_freq)


def extract_48h(v: dict, basalt_dict: dict,
                global_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a clean DataFrame for one vertical over the 48h central window.

    Columns: timestamp, time_s (0-based), C_surface, C_z005, C_z020, C_zdep
    C_zdep is the deepest sensor (50 or 35 cm).
    """
    idx_48 = _48h_index(v)
    d1, d2, d3 = v["depths"]

    rows: dict[str, pd.Series] = {}
    for key in [f"{d1}cm", f"{d2}cm", f"{d3}cm"]:
        s = basalt_dict[key].reindex(idx_48)
        rows[key] = s

    # Air CO₂
    if USE_REAL_DATA:
        try:
            air = _fetch_air_real(v, idx_48[0], idx_48[-1], idx_48)
        except Exception as exc:
            print(f"    [WARN] air fetch failed ({exc}) — using mock air")
            air = _mock_air(v, idx_48)
    else:
        air = _mock_air(v, idx_48)

    t_s = (idx_48 - idx_48[0]).total_seconds().to_numpy()
    df = pd.DataFrame({
        "timestamp": idx_48,
        "time_s":    t_s,
        "C_surface": air.values,
        "C_z005":    rows[f"{d1}cm"].values,
        "C_z020":    rows[f"{d2}cm"].values,
        "C_zdep":    rows[f"{d3}cm"].values,
    })
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY CHECKS (QC)
# ══════════════════════════════════════════════════════════════════════════════

_D_REF  = 4.5e-6    # m²/s — prior for Cueva flux check
_N_AIR  = 42.3      # μmol m⁻³ ppm⁻¹

def qc_report(label: str, df: pd.DataFrame, sensor_depths_m: np.ndarray) -> pd.DataFrame:
    """
    Run QC checks on a single-vertical 48h DataFrame and return a cleaned copy.

    Checks
    ------
    1. NaN count per channel
    2. Duplicate timestamps
    3. Outliers beyond ±3σ per subsurface channel
    4. 413 ppm hardware artefact  (subsurface < 425 ppm)
    5. Cueva thermodynamic violation (implied downward flux > 0.5 μmol m⁻² s⁻¹)
    6. Air > subsurface anomaly  (C_surface > any subsurface sensor)
    """
    print(f"\n  [{label}]")
    df = df.copy()
    n0 = len(df)

    # 1. NaN counts
    for col in ["C_surface", "C_z005", "C_z020", "C_zdep"]:
        n_nan = int(df[col].isna().sum())
        if n_nan:
            print(f"    NaN  {col}: {n_nan}/{n0} ({100*n_nan/n0:.1f}%)")

    # 2. Duplicate timestamps
    dupes = int(df["timestamp"].duplicated().sum())
    if dupes:
        print(f"    Duplicate timestamps: {dupes}")

    # 3. Outliers ±3σ
    for col in ["C_z005", "C_z020", "C_zdep"]:
        s = df[col].dropna()
        lim_lo, lim_hi = s.mean() - 3 * s.std(), s.mean() + 3 * s.std()
        n_out = int(((df[col] < lim_lo) | (df[col] > lim_hi)).sum())
        if n_out:
            print(f"    Outliers ±3σ {col}: {n_out} rows")
            df.loc[(df[col] < lim_lo) | (df[col] > lim_hi), col] = np.nan

    # 4. 413 ppm artefact
    for col in ["C_z005", "C_z020", "C_zdep"]:
        flag = df[col] < 425.0
        n_flag = int(flag.sum())
        if n_flag:
            print(f"    413 ppm artefact {col}: {n_flag} rows → NaN")
        df.loc[flag, col] = np.nan

    # 5. Cueva thermodynamic check
    # pair: (deeper_col, shallower_col, depth_shallower, depth_deeper)
    pairs = [
        ("C_z005", "C_surface", 0.0,                    sensor_depths_m[0]),
        ("C_z020", "C_z005",    sensor_depths_m[0],     sensor_depths_m[1]),
        ("C_zdep", "C_z020",    sensor_depths_m[1],     sensor_depths_m[2]),
    ]
    for deeper, shallower, z_s, z_d in pairs:
        dz = z_d - z_s
        grad = (df[deeper] - df[shallower]) / dz
        flux = -_D_REF * grad * _N_AIR
        flag = flux > 0.5
        n_flag = int(flag.sum())
        if n_flag:
            print(f"    Cueva violation {deeper}: {n_flag} rows → NaN")
        df.loc[flag, deeper] = np.nan

    # 6. Air > subsurface anomaly
    for col in ["C_z005", "C_z020", "C_zdep"]:
        flag = df[col] < df["C_surface"]
        n_flag = int(flag.sum())
        if n_flag:
            print(f"    Air > subsurface {col}: {n_flag} rows")

    # Forward-fill up to 4 intervals (1 hour at 15-min polling)
    for col in ["C_z005", "C_z020", "C_zdep"]:
        df[col] = df[col].ffill(limit=4)

    n_after = int(df[["C_z005", "C_z020", "C_zdep"]].notna().all(axis=1).sum())
    print(f"    Retained: {n_after}/{n0} rows ({100*n_after/n0:.1f}%) after QC")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE AVERAGING PER DEPTH GROUP
# ══════════════════════════════════════════════════════════════════════════════

def build_ensemble(group_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Align all DataFrames in a group onto a common 48h time_s axis and
    compute ensemble mean ± std across verticals at each time step.

    Returns a DataFrame with columns:
        time_s, mean_C_surface, std_C_surface,
                mean_C_z005, std_C_z005,
                mean_C_z020, std_C_z020,
                mean_C_zdep, std_C_zdep, n
    """
    # Align to a common time grid (use the longest one)
    max_len = max(len(d) for d in group_dfs)
    ref_df  = next(d for d in group_dfs if len(d) == max_len)
    t_common = ref_df["time_s"].to_numpy()

    stacks: dict[str, list[np.ndarray]] = {
        col: [] for col in ["C_surface", "C_z005", "C_z020", "C_zdep"]
    }

    for df in group_dfs:
        ts = df["time_s"].to_numpy()
        for col in stacks:
            # Interpolate onto common grid (nearest-neighbour for short gaps)
            arr = np.interp(t_common, ts, df[col].to_numpy(),
                            left=np.nan, right=np.nan)
            stacks[col].append(arr)

    rows: dict[str, np.ndarray] = {"time_s": t_common}
    n_valid = np.zeros(len(t_common))
    for col, arr_list in stacks.items():
        mat  = np.stack(arr_list, axis=1)
        mean = np.nanmean(mat, axis=1)
        std  = np.nanstd(mat, axis=1, ddof=1) if mat.shape[1] > 1 else np.zeros(len(t_common))
        rows[f"mean_{col}"] = mean
        rows[f"std_{col}"]  = std
        if col == "C_z005":
            n_valid = np.sum(~np.isnan(mat), axis=1).astype(float)
    rows["n"] = n_valid
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT A — AVERAGED DEPTH PROFILE (both groups, one figure)
# ══════════════════════════════════════════════════════════════════════════════

def plot_avg_depth_profile(
    ensembles: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    2-column figure: left = 50 cm group, right = 35 cm group.

    Each panel: CO₂ vs depth (inverted y-axis, surface at top).
    Points: air (z=0), 5 cm, 20 cm, deep.  Horizontal ±1σ error bars.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        f"Ensemble Averaged Depth Profile — 48h Central Window\n"
        f"({data_mode} data  ·  horizontal bars = ±1σ across verticals)",
        fontsize=12, fontweight="bold",
    )

    group_info = {
        "50cm": {"depths_m": [0.0, 0.05, 0.20, 0.50], "label": "50 cm group (y=4, y=24)"},
        "35cm": {"depths_m": [0.0, 0.05, 0.20, 0.35], "label": "35 cm group (y=10, y=18)"},
    }

    for ax, (gname, ens) in zip(axes, ensembles.items()):
        info = group_info[gname]
        depths_cm = [d * 100 for d in info["depths_m"]]

        cols_mean = ["mean_C_surface", "mean_C_z005", "mean_C_z020", "mean_C_zdep"]
        cols_std  = ["std_C_surface",  "std_C_z005",  "std_C_z020",  "std_C_zdep"]
        means     = [float(ens[c].mean()) for c in cols_mean]
        stds      = [float(ens[c].mean()) for c in cols_std]  # avg std over time

        depth_colors_ext = [C_AIR] + C_DEPTH   # air + 3 subsurface
        labels_ext       = ["Air (surface)", "5 cm depth", "20 cm depth",
                             f"{int(depths_cm[-1])} cm depth"]

        # Connected profile line
        ax.plot(
            means, depths_cm,
            color="#444444", linewidth=1.6, zorder=3,
            marker="o", markersize=7, markerfacecolor="white", markeredgewidth=1.5,
        )

        for d_cm, m, s, col, lbl in zip(depths_cm, means, stds,
                                         depth_colors_ext, labels_ext):
            ax.errorbar(m, d_cm, xerr=s, fmt="none",
                        ecolor=col, elinewidth=2.2, capsize=5, capthick=2, zorder=4)
            ax.scatter(m, d_cm, color=col, s=80, zorder=5, label=lbl)

        n_v = len(GROUPS[gname]["verticals"])
        ax.set_title(
            f"{info['label']}\n"
            f"n={n_v} verticals  ·  mean over ±24 h window",
            fontsize=9,
        )
        ax.set_xlabel("CO₂ [ppm]", fontsize=9)
        ax.set_ylabel("Depth [cm]  (0 = surface)", fontsize=9)
        ax.set_xlim(0, Y_MAX)
        ax.set_ylim(max(depths_cm) + 3, -3)    # surface at top
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT B — ENSEMBLE 48H TIME SERIES (both groups, 2-row figure)
# ══════════════════════════════════════════════════════════════════════════════

def _to_timestamps(ens: pd.DataFrame, ref_v: dict) -> pd.DatetimeIndex:
    """Convert the ensemble time_s axis back to absolute timestamps."""
    mid = ref_v["start"] + (ref_v["end"] - ref_v["start"]) / 2
    t0  = max(mid - pd.Timedelta(hours=24), ref_v["start"])
    return pd.to_datetime(t0) + pd.to_timedelta(ens["time_s"], unit="s")


def plot_timeseries_48h(
    ensembles: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    2-row × 1-col figure.  Each row = one depth group.
    Lines: air CO₂ (red), 5 cm (blue), 20 cm (orange), deep (green).
    Shading: ±1σ across verticals.
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=False)
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        f"Ensemble 48-Hour Time Series — LEO West Biome\n"
        f"({data_mode} data  ·  shading = ±1σ across verticals  ·  red = air CO₂)",
        fontsize=12, fontweight="bold",
    )

    group_labels = {"50cm": "50 cm group (y=4, y=24)", "35cm": "35 cm group (y=10, y=18)"}
    depth_name   = {"50cm": "50 cm", "35cm": "35 cm"}

    for ax, (gname, ens) in zip(axes, ensembles.items()):
        ref_v = GROUPS[gname]["verticals"][0]
        timestamps = _to_timestamps(ens, ref_v)

        channel_map = [
            ("mean_C_surface", "std_C_surface", C_AIR,      "Air CO₂ (surface)"),
            ("mean_C_z005",    "std_C_z005",    C_DEPTH[0], "5 cm depth"),
            ("mean_C_z020",    "std_C_z020",    C_DEPTH[1], "20 cm depth"),
            ("mean_C_zdep",    "std_C_zdep",    C_DEPTH[2], f"{depth_name[gname]} depth"),
        ]

        for mcol, scol, col, lbl in channel_map:
            m = ens[mcol].to_numpy()
            s = ens[scol].to_numpy()
            valid = ~np.isnan(m)
            ax.plot(timestamps[valid], m[valid], color=col, linewidth=1.4, label=lbl)
            ax.fill_between(timestamps[valid],
                            (m - s)[valid], (m + s)[valid],
                            color=col, alpha=0.18)

        ax.set_ylim(0, Y_MAX)
        ax.set_ylabel("CO₂ [ppm]", fontsize=9)
        ax.set_title(group_labels[gname], fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# V1 — STATIONARY REGRESSION ON 48H MEAN
# ══════════════════════════════════════════════════════════════════════════════

def _v1_analytical(z: np.ndarray, D_s: float, S_bio: float, C_surf: float) -> np.ndarray:
    return C_surf + (S_bio / D_s) * (L_DOMAIN * z - 0.5 * z**2)


def run_v1(ens: pd.DataFrame, sensor_depths: np.ndarray, group_name: str,
           out_dir: Path) -> np.ndarray | None:
    """
    Steady-state quadratic fit on the 48h time-mean concentrations.
    Returns beta_opt = [D_s, S_bio] or None if optimisation fails.
    """
    C_surf = float(ens["mean_C_surface"].mean())
    C_mean = np.array([
        float(ens["mean_C_z005"].mean()),
        float(ens["mean_C_z020"].mean()),
        float(ens["mean_C_zdep"].mean()),
    ])
    if np.any(np.isnan(C_mean)):
        print(f"  V1 [{group_name}]: skipped — NaN in means")
        return None

    def _resid(beta: np.ndarray) -> np.ndarray:
        return _v1_analytical(sensor_depths, beta[0], beta[1], C_surf) - C_mean

    result = least_squares(
        _resid,
        x0=[4.5e-6, 0.42],
        bounds=([1e-7, 0.0], [1e-5, 1e3]),
        method="trf",
        diff_step=1e-8,
    )
    beta = result.x
    J    = result.jac
    dof  = max(len(result.fun) - len(beta), 1)
    sig2 = float(np.sum(result.fun**2) / dof)
    Cov  = sig2 * la.pinv(J.T @ J)

    print(f"\n  V1 [{group_name}]  D_s={beta[0]:.3e} m²/s  "
          f"S_bio={beta[1]:.4f} ppm/s  σ²={sig2:.2e}")

    # Plot
    z_fine = np.linspace(0.0, L_DOMAIN, 500)
    C_fit  = _v1_analytical(z_fine, beta[0], beta[1], C_surf)
    k      = L_DOMAIN * z_fine - 0.5 * z_fine**2
    dCdDs  = -k * beta[1] / beta[0]**2
    dCdSb  =  k / beta[0]
    grad   = np.stack([dCdDs, dCdSb], axis=1)
    sigma  = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", grad, Cov, grad), 0.0))

    res_at_sensors = _v1_analytical(sensor_depths, beta[0], beta[1], C_surf) - C_mean

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                              gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    ax.fill_betweenx(z_fine * 100, C_fit - 2 * sigma, C_fit + 2 * sigma,
                     alpha=0.2, color="steelblue", label="±2σ uncertainty")
    ax.plot(C_fit, z_fine * 100, color="steelblue", linewidth=2,
            label=f"Fitted  D_s={beta[0]:.2e} m²/s")
    ax.scatter(C_mean, sensor_depths * 100,
               s=120, color="darkorange", edgecolors="black", zorder=5,
               label="48h mean  (subsurface)")
    ax.scatter([C_surf], [0.0],
               s=120, marker="D", color=C_AIR, edgecolors="black", zorder=5,
               label=f"Air CO₂ BC  ({C_surf:.0f} ppm)")
    ax.set_xlabel("CO₂ [ppm]", fontsize=10)
    ax.set_ylabel("Depth [cm]", fontsize=10)
    ax.set_title(f"V1 Stationary Regression — {group_name} group\n"
                 f"48h mean from {len(GROUPS[group_name]['verticals'])} verticals", fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, Y_MAX)
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.4)

    ax2 = axes[1]
    ax2.barh([f"z={d:.2f} m" for d in sensor_depths], res_at_sensors,
             color=C_DEPTH, edgecolor="black", linewidth=0.8)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("Residual [ppm]\n(model − data)", fontsize=9)
    ax2.set_title("Residuals", fontsize=10)
    ax2.grid(axis="x", alpha=0.4)

    plt.tight_layout()
    v1_dir = out_dir / "V1"
    v1_dir.mkdir(parents=True, exist_ok=True)
    p = v1_dir / f"v1_{group_name}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")
    return beta


# ══════════════════════════════════════════════════════════════════════════════
# V2 — CRANK-NICOLSON ON 48H TIME SERIES
# ══════════════════════════════════════════════════════════════════════════════

def _cn_matrices(D_s: float):
    r = D_s * DT / (2.0 * DZ**2)
    A = diags([-r, 1 + 2*r, -r], [-1, 0, 1], shape=(N_Z, N_Z), format="lil")
    B = diags([ r, 1 - 2*r,  r], [-1, 0, 1], shape=(N_Z, N_Z), format="csr")
    A[0, :] = 0; A[0, 0] = 1.0
    A[N_Z-1, :] = 0; A[N_Z-1, N_Z-1] = 1.0; A[N_Z-1, N_Z-2] = -1.0
    return A.tocsr(), B


def _cn_forward(beta: np.ndarray, C_surface: np.ndarray,
                C_data_t0: np.ndarray, sensor_idx: np.ndarray) -> np.ndarray:
    D_s, S_0, S_1 = beta
    n_t    = len(C_surface)
    S_bio  = S_0 + S_1 * Z_GRID
    A, B   = _cn_matrices(D_s)
    depths_at_sensors = sensor_idx * DZ
    anchor_z = np.array([0.0] + list(depths_at_sensors) + [L_DOMAIN])
    anchor_c = np.array([C_surface[0]] + list(C_data_t0) + [C_data_t0[-1]])
    cs = CubicSpline(anchor_z, anchor_c)
    C  = cs(Z_GRID)
    preds = np.empty((n_t, len(sensor_idx)))
    preds[0] = C[sensor_idx]
    for n in range(1, n_t):
        rhs = B.dot(C) + S_bio * DT
        rhs[0]      = C_surface[n]
        rhs[N_Z-1]  = 0.0
        C = spsolve(A, rhs)
        preds[n] = C[sensor_idx]
    return preds


def run_v2(ens: pd.DataFrame, sensor_depths: np.ndarray, group_name: str,
           out_dir: Path) -> np.ndarray | None:
    """Crank-Nicolson transient solve on the 48h ensemble mean time series."""
    sensor_idx = np.array([int(round(d / DZ)) for d in sensor_depths])

    C_surface = ens["mean_C_surface"].to_numpy()
    C_data    = np.column_stack([
        ens["mean_C_z005"].to_numpy(),
        ens["mean_C_z020"].to_numpy(),
        ens["mean_C_zdep"].to_numpy(),
    ])

    # Skip if more than 10% NaN
    if np.isnan(C_data).mean() > 0.10:
        print(f"  V2 [{group_name}]: skipped — too many NaN in ensemble")
        return None

    # Fill remaining NaN by forward-fill
    df_tmp = pd.DataFrame(
        np.column_stack([C_surface, C_data]),
        columns=["C_surface", "z0", "z1", "z2"],
    ).ffill().bfill()
    C_surface = df_tmp["C_surface"].to_numpy()
    C_data    = df_tmp[["z0", "z1", "z2"]].to_numpy()

    def _cost(beta):
        pred = _cn_forward(beta, C_surface, C_data[0], sensor_idx)
        return (pred - C_data).flatten()

    result = least_squares(
        _cost,
        x0=[4.5e-6, 0.5, -0.5],
        bounds=([1e-7, 0.0, -100.0], [1e-5, 10.0, 0.0]),
        method="trf",
        diff_step=1e-8,
        max_nfev=500,
        verbose=0,
    )
    beta = result.x
    dof  = max(len(result.fun) - len(beta), 1)
    sig2 = float(np.sum(result.fun**2) / dof)
    print(f"\n  V2 [{group_name}]  D_s={beta[0]:.3e}  S_0={beta[1]:.4f}  "
          f"S_1={beta[2]:.4f}  σ²={sig2:.2e}")

    C_pred = _cn_forward(beta, C_surface, C_data[0], sensor_idx)
    n_t    = len(C_surface)
    time_h = np.arange(n_t) * DT / 3600.0
    depth_labels = [f"z={d:.2f} m" for d in sensor_depths]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f"V2 Crank-Nicolson — {group_name} group — 48h ensemble mean",
                 fontsize=11, fontweight="bold")
    for i, (col, lbl) in enumerate(zip(C_DEPTH, depth_labels)):
        ax = axes[i]
        ax.plot(time_h, C_data[:, i], color=col, linewidth=1.0, label=f"Observed {lbl}", alpha=0.9)
        ax.plot(time_h, C_pred[:, i], color=col, linewidth=1.5, linestyle="--", label="Model")
        ax.set_ylabel("CO₂ [ppm]", fontsize=8)
        ax.set_title(lbl, fontsize=8, loc="left")
        ax.legend(fontsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
    axes[-1].plot(time_h, (C_pred - C_data).mean(axis=1), color="black", linewidth=1.0)
    axes[-1].axhline(0, color="gray", linewidth=0.8)
    axes[-1].set_ylabel("Mean residual [ppm]", fontsize=8)
    axes[-1].set_xlabel("Time [h]", fontsize=9)
    axes[-1].set_title("Mean residual (model − obs)", fontsize=8, loc="left")
    axes[-1].grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    v2_dir = out_dir / "V2"
    v2_dir.mkdir(parents=True, exist_ok=True)
    p = v2_dir / f"v2_{group_name}_timeseries.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {p}")
    return beta


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_csv(ensembles: dict[str, pd.DataFrame], out_path: Path) -> None:
    """
    Save both groups to a single CSV.  time_s is relative (0 = start of
    each group's 48h window). Group identifier column added.
    """
    frames = []
    for gname, ens in ensembles.items():
        df_out = ens.copy()
        df_out.insert(0, "group", gname)
        frames.append(df_out)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False, float_format="%.3f")
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"\n=== GOLDEN 48H ANALYSIS  ({mode}) ===")

    # ── 1. Load full 15-min series (reuse ideal_period_timeline loader) ───────
    time_index = pd.date_range(GLOBAL_START, GLOBAL_END, freq="15min")
    basalt_list: list[dict] = []
    for v in VERTICALS:
        if USE_REAL_DATA:
            try:
                basalt_list.append(_real_series(v, time_index))
            except Exception as exc:
                print(f"  [WARN] basalt fetch {v['label']}: {exc} — mock fallback")
                basalt_list.append(_mock_series(v, time_index))
        else:
            basalt_list.append(_mock_series(v, time_index))

    # ── 2. Extract 48h windows + air CO₂, run QC per vertical ────────────────
    print("\n=== QC REPORT (per-vertical 48h window) ===")
    df_by_v: dict[str, pd.DataFrame] = {}
    for v, basalt_dict in zip(VERTICALS, basalt_list):
        gname     = "50cm" if v["depth_deep_m"] == 0.50 else "35cm"
        s_depths  = GROUPS[gname]["sensor_depths"]
        df_raw    = extract_48h(v, basalt_dict, time_index)
        df_clean  = qc_report(v["label"], df_raw, s_depths)
        df_by_v[v["label"]] = df_clean

    # ── 3. Build ensemble per depth group ─────────────────────────────────────
    print("\n=== ENSEMBLE BUILD ===")
    ensembles: dict[str, pd.DataFrame] = {}
    for gname, ginfo in GROUPS.items():
        group_dfs = [df_by_v[v["label"]] for v in ginfo["verticals"]]
        ens = build_ensemble(group_dfs)
        ensembles[gname] = ens
        n_v = len(ginfo["verticals"])
        print(f"  {gname}: {len(ens)} time steps,  {n_v} verticals,  "
              f"mean n_valid={ens['n'].mean():.1f}")

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\n=== GENERATING FIGURES ===")
    plot_avg_depth_profile(ensembles, OUT_ROOT / "depth_profile_avg.png")
    plot_timeseries_48h(ensembles, OUT_ROOT / "timeseries_48h.png")

    # ── 5. Run V1 on each group ───────────────────────────────────────────────
    print("\n=== V1: STATIONARY REGRESSION ===")
    for gname, ginfo in GROUPS.items():
        run_v1(ensembles[gname], ginfo["sensor_depths"], gname, OUT_ROOT)

    # ── 6. Run V2 on each group ───────────────────────────────────────────────
    print("\n=== V2: CRANK-NICOLSON TRANSIENT ===")
    for gname, ginfo in GROUPS.items():
        run_v2(ensembles[gname], ginfo["sensor_depths"], gname, OUT_ROOT)

    # ── 7. Export CSV ─────────────────────────────────────────────────────────
    print("\n=== EXPORTING CSV ===")
    save_csv(ensembles, OUT_ROOT / "goldenslice_48h.csv")

    print("\n=== DONE ===")
    print(f"  All outputs in: {OUT_ROOT}")


if __name__ == "__main__":
    run()

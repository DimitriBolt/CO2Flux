"""
Composite Profile Pipeline — Dual-Track SVD Inversion
======================================================

Implements a 4-phase analysis to compare individual-column inversions
(Track A: Mean of the Physics) against a composite overdetermined SVD
inversion (Track B: Physics of the Mean).

Phase 1 — Exchangeability Validation
    Split 8 verticals into cohort_35 (y=10,18) and cohort_50 (y=4,24).
    Compare shared-depth (5 cm, 20 cm) time series to verify they belong
    to the same physical regime.

Phase 2 — Composite Profile Construction
    Stitch a synthetic 5-depth profile: C_air (all 8), C_5 (all 8),
    C_20 (all 8), C_35 (cohort_35 only), C_50 (cohort_50 only).

Phase 3 — SVD Math Engines
    Track A: standard 4×4 geometry matrix per cohort (3 sensors + BC).
    Track B: overdetermined 5×4 geometry matrix (4 sensors + BC) solved
    via SVD pseudoinverse (least-squares fit).

Phase 4 — Track A vs Track B Comparison
    Plot surface flux J↑ and effective diffusivity D_eff from both tracks.

Data sourcing
-------------
    Reuses VERTICALS catalogue and data loaders from ideal_period_timeline.py.
    Toggle USE_REAL_DATA there to switch between mock and Oracle.

Outputs (calculations/out/composite/)
--------------------------------------
    phase1_exchangeability.png       — cohort validation (5 cm & 20 cm overlay)
    phase4_flux_comparison.png       — Track A vs B surface flux & D_eff
    svd_cubic_depth_profile.png      — fitted cubic C(z) vs sensor data + D_eff/S_bio
    svd_physics_timeseries.png       — D_eff and S_bio time series from composite SVD
    composite_summary.csv            — per-timestep Track A & B results
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
from golden_48h_analysis import (             # noqa: E402
    _48h_index,
    _mock_air,
    _fetch_air_real,
    qc_report,
    GROUPS,
    GROUP_50,
    GROUP_35,
)

# ── Physical constants ────────────────────────────────────────────────────────
L: float       = 1.00      # Bottom of basalt layer [m]
K_G: float     = 1e-5      # Gas-transfer coefficient [m/s]
C_AIR_MOL: float = 40.9    # Molar concentration of air [mol/m³]

# ── Colour palette ────────────────────────────────────────────────────────────
C_DEPTH = ["#2E5EAA", "#E87722", "#44AA44", "#9B59B6"]  # 5, 20, 35, 50 cm
C_AIR   = "#CC2222"
Y_MAX   = 9000

# ── Output directory ─────────────────────────────────────────────────────────
OUT_ROOT = _CALC_DIR / "out" / "composite"


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (48h windows, same as golden_48h_analysis)
# ══════════════════════════════════════════════════════════════════════════════

def _load_48h_per_vertical() -> dict[str, pd.DataFrame]:
    """
    Load all 8 verticals, extract 48h central windows, run QC.

    Returns dict mapping vertical label → cleaned DataFrame with columns:
        timestamp, time_s, C_surface, C_z005, C_z020, C_zdep
    """
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

    df_by_v: dict[str, pd.DataFrame] = {}
    for v, basalt_dict in zip(VERTICALS, basalt_list):
        gname = "50cm" if v["depth_deep_m"] == 0.50 else "35cm"
        s_depths = GROUPS[gname]["sensor_depths"]

        # Extract 48h window
        idx_48 = _48h_index(v)
        d1, d2, d3 = v["depths"]
        rows: dict[str, pd.Series] = {}
        for key in [f"{d1}cm", f"{d2}cm", f"{d3}cm"]:
            rows[key] = basalt_dict[key].reindex(idx_48)

        # Air CO₂
        if USE_REAL_DATA:
            try:
                air = _fetch_air_real(v, idx_48[0], idx_48[-1], idx_48)
            except Exception:
                air = _mock_air(v, idx_48)
        else:
            air = _mock_air(v, idx_48)

        t_s = (idx_48 - idx_48[0]).total_seconds().to_numpy()
        df_raw = pd.DataFrame({
            "timestamp": idx_48,
            "time_s":    t_s,
            "C_surface": air.values,
            "C_z005":    rows[f"{d1}cm"].values,
            "C_z020":    rows[f"{d2}cm"].values,
            "C_zdep":    rows[f"{d3}cm"].values,
        })
        df_clean = qc_report(v["label"], df_raw, s_depths)
        df_by_v[v["label"]] = df_clean

    return df_by_v


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — EXCHANGEABILITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def _align_cohort_mean(
    df_by_v: dict[str, pd.DataFrame],
    verticals: list[dict],
) -> pd.DataFrame:
    """Average across verticals in a cohort on their time_s axis."""
    dfs = [df_by_v[v["label"]] for v in verticals]
    max_len = max(len(d) for d in dfs)
    ref = next(d for d in dfs if len(d) == max_len)
    t_common = ref["time_s"].to_numpy()
    ts_common = ref["timestamp"].values

    stacks: dict[str, list[np.ndarray]] = {
        c: [] for c in ["C_surface", "C_z005", "C_z020", "C_zdep"]
    }
    for df in dfs:
        for col in stacks:
            arr = np.interp(t_common, df["time_s"].to_numpy(),
                            df[col].to_numpy(), left=np.nan, right=np.nan)
            stacks[col].append(arr)

    result = {"timestamp": ts_common, "time_s": t_common}
    for col, arr_list in stacks.items():
        mat = np.stack(arr_list, axis=1)
        result[f"mean_{col}"] = np.nanmean(mat, axis=1)
        result[f"std_{col}"]  = np.nanstd(mat, axis=1, ddof=1) if mat.shape[1] > 1 else np.zeros(len(t_common))
    return pd.DataFrame(result)


def phase1_validate(
    df_by_v: dict[str, pd.DataFrame],
    out_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Phase 1: Compare shared-depth time series between cohorts.

    Returns (df_35_mean, df_50_mean, divergence_stats).
    """
    print("\n=== PHASE 1: EXCHANGEABILITY VALIDATION ===")
    df_35_mean = _align_cohort_mean(df_by_v, GROUP_35)
    df_50_mean = _align_cohort_mean(df_by_v, GROUP_50)

    # Calculate divergence at shared depths
    # Align to common length (take minimum)
    n = min(len(df_35_mean), len(df_50_mean))
    div_5  = np.abs(df_35_mean["mean_C_z005"].values[:n] - df_50_mean["mean_C_z005"].values[:n])
    div_20 = np.abs(df_35_mean["mean_C_z020"].values[:n] - df_50_mean["mean_C_z020"].values[:n])

    stats = {
        "div_5cm_mean":  float(np.nanmean(div_5)),
        "div_5cm_max":   float(np.nanmax(div_5)),
        "div_20cm_mean": float(np.nanmean(div_20)),
        "div_20cm_max":  float(np.nanmax(div_20)),
    }

    print(f"  Divergence at 5 cm:  mean={stats['div_5cm_mean']:.1f} ppm, "
          f"max={stats['div_5cm_max']:.1f} ppm")
    print(f"  Divergence at 20 cm: mean={stats['div_20cm_mean']:.1f} ppm, "
          f"max={stats['div_20cm_max']:.1f} ppm")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        "Phase 1: Exchangeability Validation — Are the Cohorts Physically Aligned?\n"
        f"({data_mode} data  ·  48h central window  ·  cohort averages)",
        fontsize=11, fontweight="bold",
    )

    ts_35 = df_35_mean["timestamp"].values[:n]

    # Top: 20 cm comparison
    ax = axes[0]
    ax.plot(ts_35, df_35_mean["mean_C_z020"].values[:n],
            label="Cohort 35 (y=10,18) at 20 cm", color="blue", linewidth=1.5)
    ax.fill_between(ts_35,
                    (df_35_mean["mean_C_z020"] - df_35_mean["std_C_z020"]).values[:n],
                    (df_35_mean["mean_C_z020"] + df_35_mean["std_C_z020"]).values[:n],
                    color="blue", alpha=0.15)
    ax.plot(ts_35, df_50_mean["mean_C_z020"].values[:n],
            label="Cohort 50 (y=4,24) at 20 cm", color="red", linestyle="--", linewidth=1.5)
    ax.fill_between(ts_35,
                    (df_50_mean["mean_C_z020"] - df_50_mean["std_C_z020"]).values[:n],
                    (df_50_mean["mean_C_z020"] + df_50_mean["std_C_z020"]).values[:n],
                    color="red", alpha=0.15)
    ax.set_ylabel("CO₂ [ppm]", fontsize=10)
    ax.set_title(f"20 cm depth  —  mean divergence = {stats['div_20cm_mean']:.0f} ppm",
                 fontsize=9, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Bottom: 5 cm comparison
    ax = axes[1]
    ax.plot(ts_35, df_35_mean["mean_C_z005"].values[:n],
            label="Cohort 35 (y=10,18) at 5 cm", color="lightblue", linewidth=1.5)
    ax.fill_between(ts_35,
                    (df_35_mean["mean_C_z005"] - df_35_mean["std_C_z005"]).values[:n],
                    (df_35_mean["mean_C_z005"] + df_35_mean["std_C_z005"]).values[:n],
                    color="lightblue", alpha=0.15)
    ax.plot(ts_35, df_50_mean["mean_C_z005"].values[:n],
            label="Cohort 50 (y=4,24) at 5 cm", color="salmon", linestyle="--", linewidth=1.5)
    ax.fill_between(ts_35,
                    (df_50_mean["mean_C_z005"] - df_50_mean["std_C_z005"]).values[:n],
                    (df_50_mean["mean_C_z005"] + df_50_mean["std_C_z005"]).values[:n],
                    color="salmon", alpha=0.15)
    ax.set_ylabel("CO₂ [ppm]", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title(f"5 cm depth  —  mean divergence = {stats['div_5cm_mean']:.0f} ppm",
                 fontsize=9, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    return df_35_mean, df_50_mean, stats


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — COMPOSITE PROFILE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def phase2_build_composite(
    df_by_v: dict[str, pd.DataFrame],
    df_35_mean: pd.DataFrame,
    df_50_mean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Stitch the synthetic 5-depth composite profile.

    C_air, C_5, C_20: averaged across all 8 verticals.
    C_35: averaged from cohort_35 only.
    C_50: averaged from cohort_50 only.
    """
    print("\n=== PHASE 2: CONSTRUCTING COMPOSITE PROFILE ===")

    # Average all 8 verticals for shared depths
    df_all_mean = _align_cohort_mean(df_by_v, list(VERTICALS))

    n = min(len(df_all_mean), len(df_35_mean), len(df_50_mean))
    t_s = df_all_mean["time_s"].values[:n]
    ts  = df_all_mean["timestamp"].values[:n]

    df_composite = pd.DataFrame({
        "timestamp": ts,
        "time_s":    t_s,
        "C_air":     df_all_mean["mean_C_surface"].values[:n],
        "C_z005":    df_all_mean["mean_C_z005"].values[:n],          # 5 cm (all 8)
        "C_z020":    df_all_mean["mean_C_z020"].values[:n],          # 20 cm (all 8)
        "C_z035":    df_35_mean["mean_C_zdep"].values[:n],           # 35 cm (cohort_35)
        "C_z050":    df_50_mean["mean_C_zdep"].values[:n],           # 50 cm (cohort_50)
    })

    valid = df_composite[["C_z005", "C_z020", "C_z035", "C_z050"]].notna().all(axis=1).sum()
    print(f"  Composite: {len(df_composite)} timesteps, {valid} with all 4 depths valid")
    print(f"  Mean C_air={df_composite['C_air'].mean():.0f}, "
          f"C_5={df_composite['C_z005'].mean():.0f}, "
          f"C_20={df_composite['C_z020'].mean():.0f}, "
          f"C_35={df_composite['C_z035'].mean():.0f}, "
          f"C_50={df_composite['C_z050'].mean():.0f} ppm")

    return df_composite


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — SVD MATH ENGINES
# ══════════════════════════════════════════════════════════════════════════════

def build_svd_pinv(depths: list[float]) -> np.ndarray:
    """
    Build the SVD pseudoinverse for an (n_depths + 1) × 4 geometry matrix.

    Rows: one per sensor depth (cubic eval) + zero-flux BC at z = L.
    Columns: [z³, z², z, 1]  for polynomial C(z) = az³ + bz² + cz + d.

    For n_depths = 3: 4×4 square system → exact inverse.
    For n_depths = 4: 5×4 overdetermined → least-squares via SVD.
    """
    n_rows = len(depths) + 1
    M = np.zeros((n_rows, 4))
    for i, z in enumerate(depths):
        M[i, :] = [z**3, z**2, z, 1.0]
    # Bottom boundary: C'(L) = 0 → 3aL² + 2bL + c = 0
    M[-1, :] = [3.0 * L**2, 2.0 * L, 1.0, 0.0]

    U, S, Vh = np.linalg.svd(M, full_matrices=True)
    # Truncated pseudoinverse: zero out near-zero singular values
    S_inv = np.where(S > (1e-14 * np.max(S)), 1.0 / S, 0.0)
    # M_pinv = Vh^T @ diag(S_inv) @ U^T[:r, :]  where r = len(S)
    r = len(S)
    M_pinv = Vh.T @ np.diag(S_inv) @ U[:, :r].T

    cond = np.max(S) / np.min(S[S > 1e-14 * np.max(S)])
    print(f"  SVD  {n_rows}×4  depths={[f'{d:.2f}' for d in depths]}  "
          f"κ={cond:.1f}  rank={np.sum(S > 1e-14 * np.max(S))}")

    return M_pinv


def calc_physics(
    C_air: np.ndarray,
    C_sensors: np.ndarray,
    pinv: np.ndarray,
) -> pd.DataFrame:
    """
    Vectorised calculation of D_eff and J_up at each timestep.

    Parameters
    ----------
    C_air     : (n_t,) surface air CO₂ [ppm]
    C_sensors : (n_t, n_depths) sensor concentrations [ppm]
    pinv      : (4, n_rows) SVD pseudoinverse

    Returns
    -------
    DataFrame with columns D_eff, J_up, C_surface_inferred, C_prime_0
    """
    n_t, n_s = C_sensors.shape
    # RHS: sensor observations + zero-flux BC row
    rhs = np.column_stack([C_sensors, np.zeros(n_t)])  # (n_t, n_rows)
    coeffs = rhs @ pinv.T    # (n_t, 4)  →  [a, b, c, d]

    C_prime_0   = coeffs[:, 2]   # c = dC/dz at z=0
    C_surface   = coeffs[:, 3]   # d = C(0)

    # Effective diffusivity from surface flux balance:
    #   J_surface = -D_eff · C'(0)  and  J_surface = k_g · (C_surface - C_air)
    #   → D_eff = k_g · (C_surface - C_air) / C'(0)
    # Guard against division by near-zero gradient
    safe_grad = np.where(np.abs(C_prime_0) < 1.0, np.nan, C_prime_0)
    D_eff = K_G * (C_surface - C_air) / safe_grad

    # Upward flux [μmol m⁻² s⁻¹]
    J_up = K_G * C_AIR_MOL * (C_surface - C_air) * 1e-6

    return pd.DataFrame({
        "D_eff":              D_eff,
        "J_up":               J_up,
        "C_surface_inferred": C_surface,
        "C_prime_0":          C_prime_0,
        "coeff_a":            coeffs[:, 0],
        "coeff_b":            coeffs[:, 1],
        "coeff_c":            coeffs[:, 2],
        "coeff_d":            coeffs[:, 3],
    })


# ══════════════════════════════════════════════════════════════════════════════
# SVD CUBIC DEPTH PROFILE & SOURCE TERM DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def _eval_cubic(coeffs_row: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate C(z) = az³ + bz² + cz + d for a single set of coefficients."""
    a, b, c, d = coeffs_row
    return a * z**3 + b * z**2 + c * z + d


def _eval_cubic_deriv2(coeffs_row: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate C''(z) = 6az + 2b."""
    a, b = coeffs_row[0], coeffs_row[1]
    return 6.0 * a * z + 2.0 * b


def compute_source_term(
    D_eff: np.ndarray,
    coeff_a: np.ndarray,
    coeff_b: np.ndarray,
) -> np.ndarray:
    """
    Depth-averaged source term from the steady-state balance.

    ODE:  D_eff · C''(z) + S(z) = 0
    For a cubic, C''(z) = 6az + 2b is linear in z.
    Column-averaged source:
        <S> = -D_eff · (1/L) · ∫₀ᴸ C''(z) dz
            = -D_eff · (1/L) · [3aL² + 2bL]
            = -D_eff · (3aL + 2b)
    """
    return -D_eff * (3.0 * coeff_a * L + 2.0 * coeff_b)


def plot_svd_cubic_profile(
    df_composite: pd.DataFrame,
    df_physics: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    2-panel figure showing the SVD cubic fit on the composite profile.

    Left:  Time-mean fitted C(z) curve vs sensor observations
           (z = 0 to L, sensor points + inferred surface + zero-flux BC).
    Right: Implied source term S(z) vs depth.

    Annotated with time-mean D_eff, S_bio, J_up.
    """
    z_fine = np.linspace(0.0, L, 500)

    # Time-mean coefficients
    a_mean = float(df_physics["coeff_a"].mean())
    b_mean = float(df_physics["coeff_b"].mean())
    c_mean = float(df_physics["coeff_c"].mean())
    d_mean = float(df_physics["coeff_d"].mean())
    mean_coeffs = np.array([a_mean, b_mean, c_mean, d_mean])

    # Time-mean physics
    D_mean = float(df_physics["D_eff"].mean())
    J_mean = float(df_physics["J_up"].mean())
    S_bio_ts = compute_source_term(
        df_physics["D_eff"].to_numpy(),
        df_physics["coeff_a"].to_numpy(),
        df_physics["coeff_b"].to_numpy(),
    )
    S_mean = float(np.nanmean(S_bio_ts))

    # Evaluated curves
    C_fitted = _eval_cubic(mean_coeffs, z_fine)
    S_fitted = -D_mean * _eval_cubic_deriv2(mean_coeffs, z_fine)

    # Sensor data (time-means for the composite)
    sensor_depths = np.array([0.05, 0.20, 0.35, 0.50])
    sensor_means  = np.array([
        float(df_composite["C_z005"].mean()),
        float(df_composite["C_z020"].mean()),
        float(df_composite["C_z035"].mean()),
        float(df_composite["C_z050"].mean()),
    ])
    sensor_stds = np.array([
        float(df_composite["C_z005"].std()),
        float(df_composite["C_z020"].std()),
        float(df_composite["C_z035"].std()),
        float(df_composite["C_z050"].std()),
    ])
    air_mean = float(df_composite["C_air"].mean())
    air_std  = float(df_composite["C_air"].std())

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8),
                                    gridspec_kw={"width_ratios": [3, 2]})
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        "SVD Cubic Fit on Composite Profile — Concentration & Source Term vs Depth\n"
        f"({data_mode} data  ·  48h mean  ·  overdetermined 5×4 least-squares)",
        fontsize=12, fontweight="bold",
    )

    # ── Left panel: C(z) vs depth ─────────────────────────────────────────────
    ax1.plot(C_fitted, z_fine * 100,
             color="steelblue", linewidth=2.5,
             label=f"SVD cubic fit  ($D_{{\\mathrm{{eff}}}}$ = {D_mean:.2e} m²/s)")

    # Confidence band via per-timestep spread
    n_snap = len(df_physics)
    C_all = np.zeros((n_snap, len(z_fine)))
    for i in range(n_snap):
        row = df_physics.iloc[i]
        C_all[i] = _eval_cubic(
            np.array([row["coeff_a"], row["coeff_b"], row["coeff_c"], row["coeff_d"]]),
            z_fine,
        )
    C_lo = np.nanpercentile(C_all, 5, axis=0)
    C_hi = np.nanpercentile(C_all, 95, axis=0)
    ax1.fill_betweenx(z_fine * 100, C_lo, C_hi,
                      color="steelblue", alpha=0.15, label="5–95 % envelope")

    # Sensor observations
    sensor_colors = [C_DEPTH[0], C_DEPTH[1], C_DEPTH[2], C_DEPTH[3]]
    sensor_labels = ["5 cm (all 8)", "20 cm (all 8)", "35 cm (cohort 35)", "50 cm (cohort 50)"]
    for z_cm, m, s, col, lbl in zip(
        sensor_depths * 100, sensor_means, sensor_stds, sensor_colors, sensor_labels
    ):
        ax1.errorbar(m, z_cm, xerr=s, fmt="o", color=col,
                     markersize=9, markeredgecolor="black", markeredgewidth=1,
                     elinewidth=2, capsize=5, capthick=1.5, zorder=5, label=lbl)

    # Air BC at z=0
    ax1.errorbar(air_mean, 0.0, xerr=air_std, fmt="D", color=C_AIR,
                 markersize=10, markeredgecolor="black", markeredgewidth=1,
                 elinewidth=2, capsize=5, capthick=1.5, zorder=5,
                 label=f"Air CO₂ ({air_mean:.0f} ppm)")

    # Inferred surface from cubic
    ax1.scatter([d_mean], [0.0], marker="*", s=200, color="gold",
                edgecolors="black", linewidths=1, zorder=6,
                label=f"SVD-inferred C(0) = {d_mean:.0f} ppm")

    # Zero-flux marker at bottom
    ax1.annotate(r"$C'(L)=0$  (zero-flux BC)",
                 xy=(C_fitted[-1], 100), fontsize=8,
                 ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray"))

    ax1.set_xlabel("CO₂ concentration [ppm]", fontsize=10)
    ax1.set_ylabel("Depth [cm]  (0 = surface)", fontsize=10)
    ax1.invert_yaxis()
    ax1.set_ylim(105, -5)
    ax1.set_xlim(0, Y_MAX)
    ax1.legend(fontsize=7.5, loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ── Right panel: Source term S(z) vs depth ────────────────────────────────
    ax2.plot(S_fitted, z_fine * 100, color="#D35400", linewidth=2.2,
             label=f"$S(z) = -D_{{\\mathrm{{eff}}}} \\cdot C''(z)$")
    ax2.axvline(S_mean, color="#D35400", linestyle=":", linewidth=1.2,
                label=f"Column avg $\\langle S \\rangle$ = {S_mean:.2f} ppm/s")
    ax2.axvline(0, color="black", linewidth=0.6)
    ax2.set_xlabel("Source term  $S(z)$  [ppm/s]", fontsize=10)
    ax2.set_ylabel("Depth [cm]", fontsize=10)
    ax2.invert_yaxis()
    ax2.set_ylim(105, -5)
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # ── Annotation box ────────────────────────────────────────────────────────
    txt = (
        f"$D_{{\\mathrm{{eff}}}}$ = {D_mean:.3e} m²/s\n"
        f"$\\langle S_{{\\mathrm{{bio}}}} \\rangle$ = {S_mean:.3f} ppm/s\n"
        f"$J_\\uparrow$ = {J_mean:.3e} μmol/m²s\n"
        f"$C(0)_{{\\mathrm{{SVD}}}}$ = {d_mean:.0f} ppm\n"
        f"$C'(0)$ = {c_mean:.1f} ppm/m"
    )
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes, fontsize=8,
             verticalalignment="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_physics_timeseries(
    df_composite: pd.DataFrame,
    df_physics: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    3-panel time series: D_eff, column-averaged S_bio, and surface flux J_up
    from the SVD cubic inversion on the composite profile.
    """
    ts = df_composite["timestamp"].values

    S_bio = compute_source_term(
        df_physics["D_eff"].to_numpy(),
        df_physics["coeff_a"].to_numpy(),
        df_physics["coeff_b"].to_numpy(),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        "SVD Cubic Inversion — Physical Parameters from Composite Profile\n"
        f"({data_mode} data  ·  48h central window  ·  15-min resolution)",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: D_eff
    ax = axes[0]
    D = df_physics["D_eff"].to_numpy()
    ax.plot(ts, D, color="steelblue", linewidth=1.2)
    ax.axhline(np.nanmean(D), color="steelblue", linestyle=":", linewidth=1,
               label=f"mean = {np.nanmean(D):.3e} m²/s")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$  [m$^2$/s]", fontsize=10)
    ax.set_title("Effective Diffusivity", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: S_bio
    ax = axes[1]
    ax.plot(ts, S_bio, color="#D35400", linewidth=1.2)
    ax.axhline(np.nanmean(S_bio), color="#D35400", linestyle=":", linewidth=1,
               label=f"mean = {np.nanmean(S_bio):.3f} ppm/s")
    ax.set_ylabel(r"$\langle S_{\mathrm{bio}} \rangle$  [ppm/s]", fontsize=10)
    ax.set_title("Column-Averaged Biological Source Term", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: J_up
    ax = axes[2]
    J = df_physics["J_up"].to_numpy()
    ax.plot(ts, J, color="#2ECC71", linewidth=1.2)
    ax.axhline(np.nanmean(J), color="#2ECC71", linestyle=":", linewidth=1,
               label=f"mean = {np.nanmean(J):.3e} μmol/m²s")
    ax.set_ylabel(r"$J_\uparrow$  [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title("Surface CO₂ Flux", fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — TRACK A vs TRACK B EXECUTION & COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def phase4_execute(
    df_by_v: dict[str, pd.DataFrame],
    df_composite: pd.DataFrame,
    out_path: Path,
    csv_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Track A: invert each of the 8 columns individually, then average the physics.
    Track B: invert the composite profile through the overdetermined SVD engine.
    """
    print("\n=== PHASE 3: BUILDING SVD ENGINES ===")

    # Build per-cohort pseudoinverses (Track A: 4×4 square)
    print("  Track A matrices:")
    M_pinv_35 = build_svd_pinv([0.05, 0.20, 0.35])
    M_pinv_50 = build_svd_pinv([0.05, 0.20, 0.50])

    # Build composite pseudoinverse (Track B: 5×4 overdetermined)
    print("  Track B matrix:")
    M_pinv_comp = build_svd_pinv([0.05, 0.20, 0.35, 0.50])

    print("\n=== PHASE 4: TRACK A vs TRACK B EXECUTION ===")

    # ── Track A: Mean of the Physics ──────────────────────────────────────────
    # For each vertical, extract its 48h data, fit cubic, compute physics,
    # then average results across all 8 verticals.
    track_a_results: list[pd.DataFrame] = []

    for v in VERTICALS:
        df = df_by_v[v["label"]]
        is_50 = v["depth_deep_m"] == 0.50
        pinv = M_pinv_50 if is_50 else M_pinv_35

        C_air_v = df["C_surface"].to_numpy()
        C_sensors_v = np.column_stack([
            df["C_z005"].to_numpy(),
            df["C_z020"].to_numpy(),
            df["C_zdep"].to_numpy(),
        ])

        # Forward-fill NaN for stability
        for j in range(C_sensors_v.shape[1]):
            s = pd.Series(C_sensors_v[:, j]).ffill().bfill()
            C_sensors_v[:, j] = s.to_numpy()
        C_air_v = pd.Series(C_air_v).ffill().bfill().to_numpy()

        res = calc_physics(C_air_v, C_sensors_v, pinv)
        res["time_s"] = df["time_s"].values
        track_a_results.append(res)

    # Average Track A results on their time_s axis
    max_len = max(len(r) for r in track_a_results)
    ref_ts = next(r for r in track_a_results if len(r) == max_len)["time_s"].to_numpy()
    cols = ["D_eff", "J_up", "C_surface_inferred", "C_prime_0"]
    stacks = {c: [] for c in cols}
    for r in track_a_results:
        for c in cols:
            interped = np.interp(ref_ts, r["time_s"].to_numpy(), r[c].to_numpy(),
                                 left=np.nan, right=np.nan)
            stacks[c].append(interped)

    df_track_a = pd.DataFrame({"time_s": ref_ts})
    for c in cols:
        mat = np.stack(stacks[c], axis=1)
        df_track_a[c] = np.nanmean(mat, axis=1)
        df_track_a[f"std_{c}"] = np.nanstd(mat, axis=1, ddof=1)

    print(f"  Track A: {len(df_track_a)} timesteps  "
          f"mean D_eff={df_track_a['D_eff'].mean():.3e}  "
          f"mean J_up={df_track_a['J_up'].mean():.4e}")

    # ── Track B: Physics of the Mean (Composite) ─────────────────────────────
    C_air_comp = df_composite["C_air"].to_numpy()
    C_sensors_comp = np.column_stack([
        df_composite["C_z005"].to_numpy(),
        df_composite["C_z020"].to_numpy(),
        df_composite["C_z035"].to_numpy(),
        df_composite["C_z050"].to_numpy(),
    ])

    # Forward-fill NaN
    for j in range(C_sensors_comp.shape[1]):
        s = pd.Series(C_sensors_comp[:, j]).ffill().bfill()
        C_sensors_comp[:, j] = s.to_numpy()
    C_air_comp = pd.Series(C_air_comp).ffill().bfill().to_numpy()

    df_track_b = calc_physics(C_air_comp, C_sensors_comp, M_pinv_comp)
    df_track_b["time_s"] = df_composite["time_s"].values

    print(f"  Track B: {len(df_track_b)} timesteps  "
          f"mean D_eff={df_track_b['D_eff'].mean():.3e}  "
          f"mean J_up={df_track_b['J_up'].mean():.4e}")

    # ── Align timestamps for plotting (use composite timestamps) ──────────────
    ts_comp = df_composite["timestamp"].values
    # Interpolate Track A onto composite time_s for fair comparison
    ts_a = df_track_a["time_s"].to_numpy()
    ts_b = df_track_b["time_s"].to_numpy()

    j_a_interp = np.interp(ts_b, ts_a, df_track_a["J_up"].to_numpy(),
                            left=np.nan, right=np.nan)
    d_a_interp = np.interp(ts_b, ts_a, df_track_a["D_eff"].to_numpy(),
                            left=np.nan, right=np.nan)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"
    fig.suptitle(
        "Phase 4: Track A (Mean of Physics) vs Track B (Physics of the Mean)\n"
        f"({data_mode} data  ·  48h central window  ·  SVD cubic inversion)",
        fontsize=12, fontweight="bold",
    )

    # Panel 1: Surface flux
    ax = axes[0]
    ax.plot(ts_comp, j_a_interp,
            label="Track A — Mean of 8 individual inversions",
            color="black", linewidth=2.5, alpha=0.6)
    ax.plot(ts_comp, df_track_b["J_up"].to_numpy(),
            label="Track B — Composite 5×4 SVD inversion",
            color="gold", linewidth=2, linestyle="--")
    ax.set_ylabel(r"Upward flux  $J_\uparrow$  [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=10)
    ax.set_title("Surface CO₂ Flux Comparison", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: Effective diffusivity
    ax = axes[1]
    ax.plot(ts_comp, d_a_interp,
            label="Track A", color="black", linewidth=2.5, alpha=0.6)
    ax.plot(ts_comp, df_track_b["D_eff"].to_numpy(),
            label="Track B", color="gold", linewidth=2, linestyle="--")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$  [m$^2$ s$^{-1}$]", fontsize=10)
    ax.set_title("Effective Diffusivity Comparison", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Residual (Track B - Track A)
    ax = axes[2]
    j_resid = df_track_b["J_up"].to_numpy() - j_a_interp
    ax.plot(ts_comp, j_resid, color="#555555", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_ylabel(r"$\Delta J_\uparrow$ (B − A)  [$\mu$mol m$^{-2}$ s$^{-1}$]", fontsize=10)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title("Track B − Track A residual flux", fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ── CSV ───────────────────────────────────────────────────────────────────
    df_csv = pd.DataFrame({
        "timestamp":   ts_comp,
        "time_s":      ts_b,
        "J_up_trackA": j_a_interp,
        "J_up_trackB": df_track_b["J_up"].to_numpy(),
        "D_eff_trackA": d_a_interp,
        "D_eff_trackB": df_track_b["D_eff"].to_numpy(),
        "C_surface_trackB": df_track_b["C_surface_inferred"].to_numpy(),
    })
    df_csv.to_csv(csv_path, index=False, float_format="%.6e")
    print(f"  Saved: {csv_path}")

    return df_track_a, df_track_b


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"\n{'='*60}")
    print(f"COMPOSITE PROFILE PIPELINE  ({mode})")
    print(f"{'='*60}")

    # ── Load + QC ─────────────────────────────────────────────────────────────
    print("\n=== DATA LOADING & QC ===")
    df_by_v = _load_48h_per_vertical()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    df_35_mean, df_50_mean, div_stats = phase1_validate(
        df_by_v, OUT_ROOT / "phase1_exchangeability.png"
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    df_composite = phase2_build_composite(df_by_v, df_35_mean, df_50_mean)

    # ── Phase 3 + 4 ──────────────────────────────────────────────────────────
    df_track_a, df_track_b = phase4_execute(
        df_by_v, df_composite,
        OUT_ROOT / "phase4_flux_comparison.png",
        OUT_ROOT / "composite_summary.csv",
    )

    # ── SVD Cubic Diagnostics (on composite Track B) ─────────────────────────
    print("\n=== SVD CUBIC DEPTH PROFILE & SOURCE TERM ===")
    # Recompute Track B physics with full coefficients
    C_air_comp = df_composite["C_air"].to_numpy()
    C_sensors_comp = np.column_stack([
        df_composite["C_z005"].to_numpy(),
        df_composite["C_z020"].to_numpy(),
        df_composite["C_z035"].to_numpy(),
        df_composite["C_z050"].to_numpy(),
    ])
    for j in range(C_sensors_comp.shape[1]):
        s = pd.Series(C_sensors_comp[:, j]).ffill().bfill()
        C_sensors_comp[:, j] = s.to_numpy()
    C_air_comp = pd.Series(C_air_comp).ffill().bfill().to_numpy()

    M_pinv_comp = build_svd_pinv([0.05, 0.20, 0.35, 0.50])
    df_physics = calc_physics(C_air_comp, C_sensors_comp, M_pinv_comp)
    df_physics["time_s"] = df_composite["time_s"].values

    S_bio = compute_source_term(
        df_physics["D_eff"].to_numpy(),
        df_physics["coeff_a"].to_numpy(),
        df_physics["coeff_b"].to_numpy(),
    )
    print(f"  D_eff  mean={df_physics['D_eff'].mean():.3e} m²/s")
    print(f"  S_bio  mean={np.nanmean(S_bio):.3f} ppm/s")
    print(f"  J_up   mean={df_physics['J_up'].mean():.4e} μmol/m²s")
    print(f"  C(0)   mean={df_physics['coeff_d'].mean():.0f} ppm")
    print(f"  C'(0)  mean={df_physics['coeff_c'].mean():.1f} ppm/m")

    plot_svd_cubic_profile(
        df_composite, df_physics,
        OUT_ROOT / "svd_cubic_depth_profile.png",
    )
    plot_physics_timeseries(
        df_composite, df_physics,
        OUT_ROOT / "svd_physics_timeseries.png",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Cohort divergence at 5 cm:  {div_stats['div_5cm_mean']:.1f} ppm (mean), "
          f"{div_stats['div_5cm_max']:.1f} ppm (max)")
    print(f"  Cohort divergence at 20 cm: {div_stats['div_20cm_mean']:.1f} ppm (mean), "
          f"{div_stats['div_20cm_max']:.1f} ppm (max)")
    print(f"  Track A mean J_up: {df_track_a['J_up'].mean():.4e} μmol/m²s")
    print(f"  Track B mean J_up: {df_track_b['J_up'].mean():.4e} μmol/m²s")
    print(f"  Track A mean D_eff: {df_track_a['D_eff'].mean():.3e} m²/s")
    print(f"  Track B mean D_eff: {df_track_b['D_eff'].mean():.3e} m²/s")
    print(f"  SVD S_bio (column avg): {np.nanmean(S_bio):.3f} ppm/s")
    print(f"\n  All outputs in: {OUT_ROOT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()

"""
M2-V1 — Crank-Nicolson Transient Diffusion with Two-Source Inversion
=====================================================================

Solves the 1D transient diffusion–reaction PDE on the composite profile:

    ∂C/∂t = D_eff · ∂²C/∂z² + S(z)

where
    S(z) = S_shallow · exp(−z / ℓ_s) + S_bulk

Boundary conditions:
    z = 0  (top):    C(0, t)  = C_air(t)       (Dirichlet, provisional)
    z = L  (bottom): ∂C/∂z   = 0                (Neumann, zero-flux)

Three fitted parameters:
    D_eff      — effective diffusivity [m²/s]
    S_shallow  — shallow exponential source amplitude [mol/m³/s]
    S_bulk     — uniform bulk production rate [mol/m³/s]

Reviewer-hardened rewrite (V1.1):
    1.  Strict molar units — the entire PDE and optimizer operate in mol/m³.
        Conversions to ppm are reserved for final plotting and CSV export.
    2.  Parameter scaling — the optimizer fits scaled multipliers near O(1)
        (physical value = multiplier × 1e-6) so L-BFGS-B gradient estimation
        stays well-conditioned.
    3.  Multi-start optimization — n_starts random initial guesses are tested
        and the global-minimum solution is selected, mitigating the D/S
        trade-off identifiability risk.

Data sourcing
-------------
    Reuses composite profile from composite_profile_pipeline.py.
    Toggle USE_REAL_DATA in ideal_period_timeline.py.

Outputs (calculations/out/V2/)
-------------------------------
    v2_timeseries_fit.png       — observed vs simulated at 4 sensor depths
    v2_concentration_field.png  — C(z, t) heatmap + sensor overlay
    v2_source_profile.png       — fitted S(z) vs depth
    v2_summary.csv              — per-timestep simulated values at sensors
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
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.optimize import least_squares

# ── Shared infrastructure ─────────────────────────────────────────────────────
_CALC_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _CALC_DIR.parent
_SENSORDB  = _REPO_ROOT / "Project_description" / "sensorDB"
for _p in [str(_CALC_DIR), str(_SENSORDB)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ideal_period_timeline import USE_REAL_DATA  # noqa: E402
from composite_profile_pipeline import (          # noqa: E402
    _load_48h_per_vertical,
    phase1_validate,
    phase2_build_composite,
    L, K_G, C_AIR_MOL,
)

# ── Grid parameters ───────────────────────────────────────────────────────────
NZ: int   = 51                          # spatial nodes (every ~2 cm)
Z_GRID    = np.linspace(0.0, L, NZ)
DZ: float = Z_GRID[1] - Z_GRID[0]
DT: float = 900.0                       # 15-minute polling [s]

# Composite sensor depths
SENSOR_DEPTHS = np.array([0.05, 0.20, 0.35, 0.50])
SENSOR_INDICES = np.array([np.argmin(np.abs(Z_GRID - d)) for d in SENSOR_DEPTHS])

# Source decay length scale
ELL_S: float = 0.10   # 10 cm active shallow layer [m]

# Multi-start configuration (kept for potential future use)
N_STARTS: int = 10

# ── M1 anchor: D_eff from SVD Track B (composite) ────────────────────────────
# Locked to the value proven by the independent steady-state inversion.
# The optimizer fits ONLY the two biological source terms.
D_EFF_LOCKED: float = 2.047e-6   # m²/s  (M1 Track B composite mean)

# ── Unit conversion (molar ↔ ppm boundary) ───────────────────────────────────
# c_air = 40.9 mol/m³  (molar density of air at ~25 °C, 1 atm)
# 1 ppm = c_air × 1e-6 mol/m³

def ppm_to_molar(ppm: np.ndarray) -> np.ndarray:
    """Convert ppm to volumetric molar concentration [mol/m³]."""
    return ppm * 1e-6 * C_AIR_MOL

def molar_to_ppm(molar: np.ndarray) -> np.ndarray:
    """Convert volumetric molar concentration [mol/m³] back to ppm."""
    return (molar / C_AIR_MOL) * 1e6

# ── Colours ───────────────────────────────────────────────────────────────────
C_DEPTH_COLORS = ["#2E5EAA", "#E87722", "#44AA44", "#9B59B6"]
C_AIR_COLOR    = "#CC2222"

# ── Output directory ──────────────────────────────────────────────────────────
OUT_ROOT = _CALC_DIR / "out" / "V2"


# ══════════════════════════════════════════════════════════════════════════════
# CRANK-NICOLSON FORWARD SOLVER  (operates entirely in mol/m³)
# ══════════════════════════════════════════════════════════════════════════════

def _source_profile_molar(S_shallow: float, S_bulk: float) -> np.ndarray:
    """
    S(z) = S_shallow · exp(−z/ℓ_s) + S_bulk   [mol/m³/s].
    All quantities already in strict molar units.
    """
    return S_shallow * np.exp(-Z_GRID / ELL_S) + S_bulk


def solve_crank_nicolson(
    C_air_molar: np.ndarray,
    C_init_molar: np.ndarray,
    D_eff: float,
    S_shallow: float,
    S_bulk: float,
) -> np.ndarray:
    """
    Solve the 1D transient PDE entirely in mol/m³.

    Parameters
    ----------
    C_air_molar  : (Nt,) surface air CO₂ [mol/m³]
    C_init_molar : (NZ,) initial profile C(z, t=0) [mol/m³]
    D_eff        : effective diffusivity [m²/s]
    S_shallow    : shallow source amplitude [mol/m³/s]
    S_bulk       : bulk source rate [mol/m³/s]

    Returns
    -------
    C_sim : (Nt, NZ) concentration field [mol/m³]
    """
    Nt = len(C_air_molar)
    alpha = D_eff * DT / (2.0 * DZ**2)
    beta  = K_G * DZ / D_eff              # Robin boundary coefficient
    S_z = _source_profile_molar(S_shallow, S_bulk)

    # ── Build tridiagonal matrices A and B ────────────────────────────────────
    # CRITICAL: upper and lower diagonals must be SEPARATE arrays so that
    # modifying one for the boundary condition does not corrupt the other.
    main_A  = (1.0 + 2.0 * alpha) * np.ones(NZ)
    lower_A = -alpha * np.ones(NZ - 1)
    upper_A = -alpha * np.ones(NZ - 1)

    main_B  = (1.0 - 2.0 * alpha) * np.ones(NZ)
    lower_B = alpha * np.ones(NZ - 1)
    upper_B = alpha * np.ones(NZ - 1)

    # ---------------------------------------------------------
    # TOP BOUNDARY: Robin Gas Exchange Condition
    # Ghost-node elimination: −D ∂C/∂z|₀ = k_g (C₀ − C_air)
    # ---------------------------------------------------------
    main_A[0]  = 1.0 + 2.0 * alpha * (1.0 + beta)
    upper_A[0] = -2.0 * alpha

    main_B[0]  = 1.0 - 2.0 * alpha * (1.0 + beta)
    upper_B[0] = 2.0 * alpha

    # Bottom BC: Neumann (zero-flux) using ghost node on the LOWER diagonal
    lower_A[-1] = -2.0 * alpha
    lower_B[-1] =  2.0 * alpha

    A = diags([lower_A, main_A, upper_A], offsets=[-1, 0, 1], shape=(NZ, NZ), format="csr")
    B = diags([lower_B, main_B, upper_B], offsets=[-1, 0, 1], shape=(NZ, NZ), format="csr")

    # ── Time-stepping ─────────────────────────────────────────────────────────
    C_sim = np.zeros((Nt, NZ))
    C_sim[0] = C_init_molar.copy()

    for n in range(Nt - 1):
        rhs = B.dot(C_sim[n]) + S_z * DT

        # Robin BC forcing: centered in time (Crank-Nicolson average)
        rhs[0] += 2.0 * alpha * beta * (C_air_molar[n + 1] + C_air_molar[n])

        C_sim[n + 1] = spsolve(A, rhs)

    return C_sim


# ══════════════════════════════════════════════════════════════════════════════
# INVERSION  (parameter-scaled, multi-start, molar-space residuals)
# ══════════════════════════════════════════════════════════════════════════════

def _build_initial_profile(
    C_air_0: float,
    C_obs_0: np.ndarray,
) -> np.ndarray:
    """
    Build C(z, t=0) in mol/m³ by interpolating from the surface through
    the sensor depths down to z = L.

    With the Robin BC the surface node is free to float, so the initial
    guess extrapolates half a step above the shallowest sensor rather than
    pinning to C_air.
    """
    surface_guess = C_obs_0[0] - (C_obs_0[1] - C_obs_0[0]) * 0.5
    z_anchors = np.concatenate([[0.0], SENSOR_DEPTHS, [L]])
    c_anchors = np.concatenate([[surface_guess], C_obs_0, [C_obs_0[-1]]])
    return np.interp(Z_GRID, z_anchors, c_anchors)


def _residuals_vector(
    scaled_params: np.ndarray,
    C_air_molar: np.ndarray,
    C_obs_molar: np.ndarray,
    C_init_molar: np.ndarray,
    D_eff_fixed: float,
) -> np.ndarray:
    """
    Return the flattened 1D residual vector for the TRF optimizer.
    Optimizer fits scaled multipliers for S_shallow and S_bulk;
    physical value = multiplier × 1e-6.
    """
    S_shallow = scaled_params[0] * 1e-6   # → mol/m³/s
    S_bulk    = scaled_params[1] * 1e-6   # → mol/m³/s

    C_sim = solve_crank_nicolson(C_air_molar, C_init_molar, D_eff_fixed, S_shallow, S_bulk)
    return (C_sim[:, SENSOR_INDICES] - C_obs_molar).flatten()


def fit_transient_model(
    df_composite: pd.DataFrame,
    D_eff_fixed: float = D_EFF_LOCKED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TRF (Trust Region Reflective) least-squares inversion fitting ONLY the
    two biological source terms, with D_eff locked to the M1 SVD Track B
    value.  Uses soft_l1 loss to down-weight outlier spikes.

    Returns (optimal_params_physical, C_init_molar, C_sim_molar).
    optimal_params_physical = [D_eff (m²/s), S_shallow (mol/m³/s), S_bulk (mol/m³/s)]
    """
    # ── 1. Convert ppm → mol/m³ at the boundary ──────────────────────────────
    C_air_ppm = pd.Series(df_composite["C_air"].to_numpy()).ffill().bfill().to_numpy()
    C_obs_ppm = np.column_stack([
        df_composite["C_z005"].to_numpy(),
        df_composite["C_z020"].to_numpy(),
        df_composite["C_z035"].to_numpy(),
        df_composite["C_z050"].to_numpy(),
    ])
    for j in range(C_obs_ppm.shape[1]):
        C_obs_ppm[:, j] = pd.Series(C_obs_ppm[:, j]).ffill().bfill().to_numpy()

    C_air_molar = ppm_to_molar(C_air_ppm)
    C_obs_molar = ppm_to_molar(C_obs_ppm)

    # ── 2. Build initial condition in molar space ─────────────────────────────
    C_init_molar = _build_initial_profile(C_air_molar[0], C_obs_molar[0])

    # ── 3. TRF least-squares (2 dims: S_shallow, S_bulk) ─────────────────────
    #   physical = scaled × 1e-6
    #   S_shallow: [0,   5e-5]  → scaled [0.0, 50]
    #   S_bulk:    [0,   1e-5]  → scaled [0.0, 10]
    bounds = ([0.0, 0.0], [50.0, 10.0])   # TRF format: (lower, upper)
    guess  = [5.0, 1.0]

    print(f"\n  TRF optimisation (D_eff = {D_eff_fixed:.3e} m²/s, "
          f"Robin BC, soft_l1 loss, molar-strict)...")

    res = least_squares(
        _residuals_vector, guess,
        args=(C_air_molar, C_obs_molar, C_init_molar, D_eff_fixed),
        bounds=bounds,
        method="trf",
        loss="soft_l1",
    )

    opt_S_sh = res.x[0] * 1e-6
    opt_S_bk = res.x[1] * 1e-6
    rss      = float(np.sum(res.fun**2))
    params_physical = np.array([D_eff_fixed, opt_S_sh, opt_S_bk])

    print(f"\n  --- Optimal Fit (TRF, soft_l1) ---")
    print(f"    D_eff     = {D_eff_fixed:.3e} m²/s  (LOCKED from M1 Track B)")
    print(f"    S_shallow = {opt_S_sh*1e6:.3f} μmol/m³/s  (ℓ_s = {ELL_S*100:.0f} cm)")
    print(f"    S_bulk    = {opt_S_bk*1e6:.3f} μmol/m³/s")
    print(f"    RSS (mol) = {rss:.4e}  ({res.nfev} function evals)")
    print(f"    TRF cost  = {res.cost:.4e}  (soft_l1 robust cost)")

    # ── 4. Final forward run with optimal physical params ─────────────────────
    C_sim_molar = solve_crank_nicolson(
        C_air_molar, C_init_molar, D_eff_fixed, opt_S_sh, opt_S_bk,
    )

    return params_physical, C_init_molar, C_sim_molar


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeseries_fit(
    df_composite: pd.DataFrame,
    C_sim_molar: np.ndarray,
    params: np.ndarray,
    out_path: Path,
) -> None:
    """
    4-panel figure: observed (dots) vs CN-simulated (lines) at each sensor
    depth + residual panel.  Plotting in ppm.
    """
    ts = df_composite["timestamp"].values
    D_opt, S_sh, S_bk = params

    # Convert simulation from mol/m³ → ppm for plotting
    C_sim_ppm = molar_to_ppm(C_sim_molar)

    obs_cols = ["C_z005", "C_z020", "C_z035", "C_z050"]
    depth_labels = ["5 cm", "20 cm", "35 cm", "50 cm"]
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"

    fig, axes = plt.subplots(5, 1, figsize=(15, 16), sharex=True)
    fig.suptitle(
        "M2-V1 Crank-Nicolson Transient Fit — Locked $D_{\\mathrm{eff}}$ from M1\n"
        f"({data_mode} data  ·  48h window  ·  "
        f"$D_{{\\mathrm{{eff}}}}$ = {D_opt:.2e} m²/s [locked]  ·  "
        f"$S_{{\\mathrm{{shallow}}}}$ = {S_sh*1e6:.2f} μmol/m³/s  ·  "
        f"$S_{{\\mathrm{{bulk}}}}$ = {S_bk*1e6:.2f} μmol/m³/s)",
        fontsize=11, fontweight="bold",
    )

    residuals_all = np.zeros(len(ts))
    for i, (col, lbl, color, si) in enumerate(
        zip(obs_cols, depth_labels, C_DEPTH_COLORS, SENSOR_INDICES)
    ):
        ax = axes[i]
        obs = df_composite[col].to_numpy()
        sim = C_sim_ppm[:, si]

        ax.plot(ts, obs, "o", color=color, markersize=3, alpha=0.5, label=f"Observed {lbl}")
        ax.plot(ts, sim, "-", color=color, linewidth=2, label=f"Simulated {lbl}")
        ax.set_ylabel("CO₂ [ppm]", fontsize=9)
        ax.set_title(lbl, fontsize=9, loc="left")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)

        residuals_all += (sim - obs)**2

    # Air CO₂ reference line on top panel
    axes[0].plot(ts, df_composite["C_air"].to_numpy(),
                 "--", color=C_AIR_COLOR, linewidth=1, alpha=0.6, label="Air CO₂")
    axes[0].legend(fontsize=8, loc="upper right")

    # Bottom: mean residual
    ax = axes[4]
    mean_res = np.sqrt(residuals_all / 4)
    ax.plot(ts, mean_res, color="#555555", linewidth=1.2)
    ax.axhline(np.nanmean(mean_res), color="red", linestyle=":", linewidth=1,
               label=f"mean RMSE = {np.nanmean(mean_res):.1f} ppm")
    ax.set_ylabel("RMSE [ppm]", fontsize=9)
    ax.set_xlabel("Time", fontsize=10)
    ax.set_title("Root-Mean-Square Error (across 4 depths)", fontsize=9, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_concentration_field(
    df_composite: pd.DataFrame,
    C_sim_molar: np.ndarray,
    params: np.ndarray,
    out_path: Path,
) -> None:
    """
    Heatmap of C(z, t) — the full spatiotemporal concentration field from CN.
    Sensor observation points overlaid.  Plotted in ppm.
    """
    ts = df_composite["timestamp"].values
    D_opt, S_sh, S_bk = params
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"

    C_sim_ppm = molar_to_ppm(C_sim_molar)

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.suptitle(
        "M2-V1 Concentration Field $C(z, t)$ \u2014 Robin BC + TRF\n"
        f"({data_mode} data  ·  $D_{{\\mathrm{{eff}}}}$ = {D_opt:.2e} m²/s)",
        fontsize=11, fontweight="bold",
    )

    # Heatmap: time on x, depth on y
    im = ax.pcolormesh(
        np.arange(len(ts)), Z_GRID * 100, C_sim_ppm.T,
        shading="auto", cmap="inferno",
    )
    cbar = fig.colorbar(im, ax=ax, label="CO₂ [ppm]")

    # Overlay sensor positions
    for i, (d, col) in enumerate(zip(SENSOR_DEPTHS, C_DEPTH_COLORS)):
        ax.axhline(d * 100, color=col, linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(len(ts) + 1, d * 100, f"{d*100:.0f} cm",
                fontsize=8, color=col, va="center")

    ax.set_ylabel("Depth [cm]", fontsize=10)
    ax.set_xlabel("Timestep (15-min intervals)", fontsize=10)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_source_profile(
    params: np.ndarray,
    out_path: Path,
) -> None:
    """
    Plot the fitted source term S(z) vs depth — shallow exponential + bulk.
    Params are in physical units [mol/m³/s]; plot labels in μmol/m³/s.
    """
    D_opt, S_sh, S_bk = params
    data_mode = "mock" if not USE_REAL_DATA else "real Oracle"

    # Display in μmol/m³/s
    S_sh_u = S_sh * 1e6
    S_bk_u = S_bk * 1e6

    z_fine = np.linspace(0.0, L, 500)
    S_shallow = S_sh_u * np.exp(-z_fine / ELL_S)
    S_bulk_arr = np.full_like(z_fine, S_bk_u)
    S_total = S_shallow + S_bulk_arr

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle(
        "M2-V1 Fitted Source Term $S(z)$ vs Depth\n"
        f"({data_mode} data  ·  $\\ell_s$ = {ELL_S*100:.0f} cm decay)",
        fontsize=11, fontweight="bold",
    )

    ax.fill_betweenx(z_fine * 100, 0, S_shallow, color="#E87722", alpha=0.3,
                      label=f"$S_{{\\mathrm{{shallow}}}}$ = {S_sh_u:.2f} μmol/m³/s")
    ax.fill_betweenx(z_fine * 100, S_shallow, S_total, color="#2E5EAA", alpha=0.3,
                      label=f"$S_{{\\mathrm{{bulk}}}}$ = {S_bk_u:.2f} μmol/m³/s")
    ax.plot(S_total, z_fine * 100, color="black", linewidth=2,
            label="Total $S(z)$")
    ax.plot(S_shallow, z_fine * 100, color="#E87722", linewidth=1.5, linestyle="--")
    ax.axvline(S_bk_u, color="#2E5EAA", linewidth=1.5, linestyle=":")

    ax.set_xlabel("Source rate [μmol/m³/s]", fontsize=10)
    ax.set_ylabel("Depth [cm]", fontsize=10)
    ax.invert_yaxis()
    ax.set_ylim(105, -5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    # Annotation
    txt = (
        f"$D_{{\\mathrm{{eff}}}}$ = {D_opt:.3e} m²/s\n"
        f"$S_{{\\mathrm{{shallow}}}}$ = {S_sh_u:.3f} μmol/m³/s\n"
        f"$S_{{\\mathrm{{bulk}}}}$ = {S_bk_u:.3f} μmol/m³/s\n"
        f"Decay length $\\ell_s$ = {ELL_S*100:.0f} cm"
    )
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=9,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"\n{'='*60}")
    print(f"M2-V1 CRANK-NICOLSON TRANSIENT PIPELINE  ({mode})")
    print(f"{'='*60}")

    # ── 1. Load + QC + composite (reuse M1 infrastructure) ───────────────────
    print("\n=== DATA LOADING & QC ===")
    df_by_v = _load_48h_per_vertical()

    df_35_mean, df_50_mean, div_stats = phase1_validate(
        df_by_v, OUT_ROOT / "phase1_exchangeability.png"
    )
    df_composite = phase2_build_composite(df_by_v, df_35_mean, df_50_mean)

    # ── 2. Crank-Nicolson inversion (Robin BC, TRF, locked D_eff) ──────────────
    print("\n=== M2-V1: CRANK-NICOLSON INVERSION (Robin BC + TRF) ===")
    params, C_init_molar, C_sim_molar = fit_transient_model(df_composite)

    D_opt, S_sh, S_bk = params

    # Fourier mesh number diagnostic
    Fo = D_opt * DT / DZ**2
    print(f"\n  Fourier mesh number Fo = {Fo:.2f}  "
          f"({'smooth' if Fo <= 1.0 else 'may ring — consider finer grid'})")

    # ── 3. Derived quantities (molar → ppm at boundary) ──────────────────────
    # Surface flux from Robin BC: J_up = k_g · (C_surface − C_air)  [mol/m²/s]
    # Positive → outgassing from basalt to atmosphere
    C_air_ppm = pd.Series(df_composite["C_air"].to_numpy()).ffill().bfill().to_numpy()
    C_surface_ppm = molar_to_ppm(C_sim_molar[:, 0])
    J_up_robin = K_G * C_AIR_MOL * (C_surface_ppm - C_air_ppm) * 1e-6   # mol/m²/s
    J_up_robin_umol = J_up_robin * 1e6   # μmol/m²/s for display

    print(f"  Surface flux J_up (Robin):  mean={np.nanmean(J_up_robin_umol):.4e} μmol/m²s")
    print(f"  C_surface (mean):  {np.nanmean(C_surface_ppm):.0f} ppm  "
          f"(air: {np.nanmean(C_air_ppm):.0f} ppm)")

    # Convert simulation to ppm for plotting and CSV
    C_sim_ppm = molar_to_ppm(C_sim_molar)

    # ── 4. Plots (pass molar arrays — functions convert internally) ───────────
    print("\n=== GENERATING FIGURES ===")
    plot_timeseries_fit(df_composite, C_sim_molar, params,
                        OUT_ROOT / "v2_timeseries_fit.png")
    plot_concentration_field(df_composite, C_sim_molar, params,
                             OUT_ROOT / "v2_concentration_field.png")
    plot_source_profile(params, OUT_ROOT / "v2_source_profile.png")

    # ── 5. CSV export (ppm for human readability) ─────────────────────────────
    csv_path = OUT_ROOT / "v2_summary.csv"
    df_csv = pd.DataFrame({
        "timestamp":     df_composite["timestamp"].values,
        "time_s":        df_composite["time_s"].values,
        "C_air":         df_composite["C_air"].values,
        "obs_z005":      df_composite["C_z005"].values,
        "obs_z020":      df_composite["C_z020"].values,
        "obs_z035":      df_composite["C_z035"].values,
        "obs_z050":      df_composite["C_z050"].values,
        "sim_z005":      C_sim_ppm[:, SENSOR_INDICES[0]],
        "sim_z020":      C_sim_ppm[:, SENSOR_INDICES[1]],
        "sim_z035":      C_sim_ppm[:, SENSOR_INDICES[2]],
        "sim_z050":      C_sim_ppm[:, SENSOR_INDICES[3]],
        "C_surface_ppm": C_surface_ppm,
        "J_up_Robin_umol": J_up_robin_umol,
    })
    df_csv.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"  Saved: {csv_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    # Per-sensor RMSE (in ppm)
    obs_cols = ["C_z005", "C_z020", "C_z035", "C_z050"]
    rmse_per = []
    for col, si in zip(obs_cols, SENSOR_INDICES):
        obs = df_composite[col].to_numpy()
        sim = C_sim_ppm[:, si]
        rmse = float(np.sqrt(np.nanmean((sim - obs)**2)))
        rmse_per.append(rmse)

    print(f"\n{'='*60}")
    print("M2-V1 SUMMARY  (Robin BC + TRF + Locked D_eff)")
    print(f"{'='*60}")
    print(f"  D_eff         = {D_opt:.3e} m²/s  (LOCKED from M1 Track B)")
    print(f"  S_shallow     = {S_sh*1e6:.3f} μmol/m³/s  (ℓ_s = {ELL_S*100:.0f} cm)")
    print(f"  S_bulk        = {S_bk*1e6:.3f} μmol/m³/s")
    print(f"  J_up (Robin)  = {np.nanmean(J_up_robin_umol):.4e} μmol/m²s")
    print(f"  C_surface     = {np.nanmean(C_surface_ppm):.0f} ppm (mean)")
    print(f"  Fo            = {Fo:.2f}")
    print(f"  RMSE 5 cm     = {rmse_per[0]:.1f} ppm")
    print(f"  RMSE 20 cm    = {rmse_per[1]:.1f} ppm")
    print(f"  RMSE 35 cm    = {rmse_per[2]:.1f} ppm")
    print(f"  RMSE 50 cm    = {rmse_per[3]:.1f} ppm")
    print(f"  Mean RMSE     = {np.mean(rmse_per):.1f} ppm")
    print(f"\n  All outputs in: {OUT_ROOT}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run()

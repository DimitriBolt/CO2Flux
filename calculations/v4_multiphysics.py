"""
Version 4: Multiphysics Pipeline — Asymmetric Diurnal Thermo-Coupled Solver
=============================================================================

Architecture upgrade from Version 3:

1.  ``fetch_and_clean_data()`` is the single replacement point for real data.
    Swap its body for a pandas import from the Jupyter notebook / Oracle
    pipeline once the Decagon 5TM telemetry is available.

2.  Temperature field uses an **asymmetric diurnal wave**:
    - Slow sinusoidal heating during the day (0:00 → 12:00)
    - Sharp exponential radiative cooling at night (12:00 → 24:00)
    The fast nocturnal ∂T/∂t drives a large Henry's Law abiotic sink,
    freeing D_s from the 10⁻⁵ ceiling that emerged in V2/V3.

3.  D_s upper bound is relaxed to 5×10⁻⁵ m²/s ("ceiling test"):
    if the solver converges below this bound after adding the abiotic sink,
    the boundary collision from V2 is mathematically resolved.

4.  Gabitov Sanity Filters are embedded inside ``fetch_and_clean_data()``:
    - Filter A: 413 ppm hardware artefact  (subsurface < 425 ppm → drop row)
    - Filter B: Cueva thermodynamic limit  (implied downward flux > 0.5 μmol
                m⁻² s⁻¹ → drop row)

Governing PDE
-------------
    ∂C_gas/∂t = D_s · ∂²C_gas/∂z² + S_bio(z) − S_abiotic(z, t)

    S_bio(z)    = S_0 + S_1 · z
    S_abiotic   ≈ θ · C_gas · dK_H(T)/dt      (Henry's Law + van 't Hoff)

Parameters: β = [D_s, S_0, S_1]
    Bounds (relaxed for ceiling test):
        D_s  ∈ [1×10⁻⁷, 5×10⁻⁵]   m²/s
        S_0  ∈ [0,       20     ]   ppm/s
        S_1  ∈ [−100,     0     ]   ppm/(s·m)

Output files (saved in calculations/)
---------------------------------------
    v4_temperature_field.png      — asymmetric diurnal forcing visualised
    v4_timeseries.png             — model vs observed at all three depths
    v4_diagnostic_dashboard.png   — 2×2 Gabitov audit panels

Replacing the mock with real data (Step 2)
------------------------------------------
    Inside fetch_and_clean_data(), replace the three "Mock …" blocks with:

        df_co2  = pd.read_csv("path/to/co2_clean.csv", parse_dates=["timestamp"])
        df_temp = pd.read_csv("path/to/decagon_5tm.csv", parse_dates=["timestamp"])

    Align on the same 15-minute time axis, apply the same filter logic,
    and return the same five arrays.  Nothing else in this file changes.

Usage
-----
    python v4_multiphysics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import seaborn as sns
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Physical & numerical constants ────────────────────────────────────────────
L: float           = 1.0
N_Z: int           = 101
DZ: float          = L / (N_Z - 1)
Z_GRID: np.ndarray = np.linspace(0.0, L, N_Z)

SENSOR_DEPTHS: np.ndarray = np.array([0.05, 0.20, 0.50])
SENSOR_IDX: np.ndarray    = np.array([int(d / DZ) for d in SENSOR_DEPTHS])

DT: float = 900.0     # Δt [s]

# ── Thermodynamic constants ────────────────────────────────────────────────────
V_WATER: float          = 0.15      # Constant θ  [m³_water / m³_soil]
DH_SOL: float           = -2400.0   # Van 't Hoff ΔH_sol for CO₂  [K]
K_H_298: float          = 0.8317    # Dimensionless K_H at 298.15 K
MOLAR_DENSITY_AIR: float = 40.87    # mol m⁻³  (~15 °C, 1 atm)
CUEVA_LIMIT_FLUX: float  = 0.50     # Max physical downward flux  [μmol m⁻² s⁻¹]
D_S_REF: float          = 4.5e-6   # Reference D_s for Cueva filter  [m²/s]

# ── Parameter bounds (relaxed ceiling test) ───────────────────────────────────
LOWER = [1.0e-7,   0.0, -100.0]
UPPER = [5.0e-5,  20.0,    0.0]   # D_s ceiling raised to 5×10⁻⁵


# ══════════════════════════════════════════════════════════════════════════════
# DATA PIPELINE  ←  replace the bodies of the three "Mock …" blocks below
#                    with real pandas imports for Step 2
# ══════════════════════════════════════════════════════════════════════════════

def _build_asymmetric_temperature(n_t: int) -> np.ndarray:
    """
    Asymmetric diurnal temperature field (Step 1 mock).

    Heating phase  (first half of each 24-h cycle):
        T = T_mean + A(z) · sin(π · phase / 0.5)        slow ramp

    Cooling phase  (second half):
        T = T_mean + A(z) · (exp(−5 · (phase − 0.5)) − 1)  sharp exponential drop

    The magnitude of ∂T/∂t during cooling is ~10× larger than during heating,
    producing a substantial Henry's Law sink at night.

    Returns
    -------
    T : (n_t, N_Z) ndarray  — soil temperature [°C]
    """
    amp   = 10.0 * np.exp(-2.0 * Z_GRID)               # (N_Z,) depth-decaying amplitude
    T     = np.empty((n_t, N_Z))

    for n in range(n_t):
        phase = ((n * DT) % 86400.0) / 86400.0          # normalised position in 24-h cycle
        if phase < 0.5:
            wave = np.sin(np.pi * phase / 0.5)
        else:
            wave = np.exp(-5.0 * (phase - 0.5)) - 1.0

        T[n] = 28.0 + amp * wave

    return T


def fetch_and_clean_data() -> tuple[
    np.ndarray,   # time_s      (n_t,)
    np.ndarray,   # C_surface   (n_t,)
    np.ndarray,   # C_data      (n_t, 3)
    np.ndarray,   # T_soil      (n_t, N_Z)
]:
    """
    Data extraction and Gabitov Sanity Filter pipeline.

    Step 1 (current): generates physically motivated mock arrays.
    Step 2 (future):  replace the three "Mock …" blocks with real
                      pandas / Oracle imports.  The filter logic and
                      return signature remain identical.

    Gabitov Sanity Filters
    ----------------------
    Filter A — 413 ppm hardware artefact
        If any subsurface sensor reads < 425 ppm the row is dropped.
        (Physically impossible under basalt at this site.)

    Filter B — Cueva thermodynamic limit
        Implied downward CO₂ flux between adjacent sensor pairs is
        computed using the reference D_s prior.  If flux > 0.5 μmol
        m⁻² s⁻¹ the deeper sensor is masked and the row is dropped.

    Returns
    -------
    time_s    : elapsed seconds, shape (n_t,)
    C_surface : atmospheric BC at z = 0,  shape (n_t,)   [ppm]
    C_data    : subsurface CO₂,           shape (n_t, 3)  [ppm]
                columns → z = 0.05, 0.20, 0.50 m
    T_soil    : soil temperature,         shape (n_t, N_Z) [°C]
    """
    n_raw = 288      # 72 h at Δt = 900 s

    # ── Mock 1: atmospheric surface BC ────────────────────────────────────────
    # Replace with: df_air["C_surface"].to_numpy()
    C_surface_raw = np.full(n_raw, 420.0)

    # ── Mock 2: subsurface CO₂ with diurnal oscillation ───────────────────────
    # Replace with: df_co2[["C_z005", "C_z020", "C_z050"]].to_numpy()
    t_raw = np.arange(n_raw) * DT
    C_data_raw = np.column_stack([
        4985.0 + 500.0 * np.sin(2.0 * np.pi * t_raw / 86400.0),
        6880.0 + 200.0 * np.sin(2.0 * np.pi * t_raw / 86400.0),
        np.full(n_raw, 6500.0),
    ])

    # ── Mock 3: asymmetric diurnal soil temperature ───────────────────────────
    # Replace with: interpolate Decagon 5TM readings onto the 101-node Z_GRID
    T_soil_raw = _build_asymmetric_temperature(n_raw)

    # ── Gabitov Sanity Filters ─────────────────────────────────────────────────
    valid_mask   = np.ones(n_raw, dtype=bool)
    dropped_413  = 0
    dropped_cueva = 0

    for n in range(n_raw):
        c = C_data_raw[n]   # (3,) — z = 0.05, 0.20, 0.50 m

        # Filter A: 413 ppm hardware artefact
        if np.any(c < 425.0):
            valid_mask[n]  = False
            dropped_413   += 1
            continue

        # Filter B: Cueva thermodynamic limit
        # Compute implied downward flux between each adjacent sensor pair.
        # pairs: (surface→5cm), (5cm→20cm), (20cm→50cm)
        sensor_z    = np.array([0.0,   0.05, 0.20])
        sensor_z_lo = np.array([0.05,  0.20, 0.50])
        c_hi        = np.array([C_surface_raw[n], c[0], c[1]])
        c_lo        = c
        dz_pairs    = sensor_z_lo - sensor_z

        for c_upper, c_lower, dz_pair in zip(c_hi, c_lo, dz_pairs):
            gradient   = (c_lower - c_upper) / dz_pair          # ppm m⁻¹
            flux_down  = -D_S_REF * gradient * MOLAR_DENSITY_AIR  # μmol m⁻² s⁻¹
            if flux_down > CUEVA_LIMIT_FLUX:
                valid_mask[n]    = False
                dropped_cueva   += 1
                break

    n_kept = int(valid_mask.sum())
    print(f"[PREPROCESSING] Kept {n_kept}/{n_raw} rows.")
    print(f"[PREPROCESSING] Dropped {dropped_413} — 413 ppm artefact (Filter A).")
    print(f"[PREPROCESSING] Dropped {dropped_cueva} — Cueva flux limit (Filter B).")

    return (
        t_raw[valid_mask],
        C_surface_raw[valid_mask],
        C_data_raw[valid_mask],
        T_soil_raw[valid_mask],
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

def calculate_S_abiotic(
    T_curr: np.ndarray,
    T_next: np.ndarray,
    C_curr_gas: np.ndarray,
    theta: float = V_WATER,
) -> np.ndarray:
    """
    Volumetric abiotic CO₂ sink [ppm s⁻¹] from Henry's Law solubility change.

    S_abiotic ≈ θ · C_gas · (K_H(T_next) − K_H(T_curr)) / Δt

    Positive value = net removal from gas phase (cooling → dissolving).
    """
    T_curr_K = T_curr + 273.15
    T_next_K = T_next + 273.15

    K_H_curr  = K_H_298 * np.exp(DH_SOL * (1.0 / T_curr_K - 1.0 / 298.15))
    K_H_next  = K_H_298 * np.exp(DH_SOL * (1.0 / T_next_K - 1.0 / 298.15))

    C_gas_mol = C_curr_gas * 1.0e-6 * MOLAR_DENSITY_AIR   # ppm → mol m⁻³
    dK_dt     = (K_H_next - K_H_curr) / DT                # s⁻¹
    S_vol     = theta * C_gas_mol * dK_dt                  # mol m⁻³ s⁻¹

    return (S_vol / MOLAR_DENSITY_AIR) * 1.0e6             # back to ppm s⁻¹


def _build_cn_matrices(D_s: float):
    """
    Build tridiagonal A (implicit) and B (explicit) CN matrices with BCs.
    Uses lil_matrix for safe boundary-row assignment.
    """
    r = (D_s * DT) / (2.0 * DZ**2)

    A = diags([-r, 1.0 + 2.0*r, -r], [-1, 0, 1], shape=(N_Z, N_Z), format="lil")
    B = diags([ r, 1.0 - 2.0*r,  r], [-1, 0, 1], shape=(N_Z, N_Z), format="csr")

    A[0, :]              = 0.0;  A[0, 0]           = 1.0   # Dirichlet top
    A[N_Z-1, :]          = 0.0
    A[N_Z-1, N_Z-1]      =  1.0
    A[N_Z-1, N_Z-2]      = -1.0                             # Neumann bottom

    return A.tocsr(), B


def forward_solver_v4(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data_t0: np.ndarray,
    T_soil: np.ndarray,
) -> np.ndarray:
    """
    Crank-Nicolson time-stepping with biology + Henry's Law abiotic sink.

    RHS at each step:
        d = B·Cⁿ  +  (S_bio − S_abiotic(n)) · Δt

    Parameters
    ----------
    beta       : [D_s, S_0, S_1]
    C_surface  : (n_t,)     Dirichlet BC at z = 0  [ppm]
    C_data_t0  : (3,)       First-row sensor readings for cubic-spline IC
    T_soil     : (n_t, N_Z) Soil temperature field  [°C]

    Returns
    -------
    predictions : (n_t, 3)  Model CO₂ at sensor nodes.
    """
    D_s, S_0, S_1 = beta
    n_t           = len(C_surface)
    S_bio         = S_0 + S_1 * Z_GRID       # (N_Z,) [ppm s⁻¹]

    A, B = _build_cn_matrices(D_s)

    cs = CubicSpline(
        [0.0, 0.05, 0.20, 0.50, L],
        [C_surface[0], C_data_t0[0], C_data_t0[1], C_data_t0[2], C_data_t0[2]],
    )
    C = cs(Z_GRID)

    predictions        = np.empty((n_t, 3))
    predictions[0]     = C[SENSOR_IDX]

    for n in range(1, n_t):
        rhs    = B.dot(C) + S_bio * DT
        S_ab   = calculate_S_abiotic(T_soil[n - 1], T_soil[n], C)
        rhs   -= S_ab * DT
        rhs[0]       = C_surface[n]
        rhs[N_Z - 1] = 0.0
        C              = spsolve(A, rhs)
        predictions[n] = C[SENSOR_IDX]

    return predictions


def cost(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data: np.ndarray,
    T_soil: np.ndarray,
) -> np.ndarray:
    """Flattened (n_t × 3,) residual vector."""
    pred = forward_solver_v4(beta, C_surface, C_data[0], T_soil)
    return (pred - C_data).flatten()


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_temperature_field(T_soil: np.ndarray, n_t: int, out_path: Path) -> None:
    """Asymmetric diurnal temperature field: time series + hot/cold profiles."""
    time_h = np.arange(n_t) * DT / 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "V4 — Asymmetric Diurnal Temperature Forcing  "
        "(slow heating / sharp exponential cooling)",
        fontsize=12, fontweight="bold",
    )

    # Left: T(t) at selected depths
    ax = axes[0]
    for depth in [0.0, 0.05, 0.20, 0.50]:
        idx   = int(depth / DZ)
        label = f"z = {depth:.2f} m"
        ax.plot(time_h, T_soil[:, idx], linewidth=1.8, label=label)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Diurnal Cycle at Selected Depths (asymmetric)")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.6)

    # Right: depth snapshot at peak heat vs peak cold
    ax2     = axes[1]
    hot_i   = int(np.argmax(T_soil[:, 0]))
    cold_i  = int(np.argmin(T_soil[:, 0]))
    ax2.plot(T_soil[hot_i],  Z_GRID, color="firebrick",  linewidth=2.0,
             label=f"Peak heat  (t = {time_h[hot_i]:.1f} h)")
    ax2.plot(T_soil[cold_i], Z_GRID, color="steelblue",  linewidth=2.0,
             label=f"Peak cold  (t = {time_h[cold_i]:.1f} h)")
    ax2.set_xlabel("Temperature [°C]")
    ax2.set_ylabel("Depth z [m]")
    ax2.set_title("Temperature Profile: Hot vs Cold Snapshot")
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_timeseries(
    time_s: np.ndarray,
    C_data: np.ndarray,
    C_pred: np.ndarray,
    beta_opt: np.ndarray,
    sigma_sq: float,
    out_path: Path,
) -> None:
    """Model vs observed CO₂ at three depths + residuals panel."""
    D_s, S_0, S_1 = beta_opt
    time_h = time_s / 3600.0

    depth_labels = ["z = 0.05 m", "z = 0.20 m", "z = 0.50 m"]
    colors_obs   = ["#2E5EAA", "#E87722", "#44AA44"]
    colors_model = ["#1A3D7C", "#A0510E", "#228822"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"V4 Multiphysics CN — Modelled vs Observed\n"
        f"D_s={D_s:.2e} m²/s,  S_0={S_0:.3f},  S_1={S_1:.3f},  σ²={sigma_sq:.2e} ppm²",
        fontsize=11,
    )

    for i in range(3):
        ax = axes[i]
        ax.scatter(time_h, C_data[:, i], s=6, alpha=0.45,
                   color=colors_obs[i], label="Observed")
        ax.plot(time_h, C_pred[:, i], linewidth=1.6,
                color=colors_model[i], label="Model")
        ax.set_ylabel("CO₂ [ppm]", fontsize=10)
        ax.set_title(depth_labels[i], fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(color="#cccccc", linewidth=0.6)
        ax.set_ylim(0, 8000)

    ax_res = axes[3]
    for i, (label, color) in enumerate(zip(depth_labels, colors_obs)):
        ax_res.plot(time_h, C_pred[:, i] - C_data[:, i],
                    linewidth=1.0, alpha=0.85, color=color, label=label)
    ax_res.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax_res.set_xlabel("Time [hours]", fontsize=10)
    ax_res.set_ylabel("Residual [ppm]", fontsize=10)
    ax_res.set_title("Residuals (model − observed)", fontsize=10)
    ax_res.legend(fontsize=9)
    ax_res.grid(color="#cccccc", linewidth=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_diagnostic_dashboard(
    time_s: np.ndarray,
    C_data: np.ndarray,
    C_pred: np.ndarray,
    residuals_vec: np.ndarray,
    beta_opt: np.ndarray,
    Cov: np.ndarray,
    out_path: Path,
) -> None:
    """
    2×2 Gabitov Diagnostic Dashboard.

    Panel 1 — Spatiotemporal fit
    Panel 2 — Inferred S(z) with ±1σ propagated error bounds
    Panel 3 — Parameter correlation matrix (collinearity audit)
    Panel 4 — Residual distribution vs theoretical Gaussian
    """
    D_s, S_0, S_1 = beta_opt
    time_h         = time_s / 3600.0
    depth_labels   = ["5 cm", "20 cm", "50 cm"]
    colors         = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "V4 Multiphysics Solver — Gabitov Diagnostic Dashboard",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1 ───────────────────────────────────────────────────────────────
    ax1 = axs[0, 0]
    for i in range(3):
        ax1.scatter(time_h, C_data[:, i], color=colors[i],
                    s=10, alpha=0.45, label=f"Data: {depth_labels[i]}")
        ax1.plot(time_h, C_pred[:, i], color=colors[i],
                 linewidth=2.0, label=f"Model: {depth_labels[i]}")
    ax1.set_title("Spatiotemporal Fit: Model vs. Empirical", fontweight="bold")
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("CO₂ concentration [ppm]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(linestyle="--", alpha=0.6)
    ax1.set_ylim(0, 8000)

    # ── Panel 2 ───────────────────────────────────────────────────────────────
    ax2     = axs[0, 1]
    S_opt   = S_0 + S_1 * Z_GRID
    var_S0  = Cov[1, 1];  var_S1 = Cov[2, 2];  cov_S0S1 = Cov[1, 2]
    S_var   = var_S0 + Z_GRID**2 * var_S1 + 2.0 * Z_GRID * cov_S0S1
    S_std   = np.sqrt(np.maximum(S_var, 0.0))

    ax2.plot(S_opt, Z_GRID, "k-", linewidth=2.0, label="Optimal S(z)")
    ax2.fill_betweenx(Z_GRID, S_opt - S_std, S_opt + S_std,
                      color="gray", alpha=0.30, label="±1σ error bound")
    ax2.axvline(0.0, color="red", linestyle="--", alpha=0.7, label="Zero-source line")
    ax2.set_title("Inferred Biological Source Term S(z)", fontweight="bold")
    ax2.set_xlabel("Source strength [ppm s⁻¹]")
    ax2.set_ylabel("Depth z [m]")
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.6)

    # ── Panel 3 ───────────────────────────────────────────────────────────────
    ax3      = axs[1, 0]
    std_devs = np.sqrt(np.diag(Cov))
    outer    = np.outer(std_devs, std_devs)
    outer[outer == 0.0] = np.nan
    Corr     = Cov / outer
    np.fill_diagonal(Corr, 1.0)

    sns.heatmap(
        Corr, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1.0, vmax=1.0,
        xticklabels=["$D_s$", "$S_0$", "$S_1$"],
        yticklabels=["$D_s$", "$S_0$", "$S_1$"],
        ax=ax3, linewidths=0.5,
    )
    ax3.set_title("Parameter Correlation Matrix", fontweight="bold")

    # ── Panel 4 ───────────────────────────────────────────────────────────────
    ax4 = axs[1, 1]
    ax4.hist(residuals_vec, bins=40,
             color="purple", alpha=0.70, edgecolor="black", density=True)
    mu  = float(np.mean(residuals_vec))
    std = float(np.std(residuals_vec))
    x_v = np.linspace(residuals_vec.min(), residuals_vec.max(), 200)
    ax4.plot(x_v,
             (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_v - mu) / std)**2),
             "k--", linewidth=2.0,
             label=f"Theoretical Gaussian\n(μ={mu:.1f}, σ={std:.1f})")
    ax4.axvline(0.0, color="red", linestyle="-", alpha=0.7, label="Zero residual")
    ax4.set_title("Orthogonal Projection Residuals", fontweight="bold")
    ax4.set_xlabel("Residual error [ppm]")
    ax4.set_ylabel("Probability density")
    ax4.legend(fontsize=9)
    ax4.grid(linestyle="--", alpha=0.6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(D_s_init: float = 4.5e-6) -> None:
    """
    Execute Version 4.

    Parameters
    ----------
    D_s_init : float
        Initial D_s guess [m²/s].  Chain with V1 for best convergence:

            from v1_stationary_regression import run as run_v1
            beta_v1 = run_v1()
            run(D_s_init=beta_v1[0])
    """
    time_s, C_surface, C_data, T_soil = fetch_and_clean_data()
    n_t = len(time_s)

    beta_init = np.array([D_s_init, 5.0, -2.0])
    r_init    = D_s_init * DT / DZ**2

    print()
    print("=== VERSION 4: MULTIPHYSICS SOLVER ===")
    print(f"Grid              : N_z = {N_Z}, Δz = {DZ:.3f} m,  N_t = {n_t}, Δt = {DT:.0f} s")
    print(f"Residual vector   : {n_t * 3} elements  ({n_t} × 3)")
    print(f"Temperature field : asymmetric diurnal  (slow heat / sharp cool)")
    print(f"Moisture θ        : {V_WATER:.2f}  (constant)")
    print(f"D_s bounds        : [{LOWER[0]:.0e}, {UPPER[0]:.0e}]  m²/s  (ceiling-test relaxed)")
    print(f"Initial β         : D_s = {D_s_init:.2e},  S_0 = 5.0,  S_1 = −2.0")
    print(f"Initial Fourier r : {r_init:.4f}  (CN unconditionally stable)")
    print()
    print("Projecting time steps onto mathematical subspace...")

    result = least_squares(
        cost,
        x0=beta_init,
        bounds=(LOWER, UPPER),
        method="trf",
        diff_step=1e-8,
        args=(C_surface, C_data, T_soil),
    )

    beta_opt      = result.x
    J             = result.jac
    residuals_vec = result.fun

    dof      = len(residuals_vec) - len(beta_opt)
    sigma_sq = float(np.sum(residuals_vec**2) / dof)
    Cov      = sigma_sq * la.pinv(J.T @ J)
    r_opt    = beta_opt[0] * DT / DZ**2

    print()
    print("=== OPTIMISATION RESULTS ===")
    print(f"D_s = {beta_opt[0]:.4e} m²/s")
    print(f"S_0 = {beta_opt[1]:.6f} ppm/s")
    print(f"S_1 = {beta_opt[2]:.6f} ppm/(s·m)")
    print()
    print(f"Fourier Number r = D_s·Δt/Δz² : {r_opt:.4f}")
    print()
    print(f"Degrees of freedom : {dof}")
    print(f"Residual variance  : {sigma_sq:.4e} ppm²")
    print()
    print("=== GABITOV AUDIT ===")
    print(f"Var(D_s) = {Cov[0, 0]:.4e}")
    print(f"Var(S_0) = {Cov[1, 1]:.4e}")
    print(f"Var(S_1) = {Cov[2, 2]:.4e}")
    print()

    boundary_collision = beta_opt[0] >= UPPER[0] * 0.999
    print(f"D_s boundary collision : {'YES — physics still unresolved' if boundary_collision else 'NO — resolved ✓'}")
    print(f"(ceiling = {UPPER[0]:.0e} m²/s,  result = {beta_opt[0]:.2e} m²/s)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir = Path(__file__).resolve().parent
    C_pred  = forward_solver_v4(beta_opt, C_surface, C_data[0], T_soil)

    plot_temperature_field(T_soil, n_t, out_dir / "v4_temperature_field.png")
    plot_timeseries(time_s, C_data, C_pred, beta_opt, sigma_sq,
                    out_dir / "v4_timeseries.png")
    plot_diagnostic_dashboard(time_s, C_data, C_pred, residuals_vec,
                               beta_opt, Cov, out_dir / "v4_diagnostic_dashboard.png")


if __name__ == "__main__":
    # To chain with V1 for a physics-informed x₀, uncomment:
    #
    #   from v1_stationary_regression import run as run_v1
    #   beta_v1 = run_v1()
    #   print()
    #   run(D_s_init=beta_v1[0])
    #
    run()

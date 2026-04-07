"""
Version 3: Thermo-Coupled Crank-Nicolson Inverse Solver
========================================================

Extension of Version 2 with an explicit abiotic sink term driven by
temperature-dependent CO₂ solubility (Henry's Law).

Governing PDE (multiphase coupling)
-------------------------------------
    ∂C_gas/∂t = D_s · ∂²C_gas/∂z² + S_bio(z) − S_abiotic(z, t)

Abiotic sink (Henry's Law + chain rule)
-----------------------------------------
    S_abiotic(z, t) ≈ θ · C_gas · dK_H(T)/dt

where K_H(T) is the dimensionless Henry's solubility constant:

    K_H(T) = K_H_298 · exp( ΔH_sol · (1/T − 1/298.15) )

When the soil cools (dT/dt < 0), K_H rises, CO₂ dissolves into pore water,
and S_abiotic > 0 (net gas removal).

Step 1 (this script)
---------------------
    Temperature is synthesised as a 24-hour sine wave with depth-decaying
    amplitude.  Moisture θ is held constant.  This proves that the abiotic
    sink resolves the D_s boundary collision observed in Version 2.

Step 2 (future work)
---------------------
    Replace ``build_mock_temperature()`` with a function that loads the
    actual Decagon 5TM sensor data from the Oracle pipeline or the
    Jupyter notebook CSV export.

Parameters: β = [D_s, S_0, S_1]
    Bounds:  D_s  ∈ [1×10⁻⁷, 1×10⁻⁵]   m²/s
             S_0  ∈ [0,       10     ]   ppm/s
             S_1  ∈ [−100,     0     ]   ppm/(s·m)

Output files (saved in calculations/)
---------------------------------------
    v3_diagnostic_dashboard.png  — 2×2 Gabitov audit panels
    v3_timeseries.png            — model vs observed at all three depths
    v3_temperature_field.png     — synthetic diurnal temperature forcing

Usage
-----
    python v3_thermo_coupled.py
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
from preprocess import build_mock_dataframe, preprocess

# ── Physical grid ──────────────────────────────────────────────────────────────
L: float           = 1.0
N_Z: int           = 101
DZ: float          = L / (N_Z - 1)
Z_GRID: np.ndarray = np.linspace(0.0, L, N_Z)

SENSOR_DEPTHS: np.ndarray = np.array([0.05, 0.20, 0.50])
SENSOR_IDX: np.ndarray    = np.array([int(d / DZ) for d in SENSOR_DEPTHS])

# ── Temporal grid ──────────────────────────────────────────────────────────────
DT: float = 900.0     # Δt [s]

# ── Parameter bounds: β = [D_s, S_0, S_1] ─────────────────────────────────────
LOWER = [1.0e-7,   0.0, -100.0]
UPPER = [1.0e-5,  10.0,    0.0]

# ── Thermodynamic constants ────────────────────────────────────────────────────
V_WATER: float  = 0.15      # Volumetric water content θ  [m³_water / m³_soil]
DH_SOL: float   = -2400.0   # Van 't Hoff enthalpy of solution for CO₂  [K]
K_H_298: float  = 0.8317    # Dimensionless Henry's constant at 298.15 K  [-]
# Molar density of air at ~15 °C, 1 atm ≈ 40.87 mol m⁻³
MOLAR_DENSITY_AIR: float = 40.87   # mol m⁻³


# ── Synthetic temperature field ────────────────────────────────────────────────

def build_mock_temperature(n_t: int) -> np.ndarray:
    """
    Synthesise a diurnal soil temperature field for prototype testing (Step 1).

    Physics:
        T(z, t) = T_mean + A(z) · sin(ω·t − π/2)

    Where:
        ω       = 2π / 86400 s⁻¹   (24-hour cycle)
        T_mean  = 28 °C
        A(z)    = 10 · exp(−2z)     surface amplitude 10 °C, decays with depth
        phase   = −π/2              hottest at t = 6 h (3 PM if t=0 is midnight)

    Parameters
    ----------
    n_t : int
        Number of time steps.

    Returns
    -------
    T : np.ndarray, shape (n_t, N_Z)
        Soil temperature [°C] at every grid node and time step.
    """
    omega   = 2.0 * np.pi / 86400.0
    t_sec   = np.arange(n_t) * DT                # (n_t,)
    amp     = 10.0 * np.exp(-2.0 * Z_GRID)       # (N_Z,)  — depth-decaying amplitude

    # Vectorised outer product: avoids the double for-loop from the lecture
    T = 28.0 + amp[np.newaxis, :] * np.sin(
        omega * t_sec[:, np.newaxis] - 0.5 * np.pi
    )                                              # (n_t, N_Z)
    return T


# ── Henry's Law abiotic sink ───────────────────────────────────────────────────

def calculate_S_abiotic(
    T_curr: np.ndarray,
    T_next: np.ndarray,
    C_curr_gas: np.ndarray,
    theta: float = V_WATER,
) -> np.ndarray:
    """
    Volumetric abiotic CO₂ sink due to Henry's Law solubility change over Δt.

    Derivation
    ----------
        S_abiotic = θ · C_gas · dK_H/dt
                  ≈ θ · C_gas · (K_H(T_next) − K_H(T_curr)) / Δt

    When soil cools (T_next < T_curr), K_H rises, more CO₂ dissolves,
    S_abiotic > 0 (positive = removal from gas phase).

    Unit chain
    ----------
        C_curr_gas  [ppm]                     — parts per million by volume
        C_gas_mol   [mol m⁻³]  = C [ppm] × 1e-6 × MOLAR_DENSITY_AIR
        K_H         [−]        — dimensionless (C_aq / C_gas)
        dK_H/dt     [s⁻¹]
        S_umol      [mol m⁻³ s⁻¹] = θ · C_gas_mol · dK_H/dt
        return      [ppm s⁻¹] = S_umol / MOLAR_DENSITY_AIR × 1e6

    Parameters
    ----------
    T_curr      : (N_Z,) Temperature at current step  [°C].
    T_next      : (N_Z,) Temperature at next step     [°C].
    C_curr_gas  : (N_Z,) Current gas-phase CO₂        [ppm].
    theta       : float  Volumetric water content      [m³/m³].

    Returns
    -------
    S_abiotic : (N_Z,) sink term in ppm s⁻¹  (positive = net removal).
    """
    T_curr_K = T_curr + 273.15
    T_next_K = T_next + 273.15

    # Van 't Hoff temperature dependence of K_H
    K_H_curr = K_H_298 * np.exp(DH_SOL * (1.0 / T_curr_K - 1.0 / 298.15))
    K_H_next = K_H_298 * np.exp(DH_SOL * (1.0 / T_next_K - 1.0 / 298.15))

    # Convert CO₂ concentration from ppm to mol m⁻³
    C_gas_mol = C_curr_gas * 1.0e-6 * MOLAR_DENSITY_AIR   # mol m⁻³

    # Rate of change of Henry's constant  [s⁻¹]
    dK_dt = (K_H_next - K_H_curr) / DT

    # Volumetric sink  [mol m⁻³ s⁻¹]
    S_vol = theta * C_gas_mol * dK_dt

    # Convert back to ppm s⁻¹ for the CN right-hand-side vector
    return (S_vol / MOLAR_DENSITY_AIR) * 1.0e6


# ── Crank-Nicolson matrices ────────────────────────────────────────────────────

def _build_cn_matrices(D_s: float):
    """
    Tridiagonal A (implicit) and B (explicit) matrices for CN time-stepping.
    Boundary rows set via lil_matrix assignment (safe sparse API).
    """
    r = (D_s * DT) / (2.0 * DZ**2)

    A = diags([-r, 1.0 + 2.0*r, -r], [-1, 0, 1], shape=(N_Z, N_Z), format="lil")
    B = diags([ r, 1.0 - 2.0*r,  r], [-1, 0, 1], shape=(N_Z, N_Z), format="csr")

    # Top — Dirichlet
    A[0, :]   = 0.0
    A[0, 0]   = 1.0

    # Bottom — Neumann zero-flux
    A[N_Z - 1, :]          = 0.0
    A[N_Z - 1, N_Z - 1]   =  1.0
    A[N_Z - 1, N_Z - 2]   = -1.0

    return A.tocsr(), B


# ── Thermodynamically-coupled forward model ────────────────────────────────────

def forward_solver_thermo(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data_t0: np.ndarray,
    T_soil: np.ndarray,
) -> np.ndarray:
    """
    CN time-stepping with biology + Henry's Law abiotic sink.

    At each step the RHS vector is:

        d = B · Cⁿ  +  (S_bio − S_abiotic(n)) · Δt

    Parameters
    ----------
    beta       : [D_s, S_0, S_1]
    C_surface  : (n_t,)    Dirichlet BC at z = 0  [ppm].
    C_data_t0  : (3,)      Sensor readings at t = 0 for cubic-spline IC [ppm].
    T_soil     : (n_t, N_Z) Soil temperature field  [°C].

    Returns
    -------
    predictions : (n_t, 3) Model CO₂ at sensor nodes for every time step.
    """
    D_s, S_0, S_1 = beta
    n_t = len(C_surface)

    S_bio: np.ndarray = S_0 + S_1 * Z_GRID      # (N_Z,) biological source [ppm s⁻¹]

    A, B = _build_cn_matrices(D_s)

    # Initial condition — cubic spline through 4 physical sensor anchors
    cs = CubicSpline(
        [0.0,          0.05,          0.20,          0.50,          L],
        [C_surface[0], C_data_t0[0], C_data_t0[1], C_data_t0[2], C_data_t0[2]],
    )
    C = cs(Z_GRID)

    predictions = np.empty((n_t, 3))
    predictions[0] = C[SENSOR_IDX]

    for n in range(1, n_t):
        # Explicit half of Crank-Nicolson
        rhs = B.dot(C) + S_bio * DT

        # Abiotic sink: subtract CO₂ dissolved into pore water during Δt
        S_ab = calculate_S_abiotic(T_soil[n - 1], T_soil[n], C)
        rhs -= S_ab * DT

        # Enforce BCs
        rhs[0]       = C_surface[n]
        rhs[N_Z - 1] = 0.0

        C = spsolve(A, rhs)
        predictions[n] = C[SENSOR_IDX]

    return predictions


# ── Cost function ──────────────────────────────────────────────────────────────

def cost(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data: np.ndarray,
    T_soil: np.ndarray,
) -> np.ndarray:
    """
    Flattened (n_t × 3,) residual vector for scipy.optimize.least_squares.
    """
    pred = forward_solver_thermo(beta, C_surface, C_data[0], T_soil)
    return (pred - C_data).flatten()


# ── Diagnostic plots ───────────────────────────────────────────────────────────

def plot_temperature_field(T_soil: np.ndarray, n_t: int, out_path: Path) -> None:
    """
    Figure: synthetic diurnal temperature forcing used in Step 1.

    Shows temperature vs time at four selected depths, and one snapshot
    of the full depth profile at t = peak-heat and t = peak-cold.
    """
    time_h = np.arange(n_t) * DT / 3600.0

    selected_depths = [0.0, 0.05, 0.20, 0.50]
    selected_idx    = [int(d / DZ) for d in selected_depths]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("V3 — Synthetic Diurnal Temperature Forcing (Step 1 Mock)",
                 fontsize=12, fontweight="bold")

    # Left: T(t) at selected depths
    ax = axes[0]
    for depth, idx in zip(selected_depths, selected_idx):
        label = f"z = {depth:.2f} m" if depth > 0 else "z = 0.00 m (surface)"
        ax.plot(time_h, T_soil[:, idx], linewidth=1.8, label=label)
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title("Diurnal Cycle at Selected Depths")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.6)

    # Right: depth profiles at hottest and coldest moments
    ax2 = axes[1]
    hot_idx  = int(np.argmax(T_soil[:, 0]))    # peak surface heat
    cold_idx = int(np.argmin(T_soil[:, 0]))    # peak surface cold
    ax2.plot(T_soil[hot_idx],  Z_GRID, color="firebrick",
             linewidth=2.0, label=f"Peak heat  (t = {time_h[hot_idx]:.1f} h)")
    ax2.plot(T_soil[cold_idx], Z_GRID, color="steelblue",
             linewidth=2.0, label=f"Peak cold  (t = {time_h[cold_idx]:.1f} h)")
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
    """
    Figure: model vs observed at all three sensor depths + residuals panel.
    """
    D_s, S_0, S_1 = beta_opt
    time_h = time_s / 3600.0

    depth_labels = ["z = 0.05 m", "z = 0.20 m", "z = 0.50 m"]
    colors_obs   = ["#2E5EAA", "#E87722", "#44AA44"]
    colors_model = ["#1A3D7C", "#A0510E", "#228822"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"V3 Thermo-Coupled CN — Modelled vs Observed\n"
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
    2×2 Gabitov Diagnostic Dashboard for V3.

    Panel 1 — Spatiotemporal fit
    Panel 2 — Inferred S(z) with ±1σ propagated bounds
    Panel 3 — Parameter correlation matrix
    Panel 4 — Residual distribution vs theoretical Gaussian
    """
    D_s, S_0, S_1 = beta_opt
    time_h = time_s / 3600.0

    depth_labels = ["5 cm", "20 cm", "50 cm"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "V3 Thermo-Coupled Inverse Solver — Gabitov Diagnostic Dashboard",
        fontsize=14, fontweight="bold",
    )

    # ── Panel 1: spatiotemporal fit ───────────────────────────────────────────
    ax1 = axs[0, 0]
    for i in range(3):
        ax1.scatter(time_h, C_data[:, i],
                    color=colors[i], s=10, alpha=0.45,
                    label=f"Data: {depth_labels[i]}")
        ax1.plot(time_h, C_pred[:, i],
                 color=colors[i], linewidth=2.0,
                 label=f"Model: {depth_labels[i]}")
    ax1.set_title("Spatiotemporal Fit: Model vs. Empirical", fontweight="bold")
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("CO₂ concentration [ppm]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(linestyle="--", alpha=0.6)
    ax1.set_ylim(0, 8000)

    # ── Panel 2: S(z) with ±1σ propagated error ───────────────────────────────
    ax2 = axs[0, 1]
    S_opt    = S_0 + S_1 * Z_GRID
    var_S0   = Cov[1, 1]
    var_S1   = Cov[2, 2]
    cov_S0S1 = Cov[1, 2]
    S_var    = var_S0 + Z_GRID**2 * var_S1 + 2.0 * Z_GRID * cov_S0S1
    S_std    = np.sqrt(np.maximum(S_var, 0.0))

    ax2.plot(S_opt, Z_GRID, "k-", linewidth=2.0, label="Optimal S(z)")
    ax2.fill_betweenx(Z_GRID, S_opt - S_std, S_opt + S_std,
                      color="gray", alpha=0.30, label="±1σ error bound")
    ax2.axvline(0.0, color="red", linestyle="--", alpha=0.7,
                label="Zero-source line")
    ax2.set_title("Inferred Biological Source Term S(z)", fontweight="bold")
    ax2.set_xlabel("Source strength [ppm s⁻¹]")
    ax2.set_ylabel("Depth z [m]")
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.grid(linestyle="--", alpha=0.6)

    # ── Panel 3: parameter correlation matrix ─────────────────────────────────
    ax3 = axs[1, 0]
    std_devs = np.sqrt(np.diag(Cov))
    outer = np.outer(std_devs, std_devs)
    outer[outer == 0.0] = np.nan
    Corr = Cov / outer
    np.fill_diagonal(Corr, 1.0)

    sns.heatmap(
        Corr, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1.0, vmax=1.0,
        xticklabels=["$D_s$", "$S_0$", "$S_1$"],
        yticklabels=["$D_s$", "$S_0$", "$S_1$"],
        ax=ax3, linewidths=0.5,
    )
    ax3.set_title("Parameter Correlation Matrix", fontweight="bold")

    # ── Panel 4: residual distribution ────────────────────────────────────────
    ax4 = axs[1, 1]
    ax4.hist(residuals_vec, bins=40,
             color="purple", alpha=0.70, edgecolor="black", density=True)
    mu  = float(np.mean(residuals_vec))
    std = float(np.std(residuals_vec))
    x_val   = np.linspace(residuals_vec.min(), residuals_vec.max(), 200)
    p_gauss = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x_val - mu) / std)**2
    )
    ax4.plot(x_val, p_gauss, "k--", linewidth=2.0,
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run(D_s_init: float = 4.5e-6) -> None:
    """
    Execute Version 3.

    Step 1 uses a synthetic diurnal temperature field.
    For Step 2, replace ``build_mock_temperature()`` with a loader that
    reads the Decagon 5TM sensor data from Oracle / the Jupyter pipeline.

    Parameters
    ----------
    D_s_init : float
        Initial guess for D_s [m²/s].  For the best starting point, chain
        with Version 1:

            from v1_stationary_regression import run as run_v1
            beta_v1 = run_v1()
            run(D_s_init=beta_v1[0])
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    raw = build_mock_dataframe()
    df  = preprocess(raw)

    n_t       = len(df)
    C_surface = df["C_surface"].to_numpy()
    C_data    = df[["C_z005", "C_z020", "C_z050"]].to_numpy()   # (n_t, 3)

    # ── Step 1: synthesised temperature field ─────────────────────────────────
    T_soil = build_mock_temperature(n_t)    # (n_t, N_Z)

    beta_init = np.array([D_s_init, 5.0, -2.0])
    r_init    = D_s_init * DT / DZ**2

    print("=== VERSION 3: THERMO-COUPLED CRANK-NICOLSON SOLVER ===")
    print(f"Grid              : N_z = {N_Z}, Δz = {DZ:.3f} m,  N_t = {n_t}, Δt = {DT:.0f} s")
    print(f"Retained rows     : {n_t}  (dropped {288 - n_t} in preprocessing)")
    print(f"Residual vector   : {n_t * 3} elements  ({n_t} × 3)")
    print(f"Temperature field : Step 1 synthetic diurnal  "
          f"(T_mean=28°C, A_surface=10°C, depth-decaying)")
    print(f"Moisture θ        : {V_WATER:.2f}  (constant)")
    print(f"Initial β         : D_s = {D_s_init:.2e} m²/s,  "
          f"S_0 = 5.0 ppm/s,  S_1 = −2.0 ppm/(s·m)")
    print(f"Initial Fourier r : {r_init:.4f}  (full-step; CN is unconditionally stable)")
    print()
    print(f"Projecting {n_t} time steps onto mathematical subspace...")

    # ── Optimisation ──────────────────────────────────────────────────────────
    result = least_squares(
        cost,
        x0=beta_init,
        bounds=(LOWER, UPPER),
        method="trf",
        diff_step=1e-8,
        args=(C_surface, C_data, T_soil),
    )

    beta_opt      = result.x
    J             = result.jac          # (n_t*3, 3)
    residuals_vec = result.fun

    # ── Gabitov Audit ─────────────────────────────────────────────────────────
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
    print("=== PARAMETER COVARIANCE DIAGONAL (Gabitov Audit) ===")
    print(f"Var(D_s) = {Cov[0, 0]:.4e}")
    print(f"Var(S_0) = {Cov[1, 1]:.4e}")
    print(f"Var(S_1) = {Cov[2, 2]:.4e}")
    print()
    print("NOTE: D_s boundary collision resolved if D_s < 1×10⁻⁵ m²/s.")
    print("      Correlation(D_s, S_0) exposed in dashboard Panel 3.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    out_dir  = Path(__file__).resolve().parent / "out" / "V3"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_s   = df["time_s"].to_numpy()
    C_pred   = forward_solver_thermo(beta_opt, C_surface, C_data[0], T_soil)

    plot_temperature_field(
        T_soil, n_t,
        out_dir / "v3_temperature_field.png",
    )
    plot_timeseries(
        time_s, C_data, C_pred, beta_opt, sigma_sq,
        out_dir / "v3_timeseries.png",
    )
    plot_diagnostic_dashboard(
        time_s, C_data, C_pred, residuals_vec, beta_opt, Cov,
        out_dir / "v3_diagnostic_dashboard.png",
    )


if __name__ == "__main__":
    # ── Standalone (default x₀) ────────────────────────────────────────────────
    # To chain with V1 for a physics-informed initial guess, uncomment:
    #
    #   from v1_stationary_regression import run as run_v1
    #   beta_v1 = run_v1()
    #   print()
    #   run(D_s_init=beta_v1[0])
    #
    run()

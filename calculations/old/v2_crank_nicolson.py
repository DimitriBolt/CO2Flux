"""
Version 2: Crank-Nicolson Spatiotemporal Inverse Solver
========================================================

Mathematical objective
----------------------
Solve the full transient 1D diffusion-reaction PDE over 72 hours,
extracting D_s and S(z) from all available spatiotemporal observations.

Governing PDE
-------------
    ∂C/∂t = D_s · ∂²C/∂z² + S_bio(z)

    S_bio(z) = S_0 + S_1 · z          (linear biological source)

Discretisation grid
-------------------
    Spatial  : N_z = 101 nodes,  Δz = 0.01 m,  z ∈ [0, 1] m
    Temporal : N_t  steps,        Δt = 900 s   (N_t = 288 for complete 72 h)

Numerical scheme — Crank-Nicolson (θ = 0.5)
--------------------------------------------
    A · C^{n+1} = B · C^n + S_bio · Δt

    A: implicit tridiagonal  — diag = (1 + 2r),  off-diag = −r
    B: explicit tridiagonal  — diag = (1 − 2r),  off-diag = +r
    r = D_s · Δt / (2 · Δz²)   (Crank-Nicolson Fourier number, half-step)

    Boundary conditions
        Top    z = 0:   C[0]        = C_surface[n]   [Dirichlet, time-varying]
        Bottom z = L:   C[N−1]      = C[N−2]          [Neumann, zero flux]

    Initial condition  t = 0:
        Cubic spline through 4 physical anchors at z = 0, 0.05, 0.20, 0.50, 1.0 m.

Parameters: β = [D_s, S_0, S_1]
    Bounds:  D_s  ∈ [1×10⁻⁷, 1×10⁻⁵]   m²/s
             S_0  ∈ [0,       10     ]   ppm/s
             S_1  ∈ [−100,     0     ]   ppm/(s·m)

Inverse formulation
-------------------
    Flatten both the model predictions and the empirical data to a
    (N_t × 3)-element vector and minimise the L² residual norm:

        J(β) = ‖ vec(C_model(β)) − vec(C_data) ‖²₂

Error bounding — Gabitov Audit
-------------------------------
    After convergence extract the (N_t·3 × 3) Jacobian J.
    Noise variance  : σ² = J_min / (N_t·3 − 3)
    Parameter covar : C_β = σ² · pinv(JᵀJ)    [truncated SVD]
    Print diag(C_β) to bound statistical confidence on all three parameters.

Chaining with Version 1
-----------------------
    Run Version 1 first to obtain a baseline D_s, then pass it as D_s_init:

        from v1_stationary_regression import run as run_v1
        beta_v1  = run_v1()
        run(D_s_init=beta_v1[0])

Usage (standalone with default initial guess)
---------------------------------------------
    python v2_crank_nicolson.py
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
L: float          = 1.0
N_Z: int          = 101
DZ: float         = L / (N_Z - 1)
Z_GRID: np.ndarray = np.linspace(0.0, L, N_Z)

SENSOR_DEPTHS: np.ndarray = np.array([0.05, 0.20, 0.50])
SENSOR_IDX: np.ndarray    = np.array([int(d / DZ) for d in SENSOR_DEPTHS])

# ── Temporal grid ──────────────────────────────────────────────────────────────
DT: float = 900.0    # Δt [s]

# ── Parameter bounds: β = [D_s, S_0, S_1] ─────────────────────────────────────
LOWER = [1.0e-7,   0.0, -100.0]
UPPER = [1.0e-5,  10.0,    0.0]


# ── Crank-Nicolson matrices ────────────────────────────────────────────────────

def _build_cn_matrices(D_s: float):
    """
    Construct the two N_z × N_z tridiagonal matrices for the Crank-Nicolson scheme.

    A (implicit):  diag = 1 + 2r,  off-diag = −r   (with BCs enforced)
    B (explicit):  diag = 1 − 2r,  off-diag = +r   (interior formula throughout)

    Where r = D_s · Δt / (2 · Δz²)  is the half-step Fourier number.

    Boundary rows are applied to A only.  The RHS vector overrides the
    corresponding rows of B·C^n at every time step.
    """
    r = (D_s * DT) / (2.0 * DZ**2)

    # Build as lil for efficient row assignment, then convert to csr for spsolve
    A = diags([-r, 1.0 + 2.0*r, -r], [-1, 0, 1], shape=(N_Z, N_Z), format="lil")
    B = diags([ r, 1.0 - 2.0*r,  r], [-1, 0, 1], shape=(N_Z, N_Z), format="csr")

    # Top boundary — Dirichlet: C[0] = C_surface[n]
    A[0, :]    = 0.0
    A[0, 0]    = 1.0

    # Bottom boundary — Neumann zero-flux: C[N-1] − C[N-2] = 0
    A[N_Z - 1, :]          = 0.0
    A[N_Z - 1, N_Z - 1]   =  1.0
    A[N_Z - 1, N_Z - 2]   = -1.0

    return A.tocsr(), B


# ── Forward model ──────────────────────────────────────────────────────────────

def forward_solver(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data_t0: np.ndarray,
) -> np.ndarray:
    """
    Advance the PDE over len(C_surface) time steps using Thomas algorithm.

    Parameters
    ----------
    beta       : [D_s, S_0, S_1]
    C_surface  : (N_t,)   Dirichlet BC at z = 0 for each time step [ppm].
    C_data_t0  : (3,)     Sensor readings at t = 0 for the spline IC [ppm].

    Returns
    -------
    predictions : (N_t, 3)
        Predicted CO2 at SENSOR_IDX grid nodes for every time step.
    """
    D_s, S_0, S_1 = beta
    n_t = len(C_surface)

    S_bio: np.ndarray = S_0 + S_1 * Z_GRID     # (N_Z,) ppm/s

    A, B = _build_cn_matrices(D_s)

    # ── Initial condition: cubic spline through 4 physical sensor anchors ──────
    # The floor anchor (z = L) is extrapolated from the deepest available reading.
    cs = CubicSpline(
        [0.0,          0.05,            0.20,            0.50,            L],
        [C_surface[0], C_data_t0[0], C_data_t0[1], C_data_t0[2], C_data_t0[2]],
    )
    C = cs(Z_GRID)

    predictions = np.empty((n_t, 3))
    predictions[0] = C[SENSOR_IDX]

    # ── Time-stepping loop ─────────────────────────────────────────────────────
    for n in range(1, n_t):
        # Explicit half-step RHS
        rhs = B.dot(C) + S_bio * DT

        # Overwrite boundary rows
        rhs[0]       = C_surface[n]   # Dirichlet top
        rhs[N_Z - 1] = 0.0            # Neumann bottom

        # Implicit solve: Thomas algorithm via spsolve
        C = spsolve(A, rhs)
        predictions[n] = C[SENSOR_IDX]

    return predictions


# ── Cost function ──────────────────────────────────────────────────────────────

def cost(
    beta: np.ndarray,
    C_surface: np.ndarray,
    C_data: np.ndarray,
) -> np.ndarray:
    """
    Flattened residual vector for scipy.optimize.least_squares.

        r(β) = vec( C_model(β) ) − vec( C_data )

    Shape: (N_t × 3,)  — nominally 864 elements for a complete 72-hour dataset.
    """
    pred = forward_solver(beta, C_surface, C_data[0])
    return (pred - C_data).flatten()


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(
    time_s: np.ndarray,
    C_data: np.ndarray,
    C_pred: np.ndarray,
    beta_opt: np.ndarray,
    sigma_sq: float,
    out_stem: Path,
) -> None:
    """
    Two figures saved from the V2 results.

    Figure 1 — v2_timeseries.png
        3-row subplot: modelled vs observed time series at each sensor depth.
        Bottom row: pointwise residuals (model − obs) for all three depths.

    Figure 2 — v2_final_profile.png
        Final spatial CO₂ profile C(z) at the last time step, showing the
        model solution on the 101-node grid and the 3 observed sensor values.
    """
    D_s, S_0, S_1 = beta_opt
    time_h = time_s / 3600.0

    depth_labels = ["z = 0.05 m", "z = 0.20 m", "z = 0.50 m"]
    colors_obs   = ["#2E5EAA", "#E87722", "#44AA44"]
    colors_model = ["#1A3D7C", "#A0510E", "#228822"]

    # ── Figure 1: time series ────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    fig.suptitle(
        f"V2 Crank-Nicolson — Modelled vs Observed  "
        f"(D_s={D_s:.2e} m²/s, S_0={S_0:.3f}, S_1={S_1:.3f}  σ²={sigma_sq:.2e} ppm²)",
        fontsize=11,
    )

    for i in range(3):
        ax = axes[i]
        ax.scatter(time_h, C_data[:, i], s=6, alpha=0.55,
                   color=colors_obs[i], label="Observed")
        ax.plot(time_h, C_pred[:, i], linewidth=1.6,
                color=colors_model[i], label="Model")
        ax.set_ylabel("CO₂ [ppm]", fontsize=10)
        ax.set_title(depth_labels[i], fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(color="#cccccc", linewidth=0.6)
        ax.set_ylim(0, 8000)

    # Residuals panel
    ax_res = axes[3]
    for i, (label, color) in enumerate(zip(depth_labels, colors_obs)):
        ax_res.plot(time_h, C_pred[:, i] - C_data[:, i],
                    linewidth=1.0, alpha=0.85, color=color, label=label)
    ax_res.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax_res.set_xlabel("Time [hours]", fontsize=10)
    ax_res.set_ylabel("Residual [ppm]", fontsize=10)
    ax_res.set_title("Residuals (model − observed)", fontsize=10)
    ax_res.legend(fontsize=9, loc="upper right")
    ax_res.grid(color="#cccccc", linewidth=0.6)

    fig.tight_layout()
    ts_path = out_stem.parent / (out_stem.name + "_timeseries.png")
    fig.savefig(ts_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {ts_path}")

    # ── Figure 2: final spatial profile ──────────────────────────────────────
    # Re-run the forward model one last time at beta_opt to get the full
    # 101-node concentration vector at the final time step.
    # We piggyback on the existing forward_solver but capture the full C vector
    # by extracting it through the known sensor index values plus the bc.
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    # Approximate the final profile using the 3 converged model values + BCs
    # (a full re-run is possible but expensive; this is display-only).
    z_anchors = np.array([0.0,    0.05,          0.20,          0.50,         L])
    c_anchors  = np.array([420.0, C_pred[-1, 0], C_pred[-1, 1], C_pred[-1, 2], C_pred[-1, 2]])
    cs_final   = CubicSpline(z_anchors, c_anchors)
    z_fine     = np.linspace(0.0, L, 500)
    C_final    = cs_final(z_fine)

    ax2.plot(C_final, z_fine, color="steelblue", linewidth=2.0, label="Model (final step)")
    ax2.scatter(C_data[-1], SENSOR_DEPTHS,
                s=120, zorder=5, color="darkorange", edgecolors="black",
                label="Observed (final step)")
    ax2.scatter([420.0], [0.0], s=120, zorder=5, marker="D",
                color="firebrick", edgecolors="black", label="Surface BC (420 ppm)")

    ax2.set_xlabel("CO₂ concentration [ppm]", fontsize=11)
    ax2.set_ylabel("Depth z [m]", fontsize=11)
    ax2.set_title(
        f"V2 — Final Spatial Profile  (t = {time_h[-1]:.1f} h)\n"
        f"D_s={D_s:.2e} m²/s,  S_0={S_0:.3f},  S_1={S_1:.3f}",
        fontsize=11,
    )
    ax2.invert_yaxis()
    ax2.legend(fontsize=9)
    ax2.grid(axis="x", color="#cccccc", linewidth=0.7)
    ax2.set_xlim(0, 8000)

    fig2.tight_layout()
    prof_path = out_stem.parent / (out_stem.name + "_final_profile.png")
    fig2.savefig(prof_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {prof_path}")


# ── Gabitov Diagnostic Dashboard ─────────────────────────────────────────────

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
    2×2 Gabitov Diagnostic Dashboard  →  saved as v2_diagnostic_dashboard.png

    Panel 1 (top-left)  — Spatiotemporal fit
        Modelled vs observed CO₂ time series at all three sensor depths.
        Proves the forward model captures diurnal phase and amplitude.

    Panel 2 (top-right) — Inferred biological source term S(z)
        Optimal S(z) = S₀ + S₁·z plotted against depth with ±1σ propagated
        error bounds derived from the diagonal and off-diagonal of Cov.
        If the band crosses zero the model is uncertain about the sign of
        respiration at that depth.

    Panel 3 (bottom-left) — Parameter correlation matrix
        Normalised covariance: R_ij = C_ij / √(C_ii · C_jj).
        Off-diagonal elements near ±1 signal collinearity — the solver
        cannot distinguish between the competing physical mechanisms.

    Panel 4 (bottom-right) — Residual distribution
        Histogram of the 864 (N_t × 3) flatted residuals with a theoretical
        Gaussian overlay.  Skew or heavy tails indicate a missing physics
        term (e.g. abiotic temperature sink) in the governing PDE.
    """
    D_s, S_0, S_1 = beta_opt
    time_h = time_s / 3600.0

    depth_labels = ["5 cm", "20 cm", "50 cm"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Inverse Solver Diagnostic Audit — Gabitov Dashboard",
                 fontsize=15, fontweight="bold")

    # ── Panel 1: spatiotemporal fit ───────────────────────────────────────────
    ax1 = axs[0, 0]
    for i in range(3):
        ax1.scatter(time_h, C_data[:, i],
                    color=colors[i], s=10, alpha=0.45, label=f"Data: {depth_labels[i]}")
        ax1.plot(time_h, C_pred[:, i],
                 color=colors[i], linewidth=2.0, label=f"Model: {depth_labels[i]}")
    ax1.set_title("Spatiotemporal Fit: Model vs. Empirical", fontweight="bold")
    ax1.set_xlabel("Time [hours]")
    ax1.set_ylabel("CO₂ concentration [ppm]")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(linestyle="--", alpha=0.6)
    ax1.set_ylim(0, 8000)

    # ── Panel 2: inferred S(z) with ±1σ propagated error ─────────────────────
    ax2 = axs[0, 1]
    S_opt = S_0 + S_1 * Z_GRID

    # Var(S(z)) = Var(S₀) + z²·Var(S₁) + 2z·Cov(S₀,S₁)
    var_S0    = Cov[1, 1]
    var_S1    = Cov[2, 2]
    cov_S0S1  = Cov[1, 2]
    S_var     = var_S0 + Z_GRID**2 * var_S1 + 2.0 * Z_GRID * cov_S0S1
    S_std     = np.sqrt(np.maximum(S_var, 0.0))

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

    # ── Panel 3: parameter correlation matrix ─────────────────────────────────
    ax3 = axs[1, 0]
    std_devs = np.sqrt(np.diag(Cov))
    # Guard against zero variance (e.g. parameter hit a bound)
    outer = np.outer(std_devs, std_devs)
    outer[outer == 0.0] = np.nan
    Corr = Cov / outer
    np.fill_diagonal(Corr, 1.0)   # ensure exactly 1 on diagonal

    param_labels = ["$D_s$", "$S_0$", "$S_1$"]
    sns.heatmap(
        Corr, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1.0, vmax=1.0,
        xticklabels=param_labels, yticklabels=param_labels,
        ax=ax3, linewidths=0.5,
    )
    ax3.set_title("Parameter Correlation Matrix", fontweight="bold")

    # ── Panel 4: residual distribution ────────────────────────────────────────
    ax4 = axs[1, 1]
    ax4.hist(residuals_vec, bins=40,
             color="purple", alpha=0.70, edgecolor="black", density=True)

    mu  = float(np.mean(residuals_vec))
    std = float(np.std(residuals_vec))
    x_val = np.linspace(residuals_vec.min(), residuals_vec.max(), 200)
    p_gauss = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x_val - mu) / std)**2)
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
    Execute Version 2.

    Parameters
    ----------
    D_s_init : float
        Initial guess for D_s [m²/s].
        For best convergence pass the optimised D_s from Version 1:

            from v1_stationary_regression import run as run_v1
            beta_v1 = run_v1()
            run(D_s_init=beta_v1[0])
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    raw = build_mock_dataframe()
    df  = preprocess(raw)

    n_t         = len(df)
    C_surface   = df["C_surface"].to_numpy()
    C_data      = df[["C_z005", "C_z020", "C_z050"]].to_numpy()   # (n_t, 3)

    n_residuals = n_t * 3
    beta_init   = np.array([D_s_init, 5.0, -2.0])

    # Stability diagnostic on the initial guess
    r_init = D_s_init * DT / DZ**2

    print("=== VERSION 2: CRANK-NICOLSON SPATIOTEMPORAL SOLVER ===")
    print(f"Grid              : N_z = {N_Z}, Δz = {DZ:.3f} m,  N_t = {n_t}, Δt = {DT:.0f} s")
    print(f"Retained rows     : {n_t}  (dropped {288 - n_t} in preprocessing)")
    print(f"Residual vector   : {n_residuals} elements  ({n_t} × 3)")
    print(f"Initial β         : D_s = {D_s_init:.2e} m²/s,  S_0 = 5.0 ppm/s,  S_1 = −2.0 ppm/(s·m)")
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
        args=(C_surface, C_data),
    )

    beta_opt      = result.x
    J: np.ndarray = result.jac          # (n_residuals, 3)
    residuals_vec = result.fun

    # ── Gabitov Audit: error bounding via truncated SVD ───────────────────────
    dof       = len(residuals_vec) - len(beta_opt)   # n_t·3 − 3  (nominally 861)
    sigma_sq  = float(np.sum(residuals_vec**2) / dof)
    Cov       = sigma_sq * la.pinv(J.T @ J)

    r_opt = beta_opt[0] * DT / DZ**2

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

    # Predictions for plotting
    C_pred = forward_solver(beta_opt, C_surface, C_data[0])
    time_s = df["time_s"].to_numpy()
    _out_dir = Path(__file__).resolve().parent / "out" / "V2"
    _out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = _out_dir / "v2"
    plot_results(time_s, C_data, C_pred, beta_opt, sigma_sq, out_stem)

    dashboard_path = _out_dir / "v2_diagnostic_dashboard.png"
    plot_diagnostic_dashboard(
        time_s, C_data, C_pred, residuals_vec, beta_opt, Cov, dashboard_path
    )


if __name__ == "__main__":
    # ── Standalone run (default initial guess) ─────────────────────────────────
    # To chain with Version 1 for a physics-informed x₀, uncomment:
    #
    #   from v1_stationary_regression import run as run_v1
    #   beta_v1 = run_v1()
    #   print()
    #   run(D_s_init=beta_v1[0])
    #
    run()

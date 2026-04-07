"""
Version 1: Stationary Analytical Regression  (Baseline MVP)
============================================================

Mathematical objective
----------------------
Solve the inverse problem by projecting the 72-hour time-averaged sensor
observations onto the analytical steady-state family of solutions for the
1D diffusion-reaction ODE.

Governing equation (steady-state, ∂C/∂t → 0)
----------------------------------------------
    D_s · d²C/dz² + S_bio = 0

Analytical solution
-------------------
Integrating twice and applying the two boundary conditions:

    BC Top    C(0)          = C_surface   [Dirichlet]
    BC Bottom dC/dz|_{z=L}  = 0           [Neumann — zero flux at basalt floor]

yields the quadratic family:

    C(z; β) = C_surface + (S_bio / D_s) · (L·z − z²/2)

Parameters: β = [D_s, S_bio]

Inverse formulation
-------------------
    Input  : C̄_data — 72-hour mean concentrations at z = 0.05, 0.20, 0.50 m
    Cost   : J(β) = Σᵢ [ C(zᵢ; β) − C̄_data,i ]²
    Method : scipy.optimize.least_squares  (TRF, bounded)

Gabitov Audit
-------------
    2 parameters fitted to 3 data points → DOF = 1.
    Covariance is expected to be large.
    Use the optimised D_s as the initial guess (x₀) for Version 2 only.

Usage
-----
    python v1_stationary_regression.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import build_mock_dataframe, preprocess

# ── Physical domain ────────────────────────────────────────────────────────────
L: float = 1.0                                   # Total basalt depth [m]
SENSOR_DEPTHS: np.ndarray = np.array([0.05, 0.20, 0.50])  # z-positions [m]

# ── Parameter bounds: β = [D_s, S_bio] ────────────────────────────────────────
LOWER = [1.0e-7,   0.0]    # D_s [m²/s] min;  S_bio [ppm/s] min
UPPER = [1.0e-5, 1.0e3]    # D_s [m²/s] max;  S_bio [ppm/s] max

# ── Initial guess ──────────────────────────────────────────────────────────────
# Physically motivated: D_s ≈ 4.5×10⁻⁶ m²/s from prior basalt measurements.
# S_bio is chosen so that C(0.05) − C_surface ≈ 4500 ppm → S_bio/D_s ≈ 92 000.
BETA_INIT: np.ndarray = np.array([4.5e-6, 0.42])


# ── Analytical model ───────────────────────────────────────────────────────────

def analytical_profile(
    z: np.ndarray,
    D_s: float,
    S_bio: float,
    C_surface: float,
) -> np.ndarray:
    """
    Steady-state quadratic concentration profile.

    Derived from:   D_s · d²C/dz² = −S_bio
    Integrated twice with Dirichlet top and Neumann bottom BCs.

        C(z) = C_surface + (S_bio / D_s) · (L·z − z²/2)

    Parameters
    ----------
    z         : Evaluation depths [m].
    D_s       : Effective diffusivity [m²/s].
    S_bio     : Constant biological source term [ppm/s].
    C_surface : Atmospheric boundary concentration [ppm].

    Returns
    -------
    Predicted CO2 concentrations at each z [ppm].
    """
    return C_surface + (S_bio / D_s) * (L * z - 0.5 * z**2)


def residual(
    beta: np.ndarray,
    z: np.ndarray,
    C_mean: np.ndarray,
    C_surf: float,
) -> np.ndarray:
    """
    Residual vector (length 3) consumed by scipy.optimize.least_squares.

        r(β) = C_model(z; β) − C̄_data
    """
    D_s, S_bio = beta
    return analytical_profile(z, D_s, S_bio, C_surf) - C_mean


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(
    beta_opt: np.ndarray,
    Cov: np.ndarray,
    C_mean: np.ndarray,
    C_surf_mean: float,
    out_path: Path,
) -> None:
    """
    Figure 1 — V1 Stationary Analytical Regression
    -----------------------------------------------
    Left panel  : Fitted quadratic CO2 profile C(z) with ±2σ propagated
                  uncertainty band, overlaid on the 3 observed sensor means
                  and the surface BC point.
    Right panel : Residuals at the 3 sensor depths (model − data).

    Axes follow the physical orientation: depth z increases downward.
    """
    D_s, S_bio = beta_opt

    z_fine = np.linspace(0.0, L, 500)
    C_fit  = analytical_profile(z_fine, D_s, S_bio, C_surf_mean)

    # Propagate parameter covariance to profile uncertainty
    # ∂C/∂D_s   = −k(z) · S_bio / D_s²
    # ∂C/∂S_bio =  k(z) / D_s
    # where k(z) = L·z − z²/2
    k         = L * z_fine - 0.5 * z_fine**2
    dC_dDs    = -k * S_bio / D_s**2
    dC_dSbio  =  k / D_s
    grad      = np.stack([dC_dDs, dC_dSbio], axis=1)    # (500, 2)
    var_C     = np.einsum("ni,ij,nj->n", grad, Cov, grad)
    sigma_C   = np.sqrt(np.maximum(var_C, 0.0))

    C_model_at_sensors = analytical_profile(SENSOR_DEPTHS, D_s, S_bio, C_surf_mean)
    residuals_at_sensors = C_model_at_sensors - C_mean

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: profile ─────────────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_betweenx(
        z_fine, C_fit - 2 * sigma_C, C_fit + 2 * sigma_C,
        alpha=0.20, color="steelblue", label="±2σ propagated uncertainty",
    )
    ax.plot(C_fit, z_fine, color="steelblue", linewidth=2.0,
            label=f"Fitted profile  D_s={D_s:.2e} m²/s,  S_bio={S_bio:.4f} ppm/s")
    ax.scatter(C_mean, SENSOR_DEPTHS,
               s=120, zorder=5, color="darkorange", edgecolors="black",
               label="Observed 72-h means")
    ax.scatter([C_surf_mean], [0.0],
               s=120, zorder=5, marker="D", color="firebrick", edgecolors="black",
               label=f"Surface BC  (z=0,  {C_surf_mean:.0f} ppm)")

    ax.set_xlabel("CO₂ concentration [ppm]", fontsize=11)
    ax.set_ylabel("Depth z [m]  (0 = surface, 1 = basalt floor)", fontsize=11)
    ax.set_title("V1 — Stationary Analytical Regression\nFitted CO₂ Profile", fontsize=12)
    ax.invert_yaxis()   # surface at top, floor at bottom
    ax.legend(fontsize=9)
    ax.grid(axis="x", color="#cccccc", linewidth=0.7)

    # ── Right: residuals ──────────────────────────────────────────────────────
    ax2 = axes[1]
    labels = [f"z={d:.2f} m" for d in SENSOR_DEPTHS]
    colors = ["#2E5EAA", "#E87722", "#44AA44"]
    ax2.barh(
        labels, residuals_at_sensors,
        color=colors, edgecolor="black", linewidth=0.8,
    )
    ax2.axvline(0, color="black", linewidth=1.0)
    ax2.set_xlabel("Residual [ppm]\n(model − data)", fontsize=11)
    ax2.set_title("Residuals", fontsize=12)
    ax2.grid(axis="x", color="#cccccc", linewidth=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> np.ndarray:
    """
    Execute Version 1.

    Returns
    -------
    beta_opt : np.ndarray, shape (2,)
        [D_s_opt, S_bio_opt].  Pass D_s_opt as x₀[0] to Version 2.
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    raw = build_mock_dataframe()
    df  = preprocess(raw)

    C_surf_mean: float    = float(df["C_surface"].mean())
    C_mean: np.ndarray    = df[["C_z005", "C_z020", "C_z050"]].mean().to_numpy()

    print("=== VERSION 1: STATIONARY ANALYTICAL REGRESSION ===")
    print(f"Retained rows      : {len(df)}")
    print(f"C_surface (mean)   : {C_surf_mean:.1f} ppm")
    print(f"C̄_data             : z=0.05 m → {C_mean[0]:.1f} ppm  |  "
          f"z=0.20 m → {C_mean[1]:.1f} ppm  |  z=0.50 m → {C_mean[2]:.1f} ppm")
    print()

    # ── Optimisation ──────────────────────────────────────────────────────────
    result = least_squares(
        residual,
        x0=BETA_INIT,
        bounds=(LOWER, UPPER),
        method="trf",
        diff_step=1e-8,
        args=(SENSOR_DEPTHS, C_mean, C_surf_mean),
    )

    beta_opt: np.ndarray = result.x
    J: np.ndarray        = result.jac      # shape (3, 2)
    residuals_vec        = result.fun

    # ── Gabitov Audit: covariance from pseudo-inverse ─────────────────────────
    dof: int      = len(residuals_vec) - len(beta_opt)   # 3 − 2 = 1
    sigma_sq: float = float(np.sum(residuals_vec**2) / dof)
    Cov: np.ndarray = sigma_sq * la.pinv(J.T @ J)

    print("=== OPTIMISATION RESULTS ===")
    print(f"D_s   = {beta_opt[0]:.4e} m²/s")
    print(f"S_bio = {beta_opt[1]:.6f} ppm/s")
    print()
    print(f"Degrees of freedom : {dof}")
    print(f"Residual variance  : {sigma_sq:.4e} ppm²")
    print()
    print("=== PARAMETER COVARIANCE DIAGONAL (Gabitov Audit) ===")
    print(f"Var(D_s)   = {Cov[0, 0]:.4e}")
    print(f"Var(S_bio) = {Cov[1, 1]:.4e}")
    print()
    print("Note: DOF = 1 → large covariance is expected.")
    print(f"      Feed D_s = {beta_opt[0]:.4e} m²/s into Version 2 as x₀[0].")
    print()

    out_dir = Path(__file__).resolve().parent / "out" / "V1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v1_profile_fit.png"
    plot_results(beta_opt, Cov, C_mean, C_surf_mean, out_path)

    return beta_opt


if __name__ == "__main__":
    run()

"""
Global Data Preprocessing — Sanity Filters
===========================================
Applied before both Version 1 and Version 2.

Input contract
--------------
DataFrame with exactly these columns:
    time_s     — elapsed seconds (float, monotonically increasing)
    C_surface  — atmospheric CO2 at z = 0 m          [ppm]
    C_z005     — basalt sensor at z = 0.05 m          [ppm]
    C_z020     — basalt sensor at z = 0.20 m          [ppm]
    C_z050     — basalt sensor at z = 0.50 m          [ppm]

Polling rate: Δt = 900 s (15 minutes).

Filters (applied in order)
---------------------------
 1. Forward-fill NaN, limit = 4 intervals (= 1 hour). Drop rows where NaN persists.
 2. 413 ppm anomaly mask  — hardware artefact: subsurface < 425 ppm → NaN.
 3. Cueva thermodynamic mask — physically impossible downward CO2 flux → NaN.
 4. Drop rows where any subsurface sensor remains NaN after filters 2–3.

After preprocessing every remaining row is fully valid for all three subsurface
sensors; there are no NaN values in C_z005 / C_z020 / C_z050.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Cueva thermodynamic mask constants ─────────────────────────────────────────
# D_s reference prior: used only to convert ppm/m gradients to μmol m⁻² s⁻¹.
# The true D_s is unknown at preprocessing time; this is a physical prior estimate.
_D_S_REF: float = 4.5e-6        # m²/s  — typical value for basalt in this system
# Ideal gas at 15 °C, 1 atm: n_air = P/(R·T) ≈ 42.3 mol m⁻³ → 42.3 μmol m⁻³ ppm⁻¹
_N_AIR: float = 42.3             # μmol m⁻³ ppm⁻¹

CUEVA_FLUX_THRESHOLD: float = 0.50   # μmol m⁻² s⁻¹
FFILL_LIMIT: int = 4                 # intervals at Δt = 900 s  (= 1 hour)
ANOMALY_PPM: float = 425.0           # below this, a subsurface reading is an artefact

SENSOR_COLS = ["C_surface", "C_z005", "C_z020", "C_z050"]
SUBSURFACE_COLS = ["C_z005", "C_z020", "C_z050"]


# ── Mock data ──────────────────────────────────────────────────────────────────

def build_mock_dataframe(n_time: int = 288, dt: float = 900.0) -> pd.DataFrame:
    """
    72-hour mock dataset for prototype testing.

    Parameters
    ----------
    n_time : int   Number of time steps (288 = 72 h at Δt = 900 s).
    dt     : float Polling interval in seconds.

    Returns
    -------
    DataFrame with columns [time_s, C_surface, C_z005, C_z020, C_z050].
    All values are physically plausible; preprocessing will retain all rows.
    """
    t = np.arange(0, n_time * dt, dt)
    return pd.DataFrame({
        "time_s":    t,
        "C_surface": np.full(n_time, 420.0),
        "C_z005":    4985.0 + 10.0 * np.sin(2.0 * np.pi * t / 86400.0),
        "C_z020":    6880.0 +  5.0 * np.sin(2.0 * np.pi * t / 86400.0),
        "C_z050":    np.full(n_time, 6500.0),
    })


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the four global sanity filters.

    Parameters
    ----------
    df : DataFrame
        Raw input with columns [time_s, C_surface, C_z005, C_z020, C_z050].

    Returns
    -------
    DataFrame
        Cleaned subset of rows.  All subsurface sensor columns are non-NaN.
        The time_s column is preserved; row index is reset to 0-based.
    """
    df = df.copy()

    # ── Filter 1: Forward-fill (limit = 4 intervals), drop persistent NaN ──────
    df[SENSOR_COLS] = df[SENSOR_COLS].ffill(limit=FFILL_LIMIT)
    df = df.dropna(subset=SENSOR_COLS).reset_index(drop=True)

    # ── Filter 2: 413 ppm anomaly mask ─────────────────────────────────────────
    # A subsurface CO2 reading below 425 ppm violates the minimum thermodynamic
    # baseline and is a hardware artefact (the "413 ppm anomaly").
    for col in SUBSURFACE_COLS:
        df.loc[df[col] < ANOMALY_PPM, col] = np.nan

    # ── Filter 3: Cueva thermodynamic mask ─────────────────────────────────────
    # Computing the discrete spatial gradient between adjacent sensor pairs.
    # Sign convention: z increases downward into the basalt (z = 0 at surface).
    # Normal respiration → C increases with depth → dC/dz > 0 → flux_down < 0.
    # Anomalous inversion → C decreases with depth → dC/dz < 0 → flux_down > 0.
    #
    # flux_down [μmol m⁻² s⁻¹] = −D_s_ref · (ΔC/Δz) · n_air
    #
    # If flux_down > CUEVA_FLUX_THRESHOLD the gradient implies an impossible
    # downward CO2 flux; mask the deeper sensor for that row.
    for deeper, shallower, dz in [
        ("C_z005", "C_surface", 0.05),   # Δz from 0.00 m to 0.05 m
        ("C_z020", "C_z005",    0.15),   # Δz from 0.05 m to 0.20 m
        ("C_z050", "C_z020",    0.30),   # Δz from 0.20 m to 0.50 m
    ]:
        gradient  = (df[deeper] - df[shallower]) / dz     # ppm m⁻¹
        flux_down = -_D_S_REF * gradient * _N_AIR          # μmol m⁻² s⁻¹
        df.loc[flux_down > CUEVA_FLUX_THRESHOLD, deeper] = np.nan

    # ── Filter 4: Drop rows where any subsurface sensor is still masked ─────────
    df = df.dropna(subset=SUBSURFACE_COLS).reset_index(drop=True)

    return df

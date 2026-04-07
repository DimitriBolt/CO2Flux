"""
Averaged Depth Profile  +  48-Hour Window  —  8 Verticals
==========================================================

For each of the 8 ideal-period verticals this script produces **two panels
side by side**:

  LEFT  — Mean depth profile (CO₂ vs depth) averaged over the full ideal
           period, with ±1σ horizontal error bars across time.
           Depth on Y-axis (0 at surface, deepest sensor at bottom),
           CO₂ on X-axis.  Thin lines connect the three sensor nodes.

  RIGHT — 48-hour raw time series from the centre of the ideal period.
           Three depth bands as separate lines with ±1σ shading computed
           as the rolling std over a 3-hour window (shows diurnal scatter).

Data validation notes
---------------------
* Mean and std are computed only over the ideal-period window.  No data
  outside the ideal period contributes.
* Mock data random seeds are deterministic (one per vertical) so the figures
  are reproducible.
* For real Oracle data set  USE_REAL_DATA = True  in
  ideal_period_timeline.py  (shared toggle).

Output
------
    calculations/out/depth_profile/depth_profile_48h.png
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ── Import shared catalogue, loader and USE_REAL_DATA toggle ─────────────────
_CALC_DIR = Path(__file__).resolve().parent
if str(_CALC_DIR) not in sys.path:
    sys.path.insert(0, str(_CALC_DIR))

from ideal_period_timeline import (   # noqa: E402
    VERTICALS,
    USE_REAL_DATA,
    fetch_all_data,
    GLOBAL_START,
    GLOBAL_END,
)

# ── Visual constants ──────────────────────────────────────────────────────────
DEPTH_COLORS = ["#2E5EAA", "#E87722", "#44AA44"]
ROLLING_STD_HOURS = 3       # window for diurnal scatter estimate in right panel
Y_MAX_PPM       = 9000


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ideal_window(v: dict, series: pd.Series) -> pd.Series:
    """Return the slice of `series` that falls inside the ideal period."""
    mask = (series.index >= v["start"]) & (series.index <= v["end"])
    return series[mask]


def _depth_profile_stats(v: dict, series_dict: dict) -> dict:
    """
    For one vertical compute per-depth mean and std over the ideal period.

    Returns
    -------
    {depths: [d1, d2, d3] in cm,
     means:  [m1, m2, m3] in ppm,
     stds:   [s1, s2, s3] in ppm}
    """
    d1, d2, d3 = v["depths"]
    results = {"depths": [], "means": [], "stds": []}
    for key in [f"{d1}cm", f"{d2}cm", f"{d3}cm"]:
        win = _ideal_window(v, series_dict[key]).dropna()
        results["depths"].append(int(key.replace("cm", "")))
        results["means"].append(float(win.mean()) if len(win) > 0 else np.nan)
        results["stds"].append(float(win.std(ddof=1)) if len(win) > 1 else 0.0)
    return results


def _centre_48h(v: dict, series_dict: dict) -> dict[str, pd.Series]:
    """
    Extract the 48 hours centred on the midpoint of the ideal period.
    Falls back to the whole ideal period if shorter than 48 h.
    """
    mid    = v["start"] + (v["end"] - v["start"]) / 2
    t0     = mid - pd.Timedelta(hours=24)
    t1     = mid + pd.Timedelta(hours=24)
    # clamp to ideal period bounds
    t0     = max(t0, v["start"])
    t1     = min(t1, v["end"])

    out = {}
    for key, s in series_dict.items():
        win = s[(s.index >= t0) & (s.index <= t1)].copy()
        # rolling std for shading — 3-hour window at hourly resolution
        roll_std = win.rolling(ROLLING_STD_HOURS, center=True, min_periods=1).std()
        out[key] = {"mean": win, "std": roll_std}
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _draw_depth_profile(
    ax: plt.Axes,
    v: dict,
    stats: dict,
) -> None:
    """
    LEFT panel: horizontal error bars on a CO₂ vs depth plot.
    Depth increases downward (y-axis inverted).
    """
    depths = np.array(stats["depths"], dtype=float)
    means  = np.array(stats["means"],  dtype=float)
    stds   = np.array(stats["stds"],   dtype=float)

    # Validate: flag NaN means
    valid = ~np.isnan(means)

    # Scatter + connected line
    ax.plot(
        means[valid], depths[valid],
        color="#333333", linewidth=1.4, zorder=3,
        marker="o", markersize=6, markerfacecolor="white", markeredgewidth=1.5,
    )

    # Horizontal error bars (±1σ)
    for d, m, s, col in zip(depths[valid], means[valid], stds[valid], DEPTH_COLORS):
        ax.errorbar(
            m, d,
            xerr=s,
            fmt="none",
            ecolor=col, elinewidth=2, capsize=4, capthick=1.8,
            zorder=4,
        )
        ax.scatter(m, d, color=col, s=60, zorder=5)

    depth_label = "35 cm" if v["depth_deep_m"] == 0.35 else "50 cm"
    n_hours = (v["end"] - v["start"]).total_seconds() / 3600
    ax.set_title(
        f"{v['label']}  ({depth_label} deep)\n"
        f"mean ± 1σ  over {n_hours:.0f} h ideal period",
        fontsize=8, pad=3,
    )
    ax.set_xlabel("CO₂ [ppm]", fontsize=8)
    ax.set_ylabel("Depth [cm]", fontsize=8)
    ax.set_xlim(0, Y_MAX_PPM)
    ax.set_ylim(max(depths) + 5, -5)   # inverted: surface at top
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.tick_params(labelsize=7)


def _draw_48h_timeseries(
    ax: plt.Axes,
    v: dict,
    windows: dict,
) -> None:
    """
    RIGHT panel: 48-hour time series with rolling ±1σ shading.
    """
    d1, d2, d3 = v["depths"]
    keys_in_order = [f"{d1}cm", f"{d2}cm", f"{d3}cm"]

    for key, col in zip(keys_in_order, DEPTH_COLORS):
        if key not in windows:
            continue
        mean_s = windows[key]["mean"]
        std_s  = windows[key]["std"]
        valid  = mean_s.notna()
        t      = mean_s.index

        ax.plot(
            t[valid], mean_s[valid],
            color=col, linewidth=1.2, label=f"{key} depth",
        )
        ax.fill_between(
            t[valid],
            (mean_s - std_s)[valid],
            (mean_s + std_s)[valid],
            color=col, alpha=0.18,
        )

    t_vals = list(windows.values())[0]["mean"].index
    delta_h = (t_vals[-1] - t_vals[0]).total_seconds() / 3600 if len(t_vals) > 1 else 0

    ax.set_title(
        f"48-hour window  ({t_vals[0].strftime('%Y-%m-%d %H:%M') if len(t_vals) > 0 else '—'} "
        f"→ {t_vals[-1].strftime('%m-%d %H:%M') if len(t_vals) > 0 else '—'})\n"
        f"shading = ±1σ rolling {ROLLING_STD_HOURS} h",
        fontsize=8, pad=3,
    )
    ax.set_xlabel("Time", fontsize=8)
    ax.set_ylabel("CO₂ [ppm]", fontsize=8)
    ax.set_ylim(0, Y_MAX_PPM)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.tick_params(labelsize=7)


def plot_depth_profile_grid(
    data_list: list[dict],
    out_path: Path,
) -> None:
    """
    8-row × 2-column figure.
    Left  = averaged depth profile with ±1σ error bars.
    Right = 48-hour centred time series with rolling std shading.
    """
    nrows = len(VERTICALS)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=2,
        figsize=(16, 4.5 * nrows),
    )
    data_mode = "mock data" if not USE_REAL_DATA else "real Oracle data"
    fig.suptitle(
        "CO₂ Depth Profiles + 48-Hour Window — LEO West Biome\n"
        f"({data_mode}  ·  left: mean±1σ over ideal period  ·  right: 48 h centred on midpoint)",
        fontsize=12, fontweight="bold", y=1.002,
    )

    for i, (v, series_dict) in enumerate(zip(VERTICALS, data_list)):
        ax_left  = axes[i, 0]
        ax_right = axes[i, 1]

        # ── Left: depth profile ───────────────────────────────────────────────
        stats = _depth_profile_stats(v, series_dict)
        _draw_depth_profile(ax_left, v, stats)

        # ── Right: 48-hour window ─────────────────────────────────────────────
        windows = _centre_48h(v, series_dict)
        _draw_48h_timeseries(ax_right, v, windows)

    plt.tight_layout(rect=[0, 0, 1, 0.998])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# DATA VALIDATION — printed summary
# ══════════════════════════════════════════════════════════════════════════════

def print_data_validation(data_list: list[dict], time_index: pd.DatetimeIndex) -> None:
    """
    Print per-vertical counts, means, stds, and NaN fractions to confirm
    that data extraction is correct and only ideal-period data contributes.
    """
    print("\n=== DATA EXTRACTION VALIDATION ===")
    print(f"  {'Vertical':18s} {'Depth':>6s} {'N':>6s} {'NaN%':>6s} {'Mean':>8s} {'Std':>8s}")
    print("  " + "-" * 62)
    for v, series_dict in zip(VERTICALS, data_list):
        d1, d2, d3 = v["depths"]
        for key in [f"{d1}cm", f"{d2}cm", f"{d3}cm"]:
            win     = _ideal_window(v, series_dict[key])
            n_total = len(win)
            n_nan   = int(win.isna().sum())
            valid   = win.dropna()
            n_valid = len(valid)
            pct_nan = 100 * n_nan / n_total if n_total > 0 else 100.0
            mean_v  = valid.mean() if n_valid > 0 else float("nan")
            std_v   = valid.std(ddof=1) if n_valid > 1 else float("nan")
            label   = v["label"] if key == f"{d1}cm" else ""
            print(f"  {label:18s} {key:>6s} {n_valid:>6d} {pct_nan:>5.1f}% "
                  f"{mean_v:>8.1f} {std_v:>8.1f}")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    out_dir  = Path(__file__).resolve().parent / "out" / "depth_profile"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "depth_profile_48h.png"

    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"=== DEPTH PROFILE + 48H WINDOW  ({mode}) ===")

    time_index = pd.date_range(GLOBAL_START, GLOBAL_END, freq="h")
    print(f"  Loading {len(VERTICALS)} verticals on hourly grid …")
    data_list = fetch_all_data(time_index)

    print_data_validation(data_list, time_index)

    print("Generating figure …")
    plot_depth_profile_grid(data_list, out_path)
    print("\nDone.")


if __name__ == "__main__":
    run()

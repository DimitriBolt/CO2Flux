"""
Ensemble-Averaged CO₂ Timeline — Multi-Scale Overview
======================================================

Averages all 8 ideal-period verticals into a single ensemble mean ± 1σ
and plots at three temporal resolutions in a single 3-row figure:

  • Monthly   (MS bins)
  • Weekly    (7-day bins)
  • 72-hour   (3-day bins)

Each row: mean line per depth band + ±1σ shading across verticals.
Gold spans mark the original ideal periods.

Data is masked to ideal-period windows only (NaN elsewhere), so the
ensemble average shows exactly where and how many sensors are present.

Imports catalogue and data loader from ideal_period_timeline.py —
USE_REAL_DATA toggle there controls mock vs Oracle mode.

Output
------
    calculations/averaged_timeline.png
    calculations/golden/goldenslice.csv   ← weekly-resampled ensemble
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

# ── Import shared catalogue + loader from sibling script ─────────────────────
_CALC_DIR = Path(__file__).resolve().parent
if str(_CALC_DIR) not in sys.path:
    sys.path.insert(0, str(_CALC_DIR))

from ideal_period_timeline import (   # noqa: E402
    VERTICALS,
    GLOBAL_START,
    GLOBAL_END,
    USE_REAL_DATA,
    fetch_all_data,
)

# ── Visual constants ──────────────────────────────────────────────────────────
DEPTH_COLORS = {
    "5cm":  "#2E5EAA",
    "20cm": "#E87722",
    "deep": "#44AA44",
}
Y_MAX = 9000


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def build_ensemble(
    data_list: list[dict],
    time_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    For each depth band ("5cm", "20cm", "deep") compute the ensemble mean
    and population std across all 8 verticals at every hourly time step.

    Data outside each vertical's ideal period is set to NaN before stacking,
    so only periods with real/ideal data contribute to the statistics.

    Returns
    -------
    DataFrame with columns mean_5cm, std_5cm, mean_20cm, std_20cm,
    mean_deep, std_deep, count_5cm, count_20cm, count_deep.
    Index is time_index.
    """
    arrays: dict[str, list[np.ndarray]] = {"5cm": [], "20cm": [], "deep": []}

    for v, series_dict in zip(VERTICALS, data_list):
        d1, d2, d3 = v["depths"]
        outside = ~((time_index >= v["start"]) & (time_index <= v["end"]))

        for band, key in [
            ("5cm",  f"{d1}cm"),
            ("20cm", f"{d2}cm"),
            ("deep", f"{d3}cm"),
        ]:
            arr = series_dict[key].to_numpy(dtype=float).copy()
            arr[outside] = np.nan
            arrays[band].append(arr)

    rows: dict[str, np.ndarray] = {}
    for band, arr_list in arrays.items():
        mat   = np.stack(arr_list, axis=1)          # shape (n_t, 8)
        count = np.sum(~np.isnan(mat), axis=1)

        mean = np.nanmean(mat, axis=1)
        std  = np.nanstd(mat, axis=1, ddof=0)       # population std (ddof=0)

        # Where no vertical has data, force NaN
        no_data = count == 0
        mean[no_data] = np.nan
        std[no_data]  = np.nan

        rows[f"mean_{band}"]  = mean
        rows[f"std_{band}"]   = std
        rows[f"count_{band}"] = count.astype(float)

    return pd.DataFrame(rows, index=time_index)


def resample_ensemble(ens: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample to `freq` bins.
    mean columns → arithmetic mean of hourly means within the bin.
    std  columns → RMS of hourly stds within the bin (conservative bound).
    count columns → max count (how many verticals were present at any point).
    """
    mean_cols  = [c for c in ens.columns if c.startswith("mean_")]
    std_cols   = [c for c in ens.columns if c.startswith("std_")]
    count_cols = [c for c in ens.columns if c.startswith("count_")]

    r_mean  = ens[mean_cols].resample(freq).mean()
    r_std   = ens[std_cols].pow(2).resample(freq).mean().pow(0.5)
    r_count = ens[count_cols].resample(freq).max()

    return pd.concat([r_mean, r_std, r_count], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def _draw_row(
    ax: plt.Axes,
    r: pd.DataFrame,
    scale_label: str,
) -> None:
    """
    Draw one panel: mean lines + ±1σ fill_between + gold ideal period spans.
    """
    for band, col in DEPTH_COLORS.items():
        mean  = r[f"mean_{band}"]
        std   = r[f"std_{band}"]
        valid = mean.notna()
        t     = mean.index

        ax.plot(
            t[valid], mean[valid],
            color=col, linewidth=1.8,
            label=f"{band} depth",
        )
        ax.fill_between(
            t[valid],
            (mean - std)[valid],
            (mean + std)[valid],
            color=col, alpha=0.22,
        )

    # Gold ideal-period spans (label only the first one to avoid duplicates)
    for i, v in enumerate(VERTICALS):
        ax.axvspan(
            v["start"], v["end"],
            color="gold", alpha=0.25, zorder=0,
            label="Ideal period" if i == 0 else "_nolegend_",
        )

    ax.set_ylim(0, Y_MAX)
    ax.set_ylabel("CO₂ [ppm]", fontsize=9)
    ax.set_title(
        f"Ensemble mean ± 1σ  (n ≤ 8 verticals)  —  {scale_label} bins",
        fontsize=9, loc="left", pad=3,
    )
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
    ax.tick_params(labelsize=8)


def plot_multi_scale(
    data_list: list[dict],
    time_index: pd.DatetimeIndex,
    out_path: Path,
) -> dict[str, pd.DataFrame]:
    """
    Build 3-row figure (monthly / weekly / 72-hour) with error bands.
    Returns dict of resampled DataFrames keyed by scale label.
    """
    ens = build_ensemble(data_list, time_index)

    scales = [
        ("Monthly",  "MS"),
        ("Weekly",   "7D"),
        ("72-hour",  "3D"),
    ]
    resampled: dict[str, pd.DataFrame] = {
        label: resample_ensemble(ens, freq) for label, freq in scales
    }

    data_mode = "mock data" if not USE_REAL_DATA else "real Oracle data"

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(18, 14))
    fig.suptitle(
        "CO₂ Ensemble Average — LEO West Biome  |  All 8 Verticals\n"
        f"({data_mode}  ·  shading = ±1σ across verticals  ·  gold = ideal periods)",
        fontsize=12, fontweight="bold", y=0.998,
    )

    tick_configs = [
        # (locator,                                  formatter,                     rotation)
        (mdates.MonthLocator(interval=1),            mdates.DateFormatter("%Y-%m"), 45),
        (mdates.WeekdayLocator(byweekday=0, interval=2),
                                                     mdates.DateFormatter("%b %d '%y"), 45),
        (mdates.DayLocator(interval=5),              mdates.DateFormatter("%b %d '%y"), 45),
    ]

    xlim = (GLOBAL_START, GLOBAL_END)

    for ax, (label, freq), (locator, formatter, rot) in zip(
        axes, scales, tick_configs
    ):
        _draw_row(ax, resampled[label], label)
        ax.set_xlim(*xlim)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=rot, ha="right")
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.002, 1.0),
            borderaxespad=0.0,
            fontsize=8,
            framealpha=0.9,
        )

    plt.subplots_adjust(right=0.84, hspace=0.50)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    return resampled


# ══════════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def save_golden_csv(
    weekly_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """
    Save the weekly-resampled ensemble data to golden/goldenslice.csv.

    Columns
    -------
    week_start, mean_5cm, std_5cm, count_5cm,
                mean_20cm, std_20cm, count_20cm,
                mean_deep, std_deep, count_deep
    """
    golden_dir = out_dir / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)
    csv_path = golden_dir / "goldenslice.csv"  # calculations/out/ensemble/golden/

    export = weekly_df.copy()
    export.index.name = "week_start"
    export.reset_index().to_csv(csv_path, index=False, float_format="%.2f")
    print(f"Saved: {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    out_dir  = Path(__file__).resolve().parent / "out" / "ensemble"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "averaged_timeline.png"

    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"=== ENSEMBLE-AVERAGED TIMELINE  ({mode}) ===")
    print(f"  Global window : {GLOBAL_START.date()} → {GLOBAL_END.date()}")

    time_index = pd.date_range(GLOBAL_START, GLOBAL_END, freq="h")
    print(f"  Timeline points: {len(time_index):,} (hourly)\n")

    print("Loading data (masked to ideal periods)...")
    data_list = fetch_all_data(time_index)

    print("Building ensemble statistics...")
    resampled = plot_multi_scale(data_list, time_index, out_path)

    print("Saving weekly golden slice to CSV...")
    save_golden_csv(resampled["Weekly"], out_dir)

    # Quick summary of how many verticals contribute per scale
    for label, df in resampled.items():
        max_n = int(df[[c for c in df.columns if c.startswith("count_")]].max().max())
        non_nan_bins = int(df["mean_5cm"].notna().sum())
        print(f"  {label:8s}: {non_nan_bins} non-empty bins, max n_verticals = {max_n}")

    print("\nDone.")


if __name__ == "__main__":
    run()

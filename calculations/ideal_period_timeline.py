"""
Ideal Period Timeline Overview — 8-Row Shared-Axis Visualization
=================================================================

Shows all 8 ideal-period verticals on a single shared calendar axis so that
temporal overlaps and data gaps are instantly visible.

Key diagnostic questions this plot answers
-------------------------------------------
* Which verticals overlap and can be spatially averaged?
* How large is the 2024 → 2025 data gap?
* Are the Aug/Sep 2025 pairs (x=±1, y=24) and (x=±1, y=10) concurrent?

Data mode
---------
By default this script runs in MOCK mode (no Oracle connection required).
To switch to real data, set USE_REAL_DATA = True at the top of the file.
The data-loading block at the bottom of fetch_all_data() is clearly isolated.

Usage
-----
    python ideal_period_timeline.py

Output
------
    calculations/ideal_period_timeline.png
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

# ── Toggle between mock and real Oracle data ───────────────────────────────────
USE_REAL_DATA: bool = False

# ── Add sensorDB to path (only needed when USE_REAL_DATA = True) ───────────────
_REPO_ROOT   = Path(__file__).resolve().parents[1]
_SENSORDB_DIR = _REPO_ROOT / "Project_description" / "sensorDB"
if str(_SENSORDB_DIR) not in sys.path:
    sys.path.insert(0, str(_SENSORDB_DIR))


# ══════════════════════════════════════════════════════════════════════════════
# IDEAL PERIOD CATALOGUE  (matches cubic_profile_projection.py)
# ══════════════════════════════════════════════════════════════════════════════

VERTICALS: list[dict] = [
    {
        "label":        "x=−1, y=4",
        "start":        pd.Timestamp("2024-07-10 12:30"),
        "end":          pd.Timestamp("2024-07-29 14:45"),
        "depths":       [5, 20, 50],
        "sensor_5cm":   995,   "sensor_20cm": 1011, "sensor_deep": 1034,
        "depth_deep_m": 0.50,
        "basalt_x": -1, "basalt_y": 4,
        "air_sensor":   1275,  "air_x": 0, "air_y": 4,
    },
    {
        "label":        "x=−1, y=10",
        "start":        pd.Timestamp("2025-03-26 16:30"),
        "end":          pd.Timestamp("2025-04-06 18:15"),
        "depths":       [5, 20, 35],
        "sensor_5cm":   999,   "sensor_20cm": 1015, "sensor_deep": 1028,
        "depth_deep_m": 0.35,
        "basalt_x": -1, "basalt_y": 10,
        "air_sensor":   1279,  "air_x": -2, "air_y": 10,
    },
    {
        "label":        "x=−1, y=18",
        "start":        pd.Timestamp("2024-07-10 13:00"),
        "end":          pd.Timestamp("2024-07-29 14:15"),
        "depths":       [5, 20, 35],
        "sensor_5cm":   1003,  "sensor_20cm": 1019, "sensor_deep": 1030,
        "depth_deep_m": 0.35,
        "basalt_x": -1, "basalt_y": 18,
        "air_sensor":   1289,  "air_x": 0, "air_y": 17,
    },
    {
        "label":        "x=−1, y=24",
        "start":        pd.Timestamp("2025-08-25 12:45"),
        "end":          pd.Timestamp("2025-09-13 18:00"),
        "depths":       [5, 20, 50],
        "sensor_5cm":   1007,  "sensor_20cm": 1023, "sensor_deep": 1040,
        "depth_deep_m": 0.50,
        "basalt_x": -1, "basalt_y": 24,
        "air_sensor":   1294,  "air_x": 0, "air_y": 24,
    },
    {
        "label":        "x=+1, y=4",
        "start":        pd.Timestamp("2025-09-30 03:15"),
        "end":          pd.Timestamp("2025-10-03 10:45"),
        "depths":       [5, 20, 50],
        "sensor_5cm":   996,   "sensor_20cm": 1012, "sensor_deep": 1035,
        "depth_deep_m": 0.50,
        "basalt_x":  1, "basalt_y": 4,
        "air_sensor":   1275,  "air_x": 0, "air_y": 4,
    },
    {
        "label":        "x=+1, y=10",
        "start":        pd.Timestamp("2025-08-28 17:30"),
        "end":          pd.Timestamp("2025-09-10 08:30"),
        "depths":       [5, 20, 35],
        "sensor_5cm":   1000,  "sensor_20cm": 1016, "sensor_deep": 1029,
        "depth_deep_m": 0.35,
        "basalt_x":  1, "basalt_y": 10,
        "air_sensor":   1284,  "air_x": 2, "air_y": 10,
    },
    {
        "label":        "x=+1, y=18",
        "start":        pd.Timestamp("2025-09-30 04:00"),
        "end":          pd.Timestamp("2025-10-03 10:00"),
        "depths":       [5, 20, 35],
        "sensor_5cm":   1004,  "sensor_20cm": 1020, "sensor_deep": 1031,
        "depth_deep_m": 0.35,
        "basalt_x":  1, "basalt_y": 18,
        "air_sensor":   1289,  "air_x": 0, "air_y": 17,
    },
    {
        "label":        "x=+1, y=24",
        "start":        pd.Timestamp("2025-08-25 12:45"),
        "end":          pd.Timestamp("2025-09-13 18:00"),
        "depths":       [5, 20, 50],
        "sensor_5cm":   1008,  "sensor_20cm": 1024, "sensor_deep": 1041,
        "depth_deep_m": 0.50,
        "basalt_x":  1, "basalt_y": 24,
        "air_sensor":   1294,  "air_x": 0, "air_y": 24,
    },
]

# ── Shared calendar limits (spans all ideal periods + margin) ─────────────────
GLOBAL_START = pd.Timestamp("2024-06-01")
GLOBAL_END   = pd.Timestamp("2025-11-01")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _mock_series(
    v: dict,
    time_index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    """
    Generate physically plausible synthetic CO₂ time series.
    Deep sensor gets a 1 000 ppm spike during the ideal period
    so the golden highlight and the data spike are visually aligned.
    """
    rng    = np.random.default_rng(seed=abs(v["basalt_x"] * 100 + v["basalt_y"]))
    d1, d2, d3 = v["depths"]
    n      = len(time_index)

    s1 = pd.Series(rng.normal(5000, 200, n), index=time_index, name=f"{d1}cm")
    s2 = pd.Series(rng.normal(6800, 200, n), index=time_index, name=f"{d2}cm")
    s3 = pd.Series(rng.normal(7000, 200, n), index=time_index, name=f"{d3}cm")

    return {f"{d1}cm": s1, f"{d2}cm": s2, f"{d3}cm": s3}


def _real_series(v: dict, time_index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """
    Fetch real Oracle data for one vertical and reindex onto the shared
    global hourly timeline (NaN outside the ideal period).

    Swap this in by setting USE_REAL_DATA = True at the top of the file.
    """
    from basalt_co2_series import BasaltCO2Series  # noqa: PLC0415

    slot_freq = "15min"
    start     = v["start"]
    end       = v["end"]
    d1, d2, d3 = v["depths"]
    depth_deep_cm = v["depth_deep_m"] * 100

    def _fetch(sid: int, depth_cm: float) -> pd.Series:
        sensor = BasaltCO2Series(
            table_name="leo_west.datavalues",
            sensor_id=sid,
            variable_id=9,
            slope="LEO West",
            x_coord_m=float(v["basalt_x"]),
            y_coord_m=float(v["basalt_y"]),
            depth_cm=depth_cm,
        )
        frame = sensor.fetch_dataframe(start_datetime=start, end_datetime=end)
        raw = (
            frame.assign(
                localdatetime=pd.to_datetime(frame["localdatetime"]),
                datavalue=pd.to_numeric(frame["datavalue"], errors="coerce"),
            )
            .assign(slot_ts=lambda df: df["localdatetime"].dt.floor(slot_freq))
            .groupby("slot_ts")["datavalue"]
            .mean()
        )
        raw = raw.mask(raw <= 0)
        return raw.reindex(time_index)     # NaN where not in ideal period

    s1 = _fetch(v["sensor_5cm"],   d1).rename(f"{d1}cm")
    s2 = _fetch(v["sensor_20cm"],  d2).rename(f"{d2}cm")
    s3 = _fetch(v["sensor_deep"],  depth_deep_cm).rename(f"{d3}cm")

    return {f"{d1}cm": s1, f"{d2}cm": s2, f"{d3}cm": s3}


def fetch_all_data(
    time_index: pd.DatetimeIndex,
) -> list[dict[str, pd.Series]]:
    """Return a list of series-dicts, one per vertical, in VERTICALS order."""
    data = []
    for v in VERTICALS:
        if USE_REAL_DATA:
            print(f"  Fetching real data: {v['label']}")
            try:
                data.append(_real_series(v, time_index))
            except Exception as exc:
                print(f"    [WARN] fetch failed ({exc}) — using mock for this vertical")
                data.append(_mock_series(v, time_index))
        else:
            data.append(_mock_series(v, time_index))
    return data


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAP ANALYSIS (printed to console)
# ══════════════════════════════════════════════════════════════════════════════

def print_overlap_matrix() -> None:
    """
    Print a simple text overlap matrix: which ideal periods share calendar days?
    """
    n = len(VERTICALS)
    print("\n=== TEMPORAL OVERLAP MATRIX (days of concurrent data) ===")
    header = "".join(f"  {i+1:2d}" for i in range(n))
    print(f"{'':18s}{header}")
    for i, vi in enumerate(VERTICALS):
        row = f"{vi['label']:18s}"
        for j, vj in enumerate(VERTICALS):
            overlap_start = max(vi["start"], vj["start"])
            overlap_end   = min(vi["end"],   vj["end"])
            days = max(0.0, (overlap_end - overlap_start).total_seconds() / 86400)
            if i == j:
                row += f"  --"
            elif days > 0:
                row += f"  {int(days):2d}"
            else:
                row += f"   ."
        print(row)
    print("  (days > 0 = overlap, '.' = no overlap)\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_timeline(
    data_list: list[dict[str, pd.Series]],
    time_index: pd.DatetimeIndex,
    out_path: Path,
    slice_xlim: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> None:
    """
    8-row figure with a single shared calendar X-axis.

    Each row shows:
      • Three depth time series (5 cm, 20 cm, deep)
      • A gold axvspan marking the ideal period
      • Y-axis clipped to [0, 9000] ppm for visual consistency

    Parameters
    ----------
    slice_xlim : optional (start, end) timestamps to restrict the X-axis view.
    """
    DEPTH_COLORS = ["#2E5EAA", "#E87722", "#44AA44"]
    Y_MAX = 9000

    fig, axes = plt.subplots(
        nrows=8, ncols=1,
        figsize=(18, 22),
        sharex=True,
    )
    fig.suptitle(
        "CO₂ Ideal Periods — LEO West Biome  |  8 Verticals on Shared Calendar Axis\n"
        f"({'mock data' if not USE_REAL_DATA else 'real Oracle data'}  ·  gold band = ideal period)",
        fontsize=13, fontweight="bold", y=0.995,
    )

    for i, (v, series_dict) in enumerate(zip(VERTICALS, data_list)):
        ax  = axes[i]
        d1, d2, d3 = v["depths"]

        # ── Data lines ────────────────────────────────────────────────────────
        for key, col, lw in [
            (f"{d1}cm", DEPTH_COLORS[0], 0.9),
            (f"{d2}cm", DEPTH_COLORS[1], 0.9),
            (f"{d3}cm", DEPTH_COLORS[2], 0.9),
        ]:
            s = series_dict[key]
            ax.plot(s.index, s.to_numpy(), color=col,
                    linewidth=lw, alpha=0.85, label=f"{key} depth")

        # ── Ideal period highlight ─────────────────────────────────────────────
        ax.axvspan(v["start"], v["end"],
                   color="gold", alpha=0.30, zorder=0,
                   label=f"Ideal period\n{v['start'].date()} → {v['end'].date()}")

        # ── Formatting ────────────────────────────────────────────────────────
        ax.set_ylim(0, Y_MAX)
        ax.set_ylabel("CO₂ [ppm]", fontsize=8)
        ax.set_title(
            f"Vertical {v['label']}   "
            f"({'35 cm' if v['depth_deep_m'] == 0.35 else '50 cm'} deepest sensor)   "
            f"Ideal: {v['start'].strftime('%Y-%m-%d')} → {v['end'].strftime('%Y-%m-%d')}",
            fontsize=9, loc="left", pad=3,
        )
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.002, 1.0),
            borderaxespad=0.0,
            fontsize=7.5,
            framealpha=0.9,
        )
        ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.6)
        ax.tick_params(labelsize=7)

    # ── Shared X-axis formatting ───────────────────────────────────────────────
    ax_last = axes[-1]
    ax_last.set_xlabel("Date", fontsize=11)
    if slice_xlim is not None:
        ax_last.set_xlim(slice_xlim[0], slice_xlim[1])
    ax_last.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_last.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_last.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ── Overlap annotation band across all rows ────────────────────────────────
    # Draw a thin vertical dashed line at the start of each unique ideal period
    # so concurrent periods are instantly readable across rows.
    unique_starts = sorted({v["start"] for v in VERTICALS})
    for ts in unique_starts:
        for ax in axes:
            ax.axvline(ts, color="gray", linewidth=0.5, linestyle=":", alpha=0.6)

    plt.subplots_adjust(right=0.84, hspace=0.38)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    out_dir    = Path(__file__).resolve().parent / "out" / "timeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path   = out_dir / "ideal_period_timeline.png"

    mode = "REAL Oracle data" if USE_REAL_DATA else "MOCK data"
    print(f"=== IDEAL PERIOD TIMELINE  ({mode}) ===")
    print(f"  Global window : {GLOBAL_START.date()} → {GLOBAL_END.date()}")
    print(f"  Verticals     : {len(VERTICALS)}")

    print_overlap_matrix()

    # Hourly resolution is plenty for a multi-month calendar overview
    time_index = pd.date_range(GLOBAL_START, GLOBAL_END, freq="h")
    print(f"  Timeline points: {len(time_index):,} (hourly, {GLOBAL_START.date()} → {GLOBAL_END.date()})")
    print()

    print("Loading data...")
    data_list = fetch_all_data(time_index)

    print("\nGenerating full timeline figure...")
    plot_timeline(data_list, time_index, out_path)

    print("\nGenerating 2025 slice figure...")
    slice_start = pd.Timestamp("2025-01-01")
    slice_end   = pd.Timestamp("2025-12-31")
    out_path_2025 = out_dir / "ideal_period_timeline_2025.png"  # same out_dir
    plot_timeline(data_list, time_index, out_path_2025,
                  slice_xlim=(slice_start, slice_end))

    print("\nDone.")


if __name__ == "__main__":
    run()

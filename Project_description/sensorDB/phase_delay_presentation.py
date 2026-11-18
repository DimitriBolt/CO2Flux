"""Express the archived 14-day phase comparison as signed delays in minutes."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from statistics import fmean, median, quantiles

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from short_lag_resolution import (
    PAIRS, SOURCE, file_hash, pair_label, pair_name, plt, read_csv, save_jpg, save_table,
)

INPUT = SOURCE / "phase_comparison.csv"
OUT = SOURCE / "phase_delay_final"
HALF_PERIOD_MIN = 720.0
PERIOD_MIN = 2 * HALF_PERIOD_MIN
RAD_TO_MIN = HALF_PERIOD_MIN / np.pi


def signed_change(phi_15, phi_2h):
    """Shortest signed change (2 h minus 15 min) on [-720, 720) minutes."""
    return ((phi_2h - phi_15 + np.pi) % (2 * np.pi) - np.pi) * RAD_TO_MIN


def archived_delays(path=INPUT):
    saved = read_csv(path)
    parts = []
    for pair in PAIRS:
        label = pair_name(pair)
        group = saved.loc[saved.pair.eq(label)].sort_values("start")
        phi_15 = group[f"{label}_rad_left"].where(group.comparable)
        phi_2h = group[f"{label}_rad_right"].where(group.comparable)
        assert phi_15.dropna().ge(-np.pi).all() and phi_15.dropna().lt(np.pi).all()
        assert phi_2h.dropna().ge(-np.pi).all() and phi_2h.dropna().lt(np.pi).all()
        change = signed_change(phi_15, phi_2h)
        np.testing.assert_allclose(change.abs(), group.delta_rad.abs() * RAD_TO_MIN,
                                   rtol=0, atol=4e-13, equal_nan=True)
        data = group[["pair", "window", "start", "end_exclusive", "center", "comparable"]].copy()
        data["delay_15_min"] = phi_15 * RAD_TO_MIN
        data["delay_2h_min"] = phi_2h * RAD_TO_MIN
        data["signed_change_min"] = change
        data["absolute_change_min"] = change.abs()
        data["wrap_across_grid_estimates"] = (phi_2h - phi_15).abs().gt(np.pi)
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


def summarize(rows):
    result = []
    for pair, window in rows.loc[rows.comparable].groupby("pair", sort=False):
        largest = window.loc[window.absolute_change_min.idxmax()]
        mae = window.absolute_change_min.mean()
        median_abs_15 = window.delay_15_min.abs().median()
        result.append(dict(pair=pair, windows=len(window),
            median_delay_15_min=window.delay_15_min.median(), median_delay_2h_min=window.delay_2h_min.median(),
            median_abs_delay_15_min=median_abs_15, mean_signed_change_min=window.signed_change_min.mean(),
            MAE_min=mae, p95_abs_change_min=window.absolute_change_min.quantile(.95),
            max_abs_change_min=largest.absolute_change_min, max_signed_change_min=largest.signed_change_min,
            MAE_over_median_abs_delay_15=mae/median_abs_15 if median_abs_15 else np.nan,
            max_delay_15_min=largest.delay_15_min, max_delay_2h_min=largest.delay_2h_min,
            max_start=largest.start, max_end_exclusive=largest.end_exclusive, max_center=largest.center,
            cross_branch_windows=int(window.wrap_across_grid_estimates.sum())))
    return pd.DataFrame(result)


def verify_scalar(path, rows, summary):
    """Independent CSV, degree conversion, atan2 and scalar statistics."""
    groups = defaultdict(list)
    with path.open(newline="") as stream:
        for r in csv.DictReader(stream):
            if r["comparable"] != "True":
                continue
            pair = r["pair"]
            a, b = (float(r[f"{pair}_rad_{side}"]) for side in ("left", "right"))
            angle = math.atan2(math.sin(b-a), math.cos(b-a))
            if angle == math.pi:
                angle = -math.pi
            groups[pair].append((r["start"], r["end_exclusive"], math.degrees(a)*4,
                                 math.degrees(b)*4, math.degrees(angle)*4))

    for result in summary.itertuples(index=False):
        reference = sorted(groups[result.pair])
        _, _, a, b, diff = zip(*reference)
        largest = max(reference, key=lambda row: abs(row[-1]))
        absolute = list(map(abs, diff))
        mae = fmean(absolute)
        q95 = quantiles(absolute, n=20, method="inclusive")[-1]
        expected = (median(a), median(b), fmean(diff), mae, q95, largest[-1], mae/median(map(abs, a)))
        actual = (result.median_delay_15_min, result.median_delay_2h_min, result.mean_signed_change_min,
                  result.MAE_min, result.p95_abs_change_min, result.max_signed_change_min,
                  result.MAE_over_median_abs_delay_15)
        assert len(reference) == result.windows
        assert (result.max_start, result.max_end_exclusive) == largest[:2]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=4e-13)
        observed = rows.loc[rows.pair.eq(result.pair) & rows.comparable].sort_values("start")
        assert observed.start.tolist() == [v[0] for v in reference]
        np.testing.assert_allclose(observed[["delay_15_min","delay_2h_min","signed_change_min"]],
                                   np.array([a,b,diff]).T, rtol=0, atol=4e-13)
    got = signed_change(np.deg2rad(np.array([179., -179., 0., 0.])),
                        np.deg2rad(np.array([-179., 179., 180., -180.])))
    np.testing.assert_allclose(got, [8, -8, -720, -720], rtol=0, atol=4e-13)


def plot_segments(ax, dates, values, **style):
    """Split at representative jumps and gaps, keeping both actual endpoints."""
    dates = pd.DatetimeIndex(dates)
    values = np.asarray(values)
    breaks = np.flatnonzero((np.abs(np.diff(values)) >= HALF_PERIOD_MIN)
                            | (np.diff(dates) != pd.Timedelta(days=1))) + 1
    for k, segment in enumerate(np.split(np.arange(len(values)), breaks)):
        current = style if k == 0 else {**style, "label": "_nolegend_"}
        ax.plot(dates[segment], values[segment], **current)


def figure(rows, summary, path):
    fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(11.3, 5.7), layout="constrained")
    fig.suptitle("Daily CO₂ phase delays · LEO West W_x1_y4 (x = 1 m, y = 4 m)")
    extent = 10 * math.ceil(rows.absolute_change_min.max() / 10)
    for col, pair in enumerate(PAIRS):
        data = rows.loc[rows.pair.eq(pair_name(pair))]
        dates = pd.to_datetime(data.center)
        upper, lower = axes[:,col]
        upper.set_title(f"{pair_label(pair)} · GMM222 {pair[0]} → {pair[1]}")
        plot_segments(upper, dates, data.delay_15_min, color="#164e73", lw=1, label="15 min")
        plot_segments(upper, dates, data.delay_2h_min, color="#a34920", lw=1, ls="--", label="2 h")
        plot_segments(lower, dates, data.signed_change_min, color="#354579", lw=.9)
        peak = summary.loc[summary.pair.eq(pair_name(pair))].iloc[0]
        peak_date = pd.Timestamp(peak.max_center)
        for ax in (upper, lower):
            ax.axvline(peak_date, color=".52", lw=.8, ls=":")
            ax.axhline(0, color=".63", lw=.65)
            ax.grid(axis="y",alpha=.2)
        upper.scatter([peak_date, peak_date],[peak.max_delay_15_min, peak.max_delay_2h_min],
                      s=24, color=["#164e73","#a34920"], zorder=4)
        lower.scatter(peak_date, peak.max_signed_change_min, s=25, color="#354579", zorder=4)
        lower.set_xlabel("14-day window center")
        lower.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        lower.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        upper.set_ylim(-740,740)
        upper.set_yticks([-720,-360,0,360,720])
        lower.set_ylim(-extent*1.06,extent*1.06)
    axes[0,0].legend(frameon=False, loc="lower right", ncols=2)
    axes[0,0].set_ylabel("Relative phase delay (min)")
    axes[1,0].set_ylabel("Cyclic change (min)\n2 h minus 15 min")
    save_jpg(fig, path)


def generate():
    before = file_hash(INPUT)
    rows = archived_delays()
    summary = summarize(rows)
    verify_scalar(INPUT, rows, summary)
    OUT.mkdir(parents=True, exist_ok=True)
    save_table(rows, OUT/"phase_delays_by_window.csv")
    save_table(summary, OUT/"phase_delay_summary.csv")
    figure(rows, summary, OUT/"phase_delays_and_signed_change.jpg")
    assert file_hash(INPUT) == before
    report = dict(source=INPUT.name, source_sha256=before, branch_min="[-720, 720)",
        convention="archive F_upper * conjugate(F_lower); new change 2 h minus 15 min",
        comparison="Independent scalar CSV, atan2, degrees*4 and inclusive quantiles",
        verified_windows=int(rows.comparable.sum()), cyclic_boundary_cases_passed=4,
        artifacts_sha256={p.name:file_hash(p) for p in sorted(OUT.iterdir()) if p.name!="validation.json"})
    (OUT/"validation.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    generate()

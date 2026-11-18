"""Approved Stage 0.5 short-lag diagnostic on immutable published inputs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from statistics import median
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "Research_log"
SOURCE = ROOT / "output/temporal_resolution_15min_vs_2h"
OUT = SOURCE / "short_lag_resolution"
START, END = pd.Timestamp("2025-05-01"), pd.Timestamp("2026-10-01")
DAY = pd.Timedelta(days=1)
DEPTH = {996: 5, 1012: 20, 1035: 50}
PAIRS = ((996, 1012), (1012, 1035))
LIMITS = {15: (384, 288), 120: (48, 36)}
TIE = 64 * np.finfo(float).eps


def file_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def pair_name(pair):
    return f"{DEPTH[pair[0]]}_to_{DEPTH[pair[1]]}"


def pair_label(pair):
    return f"{DEPTH[pair[0]]} → {DEPTH[pair[1]]} cm"


def read_csv(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def phase_minutes(angle):
    """Absolute circular discrepancy already represented on the phase branch."""
    return np.abs(angle) * (720 / math.pi)


def saved_phase_minutes(source):
    summary = read_csv(source / "phase_summary.csv")
    rows = read_csv(source / "phase_comparison.csv")
    extrema = read_csv(source / "phase_extrema_diagnostics.csv")
    assert set(summary.pair) == {pair_name(p) for p in PAIRS}
    assert rows.effect.eq("matched_observations_14d").all()
    assert rows.left_variant.eq("15min_subsurface").all()
    assert rows.right_variant.eq("120min_subsurface").all()
    assert (pd.to_datetime(rows.end_exclusive) - pd.to_datetime(rows.start)).eq(14 * DAY).all()
    rows = rows[["pair", "window", "start", "end_exclusive", "center", "comparable", "delta_rad"]].copy()
    rows["e_tau_min"] = phase_minutes(rows.delta_rad.where(rows.comparable))
    translated = []
    for row in summary.itertuples(index=False):
        observed = rows.loc[(rows.pair == row.pair) & rows.comparable].copy()
        assert observed.window.is_unique
        assert len(observed) == row.comparable_windows
        assert pd.to_datetime(observed.start).diff().dropna().eq(DAY).all()
        absolute = observed.delta_rad.abs()
        saved = [row.MAE_rad, row.median_abs_rad, row.p95_abs_rad, row.max_abs_rad]
        np.testing.assert_allclose([absolute.mean(), absolute.median(), absolute.quantile(.95), absolute.max()],
                                   saved, rtol=0, atol=1e-14)
        maximum = observed.loc[absolute.idxmax()]
        diagnostic = extrema.set_index("pair").loc[row.pair]
        assert maximum.start == row.max_start == diagnostic.start
        assert maximum.end_exclusive == row.max_end_exclusive == diagnostic.end_exclusive
        assert maximum.center == row.max_center == diagnostic.center
        assert maximum.delta_rad == diagnostic.delta_rad
        minutes = [phase_minutes(v) for v in saved]
        np.testing.assert_allclose(minutes, [abs(math.degrees(v))*4 for v in saved], rtol=0, atol=5e-14)
        translated.append(dict(pair=row.pair, comparable_windows=row.comparable_windows, MAE_rad=row.MAE_rad,
            MAE_min=minutes[0], median_abs_min=minutes[1], p95_abs_min=minutes[2], max_abs_min=minutes[3],
            max_start=row.max_start, max_end_exclusive=row.max_end_exclusive, max_center=row.max_center))
    np.testing.assert_allclose(rows.e_tau_min, rows.delta_rad.map(lambda a: abs(math.degrees(a))*4),
                               rtol=0, atol=5e-14)
    return rows, pd.DataFrame(translated)


def accepted_pairs(source):
    points = read_csv(source / "concentration_residuals.csv.gz", parse_dates=["localdatetime"])
    manifest = json.loads((source / "passport.json").read_text())
    assert manifest["selector"] == "technically_retained & ~doubtful"
    assert points.groupby("sensorid").size().to_dict() == {int(k): v for k, v in manifest["source_counts"].items()}
    assert points.localdatetime.ge(START).all() and points.localdatetime.lt(END).all()
    assert np.isfinite(points.datavalue).all()
    channels = {}
    for sid, frame in points.groupby("sensorid"):
        index = frame.localdatetime.dt.floor("15min")
        if index.duplicated().any():
            raise ValueError(f"Multiple accepted measurements in one 15-minute cell for {sid}")
        channels[sid] = frame.set_index(index)[["localdatetime", "datavalue"]].rename(
            columns={"localdatetime": f"time_{sid}", "datavalue": f"C_{sid}"})
    return {pair: channels[pair[0]].join(channels[pair[1]], how="inner").sort_index() for pair in PAIRS}


def background_residual(fine, index, method):
    """Subtract fixed within-window slot medians, then optional linear trend."""
    slots = index.hour * 4 + index.minute // 15
    values = pd.DataFrame(fine)
    profile = values.groupby(slots).median().reindex(range(96)).to_numpy()
    if method == "smooth_profile_trend":
        smoothed = profile.copy()
        for slot in range(96):
            neighbours = profile[(slot + np.arange(-2, 3)) % 96]
            for channel in range(fine.shape[1]):
                available = neighbours[:, channel][np.isfinite(neighbours[:, channel])]
                if len(available):
                    smoothed[slot, channel] = np.median(available)
        profile = smoothed
    residual = fine - profile[slots]
    if method != "profile_only":
        calendar = np.arange(len(index), dtype=float)
        for channel in range(fine.shape[1]):
            y = residual[:, channel]
            observed = np.isfinite(y)
            if observed.sum() >= 2:
                t = calendar[observed] - calendar[observed].mean()
                y[observed] -= y[observed].mean() + t * (np.dot(t, y[observed]) / np.dot(t, t))
    return residual


@dataclass
class Design:
    step: int
    index: pd.DatetimeIndex
    q: np.ndarray
    shifts: np.ndarray
    anchors: np.ndarray
    upper: np.ndarray
    lower: np.ndarray
    dates: int
    occupied: int
    reason: str

    def correlation(self):
        if self.reason:
            return np.full(len(self.shifts), np.nan)
        upper = self.upper - self.upper.mean()
        lower = self.lower - self.lower.mean(axis=0)
        return np.clip(upper @ lower / (np.linalg.norm(upper) * np.linalg.norm(lower, axis=0)), -1, 1)

    def scale(self):
        if not len(self.anchors):
            return [np.nan, np.nan]
        return np.quantile(np.abs(self.q[self.anchors]), .9, axis=0).tolist()


def fixed_design(residual, start, step, deleted_day=False):
    n = len(residual)
    index = pd.date_range(start + pd.Timedelta(minutes=step/2), periods=n, freq=f"{step}min")
    q = np.diff(residual, axis=0) / step
    shifts = np.arange(-240, 241, step)
    offsets = shifts // step
    anchors = np.arange(-offsets[0], len(q) - offsets[-1])
    grid = anchors[:, None] + offsets
    shared = np.isfinite(q[grid]).all(axis=(1, 2))
    anchors, grid = anchors[shared], grid[shared]
    upper, lower = q[anchors, 0], q[grid, 1]
    dates = int(index[anchors].normalize().nunique())
    minimum = LIMITS[step][int(deleted_day)]
    if len(anchors) < minimum or dates < (4 if deleted_day else 5):
        reason = "insufficient_fixed_support"
    elif np.var(upper) == 0 or np.any(np.var(lower, axis=0) == 0):
        reason = "zero_rate_variance"
    else:
        reason = ""
    return Design(step, index, q, shifts, anchors, upper, lower, dates,
                  int(np.isfinite(residual).all(axis=1).sum()), reason)


def prepare_window(frame, pair, start, *, method="profile_trend", remove=(), full_only=False):
    fine_index = pd.date_range(start, periods=7*96, freq="15min")
    frame = frame.reindex(fine_index).copy()
    frame.loc[frame.index.isin(remove), :] = np.nan
    columns = [f"C_{sid}" for sid in pair]
    present = frame[columns].notna().all(axis=1).to_numpy().reshape(-1, 8)
    if full_only:
        frame.iloc[np.repeat(~present.all(axis=1), 8), :] = np.nan
        present = frame[columns].notna().all(axis=1).to_numpy().reshape(-1, 8)
    fine = frame[columns].to_numpy()
    residual = background_residual(fine, fine_index, method)
    counts = present.sum(axis=1)

    def aggregate(values):
        sums = np.nansum(values.reshape(-1, 8, 2), axis=1)
        return np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)

    return frame, residual, aggregate(fine), aggregate(residual), counts


def window_design(frame, pair, start, *, method="profile_trend", remove=(), full_only=False, deleted_day=False):
    prepared = prepare_window(frame, pair, start, method=method, remove=remove, full_only=full_only)
    return prepared, {step: fixed_design(values, start, step, deleted_day)
                      for step, values in [(15, prepared[1]), (120, prepared[3])]}


def select_windows(frame, pair):
    # A spare day at either end allows the two fixed boundary perturbations.
    candidates = pd.date_range(START + DAY, END - 8*DAY, freq="7D")
    audit = []
    for start in candidates:
        _, designs = window_design(frame, pair, start)
        scale = designs[15].scale()[0]
        audit.append(dict(start=start, end_exclusive=start+7*DAY, score_ppm_min=scale,
            usable=all(not d.reason for d in designs.values()) and scale > 0,
            **{f"anchors_{h}": len(d.anchors) for h,d in designs.items()},
            **{f"dates_{h}": d.dates for h,d in designs.items()},
            **{f"reason_{h}": d.reason for h,d in designs.items()}))
    audit = pd.DataFrame(audit)
    audit["selection"] = ""
    eligible = audit.loc[audit.usable]
    if len(eligible):
        strong = eligible.sort_values(["score_ppm_min", "start"], ascending=[False, True]).index[0]
        audit.loc[strong, "selection"] = "expressive"
        remaining = eligible.drop(index=strong).assign(
            distance=lambda f: (f.score_ppm_min - eligible.score_ppm_min.median()).abs())
        if len(remaining):
            typical = remaining.sort_values(["distance", "start"]).index[0]
            audit.loc[typical, "selection"] = "typical"
    return audit.assign(pair=pair_name(pair))


def variants(start, prepared, designs, pair):
    frame = prepared[0]
    yield "base", start, {}
    for shift in (-1, 1):
        yield f"window_{shift:+d}d", start + shift*DAY, {}
    for method in ("profile_only", "smooth_profile_trend"):
        yield method, start, {"method": method}
    for day in range(7):
        removed = frame.index[frame.index.normalize() == start + day*DAY]
        yield f"drop_day_{day + 1}", start, {"remove": removed, "deleted_day": True}
    for channel, sid in enumerate(pair):
        values = frame[f"C_{sid}"]
        yield f"drop_min_C_{sid}", start, {"remove": [values.idxmin()]}
        yield f"drop_max_C_{sid}", start, {"remove": [values.idxmax()]}
        # Maximum over all existing adjacent baseline residual differences.
        biggest = int(np.nanargmax(np.abs(designs[15].q[:, channel])))
        yield f"drop_max_abs_q_{sid}", start, {"remove": frame.index[biggest:biggest+2]}
    yield "full_2h_only", start, {"full_only": True}


def maxima(curve, shifts):
    if not np.isfinite(curve).all():
        return []
    return shifts[curve >= curve.max() - TIE].astype(int).tolist()


def describe_peak(base, curves, shifts, core_available):
    """Distinguish a maximum, envelope stability and failure to resolve it."""
    base_max = maxima(base, shifts)
    if not base_max:
        return dict(base_maxima=[], competing=[], stable_lag_min=None, status="no_defined_curve",
                    peak_r=None, runner_up_gap=None, r_at_zero=None, peak_lower=None)
    low, high = np.min(curves, axis=0), np.max(curves, axis=0)
    competing = shifts[high >= low.max() - TIE].astype(int).tolist()
    best = int(np.argmax(base))
    stability = None
    if not core_available:
        status = "required_perturbation_unavailable"
    elif len(competing) != 1:
        status = "competing_or_wide_peak"
    elif competing[0] in (int(shifts[0]), int(shifts[-1])):
        status = "boundary_peak"
    elif low[shifts == competing[0]][0] <= 0:
        status = "nonpositive_lower_envelope"
    else:
        stability = competing[0]
        status = "stable_zero_grid_peak" if stability == 0 else "stable_nonzero_grid_peak"
    return dict(base_maxima=base_max, competing=competing, stable_lag_min=stability, status=status,
                peak_r=float(base[best]), runner_up_gap=float(base[best] - np.delete(base, best).max()),
                r_at_zero=float(base[shifts == 0][0]), peak_lower=float(low[best]))


def coarse_cells(prepared, pair, start, label):
    """Save the actual membership and mean source time for each 2-hour bin."""
    frame, residual, raw_2h, residual_2h, counts = prepared
    for k, count in enumerate(counts):
        row = dict(variant=label, bin_start=start+pd.Timedelta(hours=2*k),
                   n=int(count), complete_2h=bool(count == 8))
        fine = frame.iloc[k*8:(k+1)*8]
        for j, sid in enumerate(pair):
            times = fine[f"time_{sid}"].dropna()
            # Average offsets relative to the bin start to avoid epoch precision loss.
            mean_time = row["bin_start"] + (times - row["bin_start"]).mean() if count else pd.NaT
            row.update({f"source_times_{sid}": "|".join(t.isoformat() for t in times),
                        f"mean_source_time_{sid}": mean_time, f"C_{sid}_ppm": raw_2h[k,j],
                        f"X_{sid}_ppm": residual_2h[k,j]})
        yield row


def scalar_baseline_rates(prepared, pair):
    """Independent baseline from raw concentrations using Python scalar sums."""
    frame = prepared[0]
    channels = []
    for sid in pair:
        raw = frame[f"C_{sid}"].tolist()
        profile = {}
        for i, value in enumerate(raw):
            if math.isfinite(value):
                profile.setdefault(i % 96, []).append(value)
        profile = {k: median(series) for k, series in profile.items()}
        centered = {i: value-profile[i % 96] for i, value in enumerate(raw) if math.isfinite(value)}
        mean_time = math.fsum(centered)/len(centered)
        mean_c = math.fsum(centered.values())/len(centered)
        slope = math.fsum((i-mean_time)*(v-mean_c) for i,v in centered.items()) / math.fsum(
            (i-mean_time)**2 for i in centered)
        channels.append({i: v - mean_c - slope*(i-mean_time) for i,v in centered.items()})
    outputs = {}
    for step in (15, 120):
        width = step // 15
        rates = []
        for channel in channels:
            bins = []
            for first in range(0, len(frame), width):
                observed = [channel[i] for i in range(first,first+width) if i in channel]
                bins.append(math.fsum(observed)/len(observed) if observed else math.nan)
            rates.append([(following-current)/step for current,following in zip(bins,bins[1:])])
        outputs[step] = np.asarray(rates).T
    return outputs


def scalar_check(design, reference_q):
    """Independent scalar Pearson sum from full-grid q and the calendar."""
    if design.reason:
        return []
    # Rebuild the support by a direct calendar scan, including all signed shifts.
    np.testing.assert_allclose(design.q, reference_q, rtol=0, atol=1e-12)
    q = reference_q.tolist()
    offsets = [int(x)//design.step for x in design.shifts]
    support = [i for i in range(len(q)) if all(0 <= i+s < len(q) and
               all(math.isfinite(v) for v in q[i+s]) for s in offsets)]
    assert support == design.anchors.tolist()
    validation = []
    for shift, vector in zip(design.shifts, design.correlation()):
        a = [q[k][0] for k in support]
        b = [q[k+int(shift)//design.step][1] for k in support]
        ma, mb = math.fsum(a)/len(a), math.fsum(b)/len(b)
        cov = math.fsum((x-ma)*(y-mb) for x,y in zip(a,b))
        den = math.sqrt(math.fsum((x-ma)**2 for x in a) * math.fsum((y-mb)**2 for y in b))
        assert abs(vector - cov/den) < 5e-14
        validation.append(abs(vector - cov/den))
    return validation


def analyze_window(frame, pair, start, selection):
    prepared, baseline = window_design(frame, pair, start)
    support, curve_rows, cell_rows, summaries = [], [], [], []
    saved = {h: {} for h in LIMITS}
    labels = {}
    for label, new_start, settings in variants(start, prepared, baseline, pair):
        sample, designs = window_design(frame, pair, new_start, **settings)
        joint = all(not d.reason for d in designs.values())
        labels[label] = joint
        cell_rows.extend(coarse_cells(sample, pair, new_start, label))
        for step, d in designs.items():
            scales = d.scale()
            curve = d.correlation()
            saved[step][label] = curve
            removed = settings.get("remove", ())
            support.append(dict(variant=label, start=new_start, end_exclusive=new_start+7*DAY,
                step_min=step, occupied_cells=d.occupied, possible_cells=len(d.index), anchors=len(d.anchors),
                distinct_anchor_dates=d.dates, reason=d.reason, joint_available=joint,
                upper_P90_abs_rate_ppm_min=scales[0], lower_P90_abs_rate_ppm_min=scales[1],
                removed_cells="|".join(t.isoformat() for t in removed),
                anchor_centers="|".join(d.index[i].isoformat() for i in d.anchors),
                maxima_min=json.dumps(maxima(curve, d.shifts))))
            curve_rows.extend(dict(variant=label, step_min=step, shift_min=int(k), r=v)
                              for k,v in zip(d.shifts,curve))
    display_curves = {}
    for step, d in baseline.items():
        # Full-only is a joint sensitivity check. An unavailable version is
        # documented separately; it cannot be treated as a passed check.
        finite = [c for v,c in saved[step].items() if np.isfinite(c).all() and
                  (v != "full_2h_only" or labels[v])]
        core_available = all(np.isfinite(c).all() for v,c in saved[step].items() if v != "full_2h_only")
        peak = describe_peak(saved[step]["base"], finite, d.shifts, core_available)
        low, high = np.min(finite, axis=0), np.max(finite, axis=0)
        display_curves[step] = (d.shifts, saved[step]["base"], low, high, peak)
        restricted = np.isin(d.shifts, baseline[120].shifts)
        summaries.append(dict(step_min=step, occupied_cells=d.occupied, possible_cells=len(d.index),
            anchors=len(d.anchors), distinct_anchor_dates=d.dates,
            upper_P90_abs_rate_ppm_min=d.scale()[0], lower_P90_abs_rate_ppm_min=d.scale()[1],
            **peak, full_only_joint_available=labels["full_2h_only"],
            unavailable_core_variants="|".join(v for v,c in saved[step].items()
                                               if v != "full_2h_only" and not np.isfinite(c).all()),
            max_on_common_shifts_min=maxima(saved[step]["base"][restricted], d.shifts[restricted])))
        curve_rows.extend(dict(variant=variant, step_min=step, shift_min=int(k), r=v)
                          for variant, line in [("envelope_min",low), ("envelope_max",high)]
                          for k,v in zip(d.shifts,line))
    stable = [r["stable_lag_min"] for r in summaries]
    change = stable[1] - stable[0] if None not in stable else None
    for r in summaries:
        r["coarse_minus_fine_stable_grid_peak_min"] = change
    reference_rates = scalar_baseline_rates(prepared, pair)
    check = {str(h): {"lags":len(errors), "max_abs_r_error":max(errors)}
             for h,d in baseline.items() if (errors := scalar_check(d, reference_rates[h]))}
    prefix = dict(pair=pair_name(pair), selection=selection, baseline_start=start, baseline_end_exclusive=start+7*DAY)
    tables = [pd.DataFrame(rows).assign(**prefix) for rows in (summaries, support, curve_rows, cell_rows)]
    points = prepared[0].rename_axis("fine_start").reset_index()
    points = points.loc[points[[f"C_{s}" for s in pair]].notna().all(axis=1)].assign(**prefix)
    return tables, points, prepared, display_curves, check


def save_jpg(fig, path):
    fig.savefig(path, dpi=300, facecolor="white", pil_kwargs={"quality": 94, "subsampling": 0})
    plt.close(fig)


def figure_phase(rows, out):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 5.5), layout="constrained")
    fig.suptitle("Daily harmonic · LEO West W_x1_y4 (x = 1 m, y = 4 m)")
    for pair, ax in zip(PAIRS, axes):
        data = rows[rows.pair == pair_name(pair)]
        t = pd.to_datetime(data.center)
        ax.plot(t, data.e_tau_min, color="#184d72", lw=1, label="Absolute discrepancy")
        maximum = data.e_tau_min.idxmax()
        ax.scatter(t[maximum], data.e_tau_min[maximum], color="#184d72", s=26, zorder=4)
        for dt, style in [(15, "--"), (120, ":")]:
            ax.axhline(dt, color="#976835", ls=style, lw=1, label=f"{dt} min grid-step reference")
        ax.set_title(f"{pair_label(pair)} · GMM222 {pair[0]} → {pair[1]}", loc="left")
        ax.set_ylim(0, max(160, rows.e_tau_min.max()*1.07))
        ax.grid(axis="y", alpha=.22)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    axes[-1].set_xlabel("14-day window center (LOCALDATETIME)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.supylabel("Absolute phase-delay discrepancy between 15-min and 2-h grids (min)", fontsize=10)
    save_jpg(fig, out / "phase_delay_discrepancy_min.jpg")


def figure_pair(pair, episodes, out):
    if not episodes:
        return
    fig, axes = plt.subplots(3, len(episodes), squeeze=False, figsize=(6.5*len(episodes), 7.5),
                             layout="constrained")
    fig.suptitle(f"Short-lag comparison · LEO West W_x1_y4 (x = 1 m, y = 4 m)\n"
                 f"{pair_label(pair)} · GMM222 {pair[0]} → {pair[1]}")
    colors = {15:"#184f77", 120:"#a34b24"}
    for column, (selected, prepared, curves) in enumerate(episodes):
        frame, _, coarse, _, _ = prepared
        end = selected.start + 7*DAY
        edges = pd.date_range(selected.start, end, freq="2h")
        for i, sid in enumerate(pair):
            ax = axes[i, column]
            ax.scatter(frame[f"time_{sid}"], frame[f"C_{sid}"], s=4, color=colors[15],
                       label="Accepted pair-matched 15-min observations")
            ax.stairs(coarse[:, i], edges, baseline=None, color=colors[120], lw=1.1, label="2-h means")
            ax.set(xlim=(selected.start, end), ylabel="CO₂ concentration (ppm)")
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.set_title(f"{DEPTH[sid]} cm · GMM222 {sid}", loc="right", fontsize=9)
            ax.grid(alpha=.16)
        axes[0,column].set_title(f"{selected.selection.capitalize()} · {selected.start:%d %b %Y} to {end:%d %b %Y} (exclusive)",
                                 loc="left", fontsize=9)
        axes[1,column].set_xlabel("Time")
        ax = axes[2,column]
        for h, (shifts, line, low, high, peak) in curves.items():
            name = "15 min" if h == 15 else "2 h"
            ax.fill_between(shifts, low, high, color=colors[h], alpha=.14)
            ax.plot(shifts, line, ".-", color=colors[h], lw=1, label=name)
            sel = np.isin(shifts, peak["base_maxima"])
            ax.scatter(shifts[sel], line[sel], facecolors="white", edgecolors=colors[h], s=45, zorder=4)
        ax.axvline(0, lw=.8, color=".45", ls=":")
        ax.set(xlim=(-247,247), ylim=(-1,1), xlabel="Shift of lower signal (min)", ylabel="Pearson r (dimensionless)")
        ax.set_xticks(np.arange(-240, 241, 120))
        ax.grid(alpha=.2)
    axes[0,0].legend(loc="best", fontsize=7, frameon=True)
    axes[2,0].legend(loc="lower left", fontsize=8, frameon=False)
    save_jpg(fig, out / f"short_lags_{pair_name(pair)}.jpg")


def figure_hypothesis(out):
    fig, ax = plt.subplots(figsize=(9.2, 5.5), layout="constrained")
    ax.set(xlim=(0,1), ylim=(0,1))
    ax.axis("off")
    ax.text(.5,.97,"Candidate physical hypothesis",ha="center",va="top",fontsize=22,weight="bold",color="#1b3039")
    ax.text(.5,.885,"Hypothesized relationships — not established causality",
            ha="center",va="top",fontsize=11,color="#724333")
    names = ("Irrigation", "Changes in substrate moisture", "Possible changes in effective CO₂ transport",
             "Changes in characteristic response times", "Possible changes in relative spectral phases",
             "Research question: Does 2-hour averaging preserve\nthe information needed to detect these changes?")
    positions = (.773,.654,.535,.416,.297,.119)
    for i, (label, y) in enumerate(zip(names, positions)):
        ax.text(.5,y,label,ha="center",va="center",fontsize=13,color="#16333b",
                bbox=dict(boxstyle="round,pad=.54",facecolor="#dceced" if i<5 else "#f3e7cf",
                          edgecolor="#698489" if i<5 else "#a18a56",linewidth=.9))
        if i:
            ax.annotate("", xy=(.5,y+(.066 if i==5 else .043)), xytext=(.5,positions[i-1]-.043),
                        arrowprops=dict(arrowstyle="-|>",color="#698489",lw=1.15))
    # Embed vector fonts in PDF and preserve editable text in SVG.
    with plt.rc_context({"pdf.fonttype":42,"svg.fonttype":"none","svg.hashsalt":"co2-short-lag"}):
        fig.savefig(out/"candidate_physical_hypothesis.pdf", metadata={"CreationDate":None, "ModDate":None})
        svg = out/"candidate_physical_hypothesis.svg"
        fig.savefig(svg, metadata={"Date":None})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
        save_jpg(fig, out/"candidate_physical_hypothesis.jpg")


def save_table(frame, path):
    frame = frame.copy()
    for name in frame.columns:
        if frame[name].map(lambda value: isinstance(value, list)).any():
            frame[name] = frame[name].map(lambda value: json.dumps(value) if isinstance(value,list) else value)
    compression = {"method":"gzip", "mtime":0} if path.suffix == ".gz" else None
    frame.to_csv(path, index=False, float_format="%.17g", compression=compression)


def generate(out=OUT, source=SOURCE):
    out, source = Path(out), Path(source)
    names = ("concentration_residuals.csv.gz","phase_summary.csv","phase_comparison.csv",
             "phase_extrema_diagnostics.csv","passport.json")
    hashes = {name: file_hash(source/name) for name in names}
    validation = json.loads((source/"validation.json").read_text())
    for name, observed in hashes.items():
        assert validation["artifact_sha256"][name] == observed, name
    input_passport = json.loads((source/"passport.json").read_text())
    assert (input_passport["f0_cycles_per_day"],input_passport["window_days"],input_passport["step_days"]) == (1,14,1)
    points_by_pair = accepted_pairs(source)
    rows, phase_summary = saved_phase_minutes(source)
    control_angles = [0, math.pi/12, math.pi/6, math.pi]
    control_minutes = [0, 60, 120, 720]
    np.testing.assert_allclose(phase_minutes(control_angles), control_minutes, rtol=0, atol=1e-12)
    np.testing.assert_allclose([abs(math.degrees(a))*4 for a in control_angles], control_minutes, rtol=0, atol=1e-12)
    out.mkdir(parents=True, exist_ok=True)
    save_table(rows, out/"phase_delay_discrepancy_windows.csv")
    save_table(phase_summary, out/"phase_delay_discrepancy_summary.csv")
    figure_phase(rows, out)
    figure_hypothesis(out)
    all_tables = {name:[] for name in ("short_lag_summary.csv","variant_support.csv.gz","lag_curves.csv",
                                    "two_hour_membership.csv.gz","selected_source_points.csv.gz","candidate_screen.csv")}
    checks = {}
    for pair, frame in points_by_pair.items():
        audit = select_windows(frame, pair)
        all_tables["candidate_screen.csv"].append(audit)
        episodes = []
        for selected in audit[audit.selection != ""].sort_values("selection").itertuples(index=False):
            tables, points, prepared, curves, check = analyze_window(frame, pair, selected.start, selected.selection)
            checks[f"{pair_name(pair)}:{selected.start.date()}"] = check
            episodes.append((selected, prepared, curves))
            for name, table in zip(all_tables, [*tables, points]):
                all_tables[name].append(table)
        figure_pair(pair, episodes, out)
    for name, tables in all_tables.items():
        if tables:
            save_table(pd.concat(tables, ignore_index=True), out/name)
    assert all(file_hash(source/name) == observed for name,observed in hashes.items())
    passport = dict(baseline_commit="f71ca35b1b4a0e01941adae8c049e83f4643866d",
        stage="0.5", selector=input_passport["selector"], vertical="W_x1_y4", x_m=1,y_m=4,
        sensor_depth_cm=DEPTH, acquisition_calendar="[2025-05-01, 2026-10-01), LOCALDATETIME",
        part_a="Absolute time equivalent of saved cyclic daily phase discrepancy (1 cycle/day, 14-day windows, 1-day step)",
        part_b=dict(window_days=7, step_minutes=[15,120], permitted_lag_bounds_min=[-240,240],
            coverage_minima=dict(ordinary={"15":384,"120":48,"anchor_dates":5},
                                 drop_day={"15":288,"120":36,"anchor_dates":4}),
            analysis_time="Calendar centers, raw timestamps and actual per-bin mean source times retained",
            own_clock_note=input_passport["time_basis"],
            amplitude="Linear empirical 90th percentile of absolute q at unshifted anchors on each grid",
            selection="Positive upper fine amplitude; two fixed eligible windows (max, nearest median excluding max); earlier ties",
            fixed_support="Both channels have q at each shifted anchor; q requires adjacent occupied calendar cells",
            perturbations="Window ±1 day, two background alternatives, each day, C extrema, both cells forming max |q| per sensor, full 2h subset",
            peak_criterion="All required core curves defined; unique interior envelope candidate with lower correlation > 0. Full-bin variant included only when jointly supported. A stable discrete peak alone does not establish a resolved delay.",
            ties="64 machine epsilons for Pearson equality only",
            sensitivity="Min/max across defined variants, not a statistical confidence interval"),
        input_sha256=hashes, accepted_passport_sha256=file_hash(ROOT/"output/stage02_candidate_selection/passport.json"),
        implementation_sha256=file_hash(Path(__file__)), fourier_recomputed=False,
        version=dict(numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__))
    (out/"passport.json").write_text(json.dumps(passport, indent=2, ensure_ascii=False)+"\n")
    report = dict(scalar_Pearson_checks=checks, conversion_reference_minutes=control_minutes,
                  source_files_unchanged=True, artifacts_sha256={p.name:file_hash(p) for p in sorted(out.iterdir())
                                                                if p.name!="validation.json"})
    (out/"validation.json").write_text(json.dumps(report,indent=2)+"\n")


def display_figure(name, out=OUT):
    """Show a compact embedded preview of the print-resolution JPG."""
    from io import BytesIO
    from IPython.display import display, Image
    from PIL import Image as PILImage

    with PILImage.open(Path(out)/name) as image:
        image.thumbnail((940, 650), PILImage.Resampling.LANCZOS)
        target = BytesIO()
        image.save(target, format="JPEG", quality=60, optimize=True)
        display(Image(data=target.getvalue(), format="jpeg", width=image.width))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    generate(parser.parse_args().output)

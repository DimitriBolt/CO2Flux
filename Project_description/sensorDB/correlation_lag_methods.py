"""Stage 0.5: concentration and residual correlations on four archived intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import short_lag_resolution as archive


OUT = archive.SOURCE / "correlation_methods"
BACKGROUNDS = ("profile_only", "smooth_profile_trend")
SUMMARY_FIELDS = ("base_maxima", "peak_r", "competing", "stable_lag_min", "status",
                  "runner_up_gap", "r_at_zero", "peak_lower")


def fixed_support(occupied, step):
    """Both series must exist at every offset for every retained calendar anchor."""
    shifts = np.arange(-240, 241, step)
    offsets = shifts // step
    candidates = np.arange(-offsets[0], len(occupied) - offsets[-1])
    index = candidates[:, None] + offsets
    valid = occupied[index].all(axis=1)
    return shifts, candidates[valid], index[valid]


def coverage(occupied, step, deleted_day):
    shifts, anchors, shifted = fixed_support(occupied, step)
    days = len(np.unique(anchors // (1440 // step)))
    enough = len(anchors) >= archive.LIMITS[step][int(deleted_day)] and days >= (4 if deleted_day else 5)
    return shifts, anchors, shifted, days, enough


def correlation_curve(values, step, deleted_day=False):
    occupied = np.isfinite(values).all(axis=1)
    shifts, anchors, shifted, days, enough = coverage(occupied, step, deleted_day)
    reason = ""
    line = np.full(len(shifts), np.nan)
    if not enough:
        reason = "insufficient_fixed_support"
    else:
        upper = values[anchors, 0]
        lower = values[shifted, 1]
        if np.ptp(upper) == 0 or (np.ptp(lower, axis=0) == 0).any():
            reason = "zero_variance"
        else:
            line = np.corrcoef(upper, lower.T)[0, 1:]
    return shifts, line, anchors, days, reason


def level_signals(frame, pair, variant, method, step):
    """Apply the archived paired exclusions before fitting the background or averaging."""
    start = pd.Timestamp(variant.start)
    index = pd.date_range(start, periods=7*96, freq="15min")
    signals = frame.reindex(index)[[f"C_{sid}" for sid in pair]].to_numpy(copy=True)
    if pd.notna(variant.removed_cells) and variant.removed_cells:
        removed = pd.to_datetime(variant.removed_cells.split("|"))
        signals[index.isin(removed)] = np.nan
    paired = np.isfinite(signals).all(axis=1)
    if variant.variant == "full_2h_only":
        paired = np.repeat(paired.reshape(-1, 8).all(axis=1), 8)
    signals[~paired] = np.nan
    if method == "B":
        background = variant.variant if variant.variant in BACKGROUNDS else "profile_trend"
        signals = archive.background_residual(signals, index, background)
    if step == 120:
        counts = paired.reshape(-1, 8).sum(axis=1)[:, None]
        totals = np.nansum(signals.reshape(-1, 8, 2), axis=1)
        signals = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return signals, paired


def analyze_grid(frame, pair, variants, method, step):
    curves, audit = {}, []
    for variant in variants.itertuples(index=False):
        name = variant.variant
        if method == "A" and name in BACKGROUNDS:
            continue
        values, paired = level_signals(frame, pair, variant, method, step)
        shifts, line, anchors, days, reason = correlation_curve(values, step, name.startswith("drop_day_"))
        if name == "full_2h_only" and not reason:
            # Only calendar occupancy on the other grid is needed here.
            other_coverage = (coverage(paired.reshape(-1, 8).all(axis=1), 120, False)
                              if step == 15 else coverage(paired, 15, False))
            if not other_coverage[-1]:
                reason = "insufficient_joint_full_bin_support"
        if reason:
            line = np.full(len(shifts), np.nan)
        curves[name] = line
        audit.append(dict(variant=name, start=variant.start, removed_cells=variant.removed_cells,
                          anchors=len(anchors), anchor_offsets=anchors.tolist(), days=days, reason=reason))
    core_ok = all(not r["reason"] for r in audit if r["variant"] != "full_2h_only")
    defined = [line for line in curves.values() if np.isfinite(line).all()]
    peak = archive.describe_peak(curves["base"], defined, shifts, core_ok)
    if defined:
        curves["envelope_min"] = np.min(defined, axis=0)
        curves["envelope_max"] = np.max(defined, axis=0)
    return peak, pd.DataFrame(curves, index=pd.Index(shifts, name="shift_min")), pd.DataFrame(audit)


def compare_grids(fine, coarse):
    if fine["stable_lag_min"] is None:
        return "no_fine_estimate"
    if coarse["status"] in ("no_defined_curve", "required_perturbation_unavailable"):
        return "undetermined"
    if coarse["stable_lag_min"] is None:
        return "lost_resolvability"
    grid = np.arange(-240, 241, 120)
    distances = np.abs(grid - fine["stable_lag_min"])
    nearest = grid[distances == distances.min()]
    if coarse["stable_lag_min"] not in nearest:
        return "different_discrete_estimate"
    return "compatible_tied_nearest" if len(nearest) == 2 else "compatible_nearest"


def analyze_method(frame, pair, variants, method):
    fine = analyze_grid(frame, pair, variants, method, 15)
    results = {15: fine}
    coarse_peak = None
    if fine[0]["stable_lag_min"] is not None:
        results[120] = analyze_grid(frame, pair, variants, method, 120)
        coarse_peak = results[120][0]
    return results, compare_grids(fine[0], coarse_peak)


def comparison_figure(pair, summary, curves, out):
    colors = dict(A="#1c5880", B="#a34c25", C="#396c4c")
    titles = dict(A="A · Concentrations", B="B · Residuals", C="C · Residual rates")
    fig, axes = archive.plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 5.6),
                                     layout="constrained")
    fig.suptitle(f"15-min correlations · LEO West W_x1_y4 · {archive.pair_label(pair)} · "
                 f"GMM222 {pair[0]} / {pair[1]}")
    for row, selection in enumerate(("expressive", "typical")):
        for col, method in enumerate("ABC"):
            ax = axes[row, col]
            one = summary.loc[(summary.selection == selection) & (summary.method == method)].iloc[0]
            series = curves[(curves.selection == selection) & (curves.method == method)]
            lines = series.pivot(index="shift_min", columns="variant", values="r")
            color = colors[method]
            ax.fill_between(lines.index, lines.envelope_min, lines.envelope_max, color=color,
                            alpha=.22, label="Sensitivity envelope")
            ax.plot(lines.index, lines.base, ".-", ms=3, lw=1.1, color=color, label="Baseline")
            peak = json.loads(one.base_maxima)
            ax.scatter(peak, lines.base.loc[peak], s=45, facecolor="white", edgecolor=color,
                       zorder=4, label="Baseline maximum")
            candidates = json.loads(one.competing)
            ax.plot(candidates, np.full(len(candidates), -1.025), "|", color=color,
                    ms=6, label="Competing shifts")
            if row == 0:
                ax.set_title(titles[method])
            if col == 0:
                start = pd.Timestamp(one.baseline_start)
                ax.set_ylabel(f"{selection.capitalize()} · {start:%d %b %Y}\nPearson r")
            ax.axvline(0, color=".45", linestyle=":", lw=.8)
            ax.set(ylim=(-1.10, 1.06), xlim=(-255, 255), xticks=np.arange(-240,241,120))
            ax.grid(alpha=.18)
    axes[1, 1].set_xlabel("Shift of lower signal (min) · positive = lower signal later", fontsize=10)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
               ncols=4, frameon=False, fontsize=9)
    archive.save_jpg(fig, out / f"correlations_{archive.pair_name(pair)}.jpg")


def generate(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive_dir = archive.OUT
    originals = json.loads((archive_dir / "validation.json").read_text())
    inputs = {archive_dir/name: originals["artifacts_sha256"][name] for name in
              ("variant_support.csv.gz", "short_lag_summary.csv", "lag_curves.csv")}
    for name in ("concentration_residuals.csv.gz", "passport.json"):
        path = archive.SOURCE/name
        inputs[path] = originals["original_files_sha256"][str(path.relative_to(archive.ROOT.parents[1]))]
    for path, expected in inputs.items():
        if archive.file_hash(path) != expected:
            raise ValueError(f"Published input has changed: {path.name}")

    frames = archive.accepted_pairs(archive.SOURCE)
    frozen = archive.read_csv(archive_dir/"variant_support.csv.gz")
    historic = archive.read_csv(archive_dir/"short_lag_summary.csv")
    saved_lines = archive.read_csv(archive_dir/"lag_curves.csv")
    summaries, curves, audits = [], [], []
    keys = ["pair", "selection", "baseline_start", "baseline_end_exclusive"]
    for pair, frame in frames.items():
        group = frozen.loc[(frozen.pair == archive.pair_name(pair)) & (frozen.step_min == 15)]
        for key, variants in group.groupby(keys, sort=False):
            episode = dict(zip(keys, key))
            for method in ("A", "B"):
                results, comparison = analyze_method(frame, pair, variants, method)
                for step, (peak, lines, audit) in results.items():
                    ident = dict(**episode, method=method, step_min=step)
                    summaries.append(dict(**ident, **peak, comparison=comparison))
                    curves.append(lines.reset_index().melt(id_vars="shift_min", var_name="variant",
                                  value_name="r").assign(**ident))
                    audits.append(audit.assign(**ident))
    summary = pd.DataFrame(summaries)
    historic = historic[keys+["step_min", *SUMMARY_FIELDS]].copy()
    historic["method"] = "C"
    # The published C baseline was unresolved in all four fine-grid episodes.
    if historic.loc[historic.step_min == 15, "stable_lag_min"].notna().any():
        raise ValueError("Expected the four archived unresolved method C episodes")
    historic["comparison"] = "no_fine_estimate"
    for col in ("base_maxima", "competing"):
        historic[col] = historic[col].map(json.loads)
    summary = pd.concat([summary, historic], ignore_index=True).sort_values(keys+["method","step_min"])
    archive.save_table(summary, out/"summary.csv")
    curves = pd.concat(curves, ignore_index=True)
    archive.save_table(curves, out/"curves_ab.csv.gz")
    archive.save_table(pd.concat(audits, ignore_index=True), out/"support_ab.csv.gz")
    plotted = pd.concat([curves, saved_lines.assign(method="C")], ignore_index=True)
    summary = archive.read_csv(out/"summary.csv")
    for pair in archive.PAIRS:
        name = archive.pair_name(pair)
        comparison_figure(pair, summary[(summary.pair == name) & (summary.step_min == 15)],
                          plotted[(plotted.pair == name) & (plotted.step_min == 15)], out)
    files = ["summary.csv", "curves_ab.csv.gz", "support_ab.csv.gz",
             *(f"correlations_{archive.pair_name(p)}.jpg" for p in archive.PAIRS)]
    passport = dict(baseline_commit="cc4cf82ad61abe672dc307fc617193a794592c2a",
                    protocol="PLAN.md#short-lag-resolution",
                    protocol_sha256_at_run=archive.file_hash(archive.ROOT/"PLAN.md"),
                    input_sha256={str(p.relative_to(archive.ROOT)): h for p,h in inputs.items()},
                    artifact_sha256={name: archive.file_hash(out/name) for name in files},
                    new_coarse_curves=int(((summary.method != "C") & (summary.step_min == 120)).sum()),
                    method_c_source="archived tables", fourier_recomputed=False)
    (out/"passport.json").write_text(json.dumps(passport, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    generate(parser.parse_args().output)

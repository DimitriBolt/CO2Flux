"""Approved Chapter 3 paired, calendar-preserving comparisons; no raw-data fitting."""

from itertools import combinations

import numpy as np
import pandas as pd

LEVELS = ("D1", "D2", "D3")
PARAMETERS = {
    "procedure_version": "2026-09-29",
    "source_main_commit": "5b44f644651954e467ee70bfc2a31beee09b5a1f",
    "column": "W_R4_C-4",
    "sensor_ids": {"D1": 994, "D2": 1010, "D3": 1026},
    "start": "2017-10-01", "end_exclusive": "2020-09-01",
    "expected_main_days": 783, "min_pairs": 10, "min_iso_weeks_each_year": 2,
    "bootstrap_repetitions": 2000, "primary_block_days": 7,
    "block_days": [7, 3, 14], "confidence": 0.95, "seed": 20260929,
    "period_hours": 24, "resultant_numerical_tolerance": 1e-12,
    "phase_local_half_arc_hours": 6,
    "quantile_method": "linear",
    "coverage_variants": ["pairwise", "all_years_common"],
    "block_rule": "moving full calendar blocks; every pair covered; at least two distinct blocks",
    "multiplicity": "pointwise intervals only; no family-wide significance claim",
}


def wrap24(values):
    return (np.asarray(values) + 12.0) % 24.0 - 12.0


def circular_mean(hours, axis=0):
    z = np.mean(np.exp(2j * np.pi * np.asarray(hours) / 24.0), axis=axis)
    r = np.abs(z)
    mean = wrap24(np.angle(z) * 24.0 / (2 * np.pi))
    return np.where(r > PARAMETERS["resultant_numerical_tolerance"], mean, np.nan), r


def calendar_blocks(days, length):
    """Indices for full blocks, never bridging a missing calendar day."""
    days = np.asarray(days, dtype=int)
    if len(days) and (np.diff(days) <= 0).any():
        raise ValueError("Day-of-month keys must be strictly increasing")
    blocks = [np.arange(i, i + length) for i in range(len(days) - length + 1)
              if days[i + length - 1] - days[i] == length - 1]
    blocks = np.asarray(blocks, dtype=int).reshape(-1, length)
    covered = np.zeros(len(days), dtype=bool)
    if blocks.size:
        covered[blocks.ravel()] = True
    reason = "ok"
    if not covered.all() or not len(days):
        reason = "some_pairs_outside_full_blocks"
    elif len(blocks) < 2:
        reason = "fewer_than_two_distinct_blocks"
    return blocks, reason


def sample_indices(blocks, n, repetitions, rng):
    length = blocks.shape[1]
    choices = rng.integers(len(blocks), size=(repetitions, (n + length - 1) // length))
    return blocks[choices].reshape(repetitions, -1)[:, :n]


def phase_interval(means, estimate):
    """A local 95% circular percentile arc, with an explicit abstention rule."""
    result = {"phase_ci_status": "undefined_bootstrap_mean", "phase_ci_low_unwrapped": np.nan,
              "phase_ci_high_unwrapped": np.nan, "phase_ci_low_wrapped": np.nan,
              "phase_ci_high_wrapped": np.nan, "phase_ci_crosses_cut": False,
              "phase_ci_excludes_zero": False, "phase_ci_direction": 0,
              "phase_bootstrap_undefined": int(np.isnan(means).sum()),
              "phase_bootstrap_R": np.nan}
    if not np.isfinite(estimate) or not np.isfinite(means).all():
        return result
    _, result["phase_bootstrap_R"] = circular_mean(means)
    errors = wrap24(means - estimate)
    low, high = np.quantile(errors, [.025, .975], method="linear")
    if not (-6 < low <= high < 6):
        result["phase_ci_status"] = "not_localized_in_open_semicircle"
        return result
    low, high = float(estimate + low), float(estimate + high)
    contains_zero = any(low <= 24 * k <= high for k in (-1, 0, 1))
    # Direction is ambiguous for an arc crossing the antipodal +/-12 h cut.
    crosses = low < -12 or high >= 12
    direction = 0 if contains_zero or crosses else (1 if low > 0 else -1)
    result.update(phase_ci_status="ok", phase_ci_low_unwrapped=low,
                  phase_ci_high_unwrapped=high, phase_ci_low_wrapped=float(wrap24(low)),
                  phase_ci_high_wrapped=float(wrap24(high)), phase_ci_crosses_cut=crosses,
                  phase_ci_excludes_zero=not contains_zero, phase_ci_direction=direction)
    return result


def scheduled_comparisons():
    months = pd.period_range(PARAMETERS["start"], "2020-08", freq="M")
    for month in range(1, 13):
        years = [p.year for p in months if p.month == month]
        for earlier, later in combinations(years, 2):
            yield month, earlier, later, years


def compare(main):
    """Return all planned contrasts, their paired dates, and coverage sensitivity."""
    source = main.copy()
    source["year"] = source.date.dt.year
    source["month_number"] = source.date.dt.month
    source["day_number"] = source.date.dt.day
    rows, pairs = [], []
    for month, y1, y2, years in scheduled_comparisons():
        yearly = {y: source[(source.year == y) & (source.month_number == month)]
                  .set_index("day_number") for y in years}
        common = set.intersection(*(set(d.index) for d in yearly.values()))
        pairwise = set(yearly[y1].index) & set(yearly[y2].index)
        for variant_no, (variant, day_set) in enumerate([
                ("pairwise", pairwise), ("all_years_common", common)]):
            days = np.array(sorted(day_set), dtype=int)
            one, two = yearly[y1].loc[days], yearly[y2].loc[days]
            n = len(days)
            weeks1 = len(one.date.dt.isocalendar().drop_duplicates(["year", "week"]))
            weeks2 = len(two.date.dt.isocalendar().drop_duplicates(["year", "week"]))
            coverage_ok = n >= 10 and min(weeks1, weeks2) >= 2
            da = two[["A_" + j for j in LEVELS]].to_numpy() - one[["A_" + j for j in LEVELS]].to_numpy()
            dh = wrap24(two[["H_" + j for j in LEVELS]].to_numpy() - one[["H_" + j for j in LEVELS]].to_numpy())
            med = np.median(da, axis=0) if n else np.full(3, np.nan)
            mu, r = circular_mean(dh) if n else (np.full(3, np.nan), np.full(3, np.nan))
            fragments = np.split(days, np.flatnonzero(np.diff(days) != 1) + 1) if n else []
            tag = {"month": month, "year1": y1, "year2": y2, "coverage_variant": variant}
            for pos, day in enumerate(days):
                record = dict(tag, day=int(day), date1=one.iloc[pos].date.date().isoformat(),
                              date2=two.iloc[pos].date.date().isoformat())
                for k, level in enumerate(LEVELS):
                    record.update({"dA_"+level: da[pos, k], "dh_"+level: dh[pos, k],
                                   "phase_status1_"+level: one.iloc[pos]["phase_status_"+level],
                                   "phase_status2_"+level: two.iloc[pos]["phase_status_"+level]})
                pairs.append(record)
            for length in PARAMETERS["block_days"]:
                blocks, block_status = calendar_blocks(days, length)
                status = block_status if coverage_ok else "insufficient_calendar_coverage"
                boot_a = boot_h = None
                if status == "ok":
                    rng = np.random.default_rng(np.random.SeedSequence([
                        PARAMETERS["seed"], month, y1, y2, variant_no, length]))
                    indices = sample_indices(blocks, n, PARAMETERS["bootstrap_repetitions"], rng)
                    boot_a = np.median(da[indices], axis=1)
                    boot_h, _ = circular_mean(dh[indices], axis=1)
                for k, level in enumerate(LEVELS):
                    rec = dict(tag, sensor=level, sensor_id=PARAMETERS["sensor_ids"][level],
                               n_pairs=n, iso_weeks_year1=weeks1, iso_weeks_year2=weeks2,
                               coverage_ok=coverage_ok, block_days=length, n_full_blocks=len(blocks),
                               fragment_lengths=";".join(str(len(x)) for x in fragments),
                               paired_days=";".join(map(str, days)), bootstrap_status=status,
                               delta_A=float(med[k]), delta_h=float(mu[k]), phase_R=float(r[k]),
                               A_ci_low=np.nan, A_ci_high=np.nan, A_ci_direction=0,
                               phase_ci_status="bootstrap_unavailable", phase_ci_direction=0,
                               phase_ci_excludes_zero=False,
                               known_sensitive_pairs=int(((one["phase_status_"+level] == "known_sensitive") |
                                                          (two["phase_status_"+level] == "known_sensitive")).sum()),
                               phase_reliability="inherited_not_established")
                    if boot_a is not None:
                        low, high = np.quantile(boot_a[:, k], [.025, .975], method="linear")
                        rec.update(A_ci_low=float(low), A_ci_high=float(high),
                                   A_ci_direction=int(1 if low > 0 else -1 if high < 0 else 0))
                        rec.update(phase_interval(boot_h[:, k], mu[k]))
                    rows.append(rec)
    contrasts, paired = pd.DataFrame(rows), pd.DataFrame(pairs)
    keys = ["month", "year1", "year2", "sensor", "block_days"]
    a = contrasts[contrasts.coverage_variant == "pairwise"]
    b = contrasts[contrasts.coverage_variant == "all_years_common"]
    sensitivity = a.merge(b, on=keys, suffixes=("_pairwise", "_common"))
    sensitivity["delta_A_coverage"] = sensitivity.delta_A_common - sensitivity.delta_A_pairwise
    sensitivity["delta_h_coverage"] = wrap24(sensitivity.delta_h_common - sensitivity.delta_h_pairwise)
    return contrasts, paired, sensitivity


def reproducibility_table(contrasts):
    records = []
    for (month, sensor), group in contrasts.groupby(["month", "sensor"]):
        primary = group[(group.coverage_variant == "pairwise") & (group.block_days == 7)]
        valid = primary[primary.bootstrap_status == "ok"]
        combinations_valid = list(combinations(valid.itertuples(), 2))
        independent = sum(not ({a.year1, a.year2} & {b.year1, b.year2}) for a, b in combinations_valid)
        rec = {"month": month, "sensor": sensor, "planned_year_pairs": len(primary),
               "coverage_eligible_pairs": int(primary.coverage_ok.sum()),
               "primary_bootstrap_pairs": len(valid), "disjoint_year_comparison_pairs": independent,
               "dependency_note": "shared-year comparisons are dependent"}
        for kind in ("A", "phase"):
            field = kind + "_ci_direction"
            directions = valid[field].astype(int)
            rec[kind+"_positive"] = int((directions == 1).sum())
            rec[kind+"_negative"] = int((directions == -1).sum())
            rec[kind+"_zero_or_unresolved"] = int((directions == 0).sum())
            # Descriptive concordance, not an independent replication or global test.
            rec[kind+"_concordant_primary"] = bool(len(valid) >= 2 and
                (directions.eq(1).all() or directions.eq(-1).all()))
        records.append(rec)
    return pd.DataFrame(records)

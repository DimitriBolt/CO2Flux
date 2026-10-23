"""Offline stage 0.2 candidate admission at Dimitri Bolt's 2.5 threshold.

The frozen strict-8 calculation, point masks and original archives are inputs.
Only the monthly amplitude inequality changes. No Fourier values are calculated.
Run from the repository root with python3 -B and this file's relative path.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage02_final_admission as strict
from chapter02 import _monthly_median_bounds, clean_admission_measurements, monthly_admission

ROOT = strict.ROOT
OUT = ROOT / 'Project_description/Research_log/output/stage02_candidate_selection'
LOCAL = ROOT / 'Project_description/Research_log/data/stage02_candidate_selection'
BASE_COMMIT = '9c677cacd2a6a1b008a3cc7b73aa30b63e998c8d'
THRESHOLD = 2.5


def classify_months(metrics, threshold):
    """Reuse frozen monthly bounds; the sole parameterized inequality is A >= a.

    This is the exact decision branch of chapter02.monthly_admission. Its daily
    fits and optional-sample median bounds are not recalculated with new rules.
    """
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Positive finite amplitude threshold required')
    result = metrics.copy(deep=True)
    for index, row in result.iterrows():
        reasons = []
        if row.valid_days < 10:
            status = 'fail'
            reasons.append('fewer_than_10_valid_days')
        else:
            clo, chi, alo, ahi = (row[k] for k in (
                'concentration_lower', 'concentration_upper', 'amplitude_lower', 'amplitude_upper'))
            if not np.isfinite([clo, chi, alo, ahi]).all():
                raise ValueError('Missing monthly uncertainty bounds')
            if chi < 100 or clo > 2000:
                reasons.append('median_concentration_outside_100_2000')
            if ahi < threshold:
                reasons.append(f'median_raw_amplitude_below_{threshold:g}')
            if reasons:
                status = 'fail'
            elif row.minimum_valid_days >= 10 and clo >= 100 and chi <= 2000 and alo >= threshold:
                status = 'pass'
                reasons.append('all_three_criteria_invariant_to_flagged_samples')
            else:
                status = 'undetermined'
                reasons.append('unresolved_excursions_can_affect_monthly_gate')
        result.loc[index, 'status'] = status
        result.loc[index, 'reason'] = ';'.join(reasons)
    return result


def reproduce_strict(metrics, daily):
    """Check all frozen decisions and reconstruct all monthly metrics from days."""
    rebuilt = classify_months(metrics, 8)
    if not rebuilt[['status', 'reason']].equals(metrics[['status', 'reason']]):
        raise ValueError('Strict-8 decisions no longer reproduce')
    numeric_checks = 0
    for row in metrics.itertuples():
        d = daily.loc[daily.month.eq(row.month) & daily.sensorid.eq(row.sensorid) & daily.valid].copy()
        if len(d) != row.valid_days:
            raise ValueError('Strict valid-day count mismatch')
        d['guaranteed'] = d.guaranteed.astype(bool)
        values = dict(minimum_valid_days=int(d.guaranteed.sum()),
                      doubtful_valid_days=int(d.doubtful_samples.gt(0).sum()),
                      median_concentration=d.concentration.median() if len(d) else np.nan,
                      median_raw_amplitude=d.amplitude.median() if len(d) else np.nan)
        clo, chi = _monthly_median_bounds(d, 'c') if len(d) >= 10 else (np.nan, np.nan)
        alo, ahi = _monthly_median_bounds(d, 'a') if len(d) >= 10 else (np.nan, np.nan)
        values.update(concentration_lower=clo, concentration_upper=chi, amplitude_lower=alo, amplitude_upper=ahi)
        for key, value in values.items():
            if not np.isclose(value, getattr(row, key), rtol=0, atol=1e-11, equal_nan=True):
                raise ValueError(f'Strict monthly metric mismatch: {row.month}/{row.sensorid}/{key}')
            numeric_checks += 1
    return dict(sensor_month_decisions=len(metrics), numeric_metrics_checked=numeric_checks,
                decision_and_reason_equality='exact', numeric_csv_roundtrip_atol=1e-11,
                source='Frozen daily fits and uncertainty bounds; no raw re-cleaning or new fit')


def candidate_points(work, start, end):
    """A separate view, with no mutation of the strict measurement decisions."""
    return work.loc[work.technically_retained & ~work.doubtful &
                    work.localdatetime.ge(start) & work.localdatetime.lt(end)].copy()


def candidate_grid(works, start, end):
    """Calendar 2h cells; absent concentrations remain NaN, never zeros."""
    points = {sid: candidate_points(work, start, end) for sid, work in works.items()}
    grid = strict.make_grid(points, start, end)
    for sid, frame in points.items():
        means = frame.groupby(frame.localdatetime.dt.floor('2h')).datavalue.mean()
        grid[f'C_{sid}'] = means.reindex(grid.index)
        if not grid[f'C_{sid}'].notna().equals(grid[f'n_{sid}'].gt(0)):
            raise ValueError('Occupancy and actual-concentration means disagree')
    grid['bin_center'] = grid.index + pd.Timedelta(hours=1)
    return grid


def month_runs(months):
    """Calendar adjacency, not a count of all positive months."""
    periods = sorted(set(pd.Period(m, freq='M') for m in months))
    runs = []
    for month in periods:
        if not runs or month.ordinal != runs[-1][-1].ordinal + 1:
            runs.append([])
        runs[-1].append(month)
    return [[str(m) for m in run] for run in runs]


def eligible_months(joint, grid, start, end):
    result = joint.copy()
    common = grid.loc[grid.complete].groupby(grid.loc[grid.complete].index.strftime('%Y-%m')).size()
    result['common_2h_bins'] = result.month.map(common).fillna(0).astype(int)
    periods = pd.PeriodIndex(result.month, freq='M')
    result['full_calendar_month'] = (periods.start_time >= start) & ((periods + 1).start_time <= end)
    result['eligible'] = result.status.eq('pass') & result.common_2h_bins.gt(0) & result.full_calendar_month
    return result


def longest_missing_cells(occupied):
    empty = ~np.asarray(occupied, dtype=bool)
    bounds = np.diff(np.r_[False, empty, False].astype(int))
    starts, ends = np.flatnonzero(bounds == 1), np.flatnonzero(bounds == -1)
    return int((ends - starts).max()) if len(starts) else 0


def season(timestamp):
    t = pd.Timestamp(timestamp)
    name = ('DJF' if t.month in (12, 1, 2) else 'MAM' if t.month in (3, 4, 5)
            else 'JJA' if t.month in (6, 7, 8) else 'SON')
    return f'{t.year + (t.month == 12)}-{name}'


def check_inputs():
    passport = json.loads((strict.OUT / 'passport.json').read_text())
    paths = {}
    for key in ('input_sha256', 'derived_local_sha256', 'implementation_sha256', 'reference_input_sha256'):
        paths.update(passport[key])
    manifest = json.loads((strict.ARCHIVE / 'manifest.json').read_text())
    summary = json.loads((strict.ARCHIVE / 'diagnostics/summary.json').read_text())
    channels = manifest['channels']
    if (manifest['status'] != 'complete' or len(channels) != 53 or summary['verticals'] != 16
            or len({c['sensorid'] for c in channels}) != 53
            or sum(c['kind'] == 'GMM222' for c in channels) != 48
            or sum(c['kind'] == 'LI-COR' for c in channels) != 5):
        raise ValueError('Wrong completed archive or channel/vertical inventory')
    for rel, sha in summary['files_sha256'].items():
        paths[str((strict.ARCHIVE / 'diagnostics' / rel).relative_to(ROOT))] = sha
    # Freeze all strict outputs and all existing 0.3/0.4 files, not just passports.
    protected = [strict.OUT, ROOT / 'Project_description/Research_log/output/preliminary_spectra',
                 ROOT / 'Project_description/Research_log/output/preliminary_fourier_phases']
    for directory in protected:
        for path in directory.rglob('*'):
            if path.is_file():
                paths.setdefault(str(path.relative_to(ROOT)), strict.digest(path))
    for rel, expected in paths.items():
        if strict.digest(ROOT / rel) != expected:
            raise ValueError(f'Frozen input checksum mismatch: {rel}')
    return passport, manifest, paths


def inspect_short_alternative(record, metadata):
    """Only the four months with any common data require monthly fits here.

    The all-history diagnostic rules out a 13-month run before any fits. This
    small calculation supplies exact comparison counts, not a second search.
    """
    ids = (int(record.air_sensorid), *map(int, record.basalt_sensorids.split(',')))
    start = pd.Timestamp(record.retained_first_day).normalize()
    end = pd.Timestamp(record.retained_last_day).normalize() + pd.Timedelta(days=1)
    first_month = start.to_period('M').start_time
    end_month = (end.to_period('M') + 1).start_time
    works, parts = {}, []
    for sid in ids:
        c = metadata[sid]
        path = strict.ARCHIVE / f'diagnostics/sensors/{sid}_working.parquet'
        work = pd.read_parquet(path)
        if c['kind'] == 'LI-COR':
            old_retained = set(work.loc[work.technically_retained, 'localdatetime'])
            source = pd.read_parquet(strict.ARCHIVE / f'diagnostics/sensors/{sid}_source_decisions.parquet')
            raw = source[['sensorid', 'variableid', 'localdatetime']].copy()
            raw['datavalue'] = source.original_value.to_numpy(float)
            work = clean_admission_measurements(raw, {str(sid): sid}, strict.EMPTY_EXCLUSIONS,
                channel_metadata={str(sid): {k: c[k] for k in ('variableid', 'units', 'table', 'sensorcode')}})
            work['technically_retained'] = work.exclusion_reason.eq('')
            work['doubtful'] = work.doubt_reason.ne('') & work.technically_retained
            work['plateau_review'] = work.constant_run_hours.ge(1) & work.technically_retained
            if not set(work.loc[work.technically_retained, 'localdatetime']).issubset(old_retained):
                raise ValueError('Uniform air cleaning unexpectedly restores an old excluded time')
        works[sid] = work
        retained = work.loc[work.technically_retained & work.localdatetime.ge(first_month) & work.localdatetime.lt(end_month)]
        months, _ = monthly_admission(retained, {str(sid): sid}, first_month, end_month)
        parts.append(classify_months(months, THRESHOLD))
    months = pd.concat(parts, ignore_index=True)
    # make_grid has fixed selected IDs; relabel only dictionary keys for occupancy,
    # retaining actual identities and columns in the monthly comparison outputs.
    grid = candidate_grid(dict(zip(strict.IDS, works.values())), first_month, end_month)
    joint = eligible_months(strict.joint_months(months, ids), grid, first_month, end_month)
    runs = month_runs(joint.loc[joint.eligible, 'month'])
    return months, dict(joint_pass_months=int(joint.eligible.sum()),
        max_consecutive_months=max(map(len, runs), default=0),
        common_2h_bins=int(grid.complete.sum()), candidate_windows=0,
        decision='insufficient_all_history_overlap_for_13_months',
        monthly_evaluation_scope='Only months with possible common observations; other months fail real-common-data requirement')


def run(out=OUT, local=LOCAL):
    frozen, manifest, hashes = check_inputs()
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(strict.OUT / 'channel_months.csv')
    daily = pd.read_csv(strict.OUT / 'channel_days.csv')
    reproduction = reproduce_strict(metrics, daily)
    months = classify_months(metrics, THRESHOLD)
    months.to_csv(out / 'channel_months.csv', index=False)
    works = {sid: pd.read_parquet(strict.LOCAL / f'{sid}_measurements.parquet') for sid in strict.IDS}
    start, end = pd.Timestamp(frozen['full_history_start']), pd.Timestamp(frozen['full_history_end_exclusive'])
    grid = candidate_grid(works, start, end)
    joint = eligible_months(strict.joint_months(months), grid, start, end)
    joint.to_csv(out / 'joint_months.csv', index=False)
    runs = sorted(month_runs(joint.loc[joint.eligible, 'month']), key=lambda r: (-len(r), r[0]))
    if not runs or len(runs[0]) < 13:
        raise ValueError('Selected candidate does not have 13 consecutive full months')
    selected = runs[0]
    begin = pd.Period(selected[0], freq='M').start_time
    finish = (pd.Period(selected[-1], freq='M') + 1).start_time
    grid = grid.loc[(grid.index >= begin) & (grid.index < finish)].copy()
    previous = strict.make_grid(works, begin, finish)
    before_bins = int(previous.complete.sum())
    grid.to_parquet(local / 'calendar_2h_grid.parquet')
    grid.loc[grid.complete].to_parquet(local / 'common_2h_input.parquet')
    missing = grid.index[~grid.complete]
    groups = (missing.to_series().diff() != pd.Timedelta(hours=2)).cumsum()
    gap_runs = [dict(start=str(part.iloc[0]), end_exclusive=str(part.iloc[-1] + pd.Timedelta(hours=2)),
                     missing_2h_cells=len(part), missing_hours=2 * len(part))
                for _, part in missing.to_series().groupby(groups)]
    pd.DataFrame(gap_runs, columns=['start', 'end_exclusive', 'missing_2h_cells', 'missing_hours']).to_csv(out / 'common_gap_runs.csv', index=False)
    windows = strict.classify_windows(grid, months)
    windows['center'] = pd.to_datetime(windows.start) + pd.Timedelta(days=15)
    windows['center_season'] = windows.center.map(season)
    windows['common_days'] = [int(grid.loc[(grid.index >= r.start) & (grid.index < r.end_exclusive) & grid.complete].index.normalize().nunique()) for r in windows.itertuples()]
    windows['max_missing_2h_cells'] = [longest_missing_cells(grid.loc[(grid.index >= r.start) & (grid.index < r.end_exclusive), 'complete']) for r in windows.itertuples()]
    windows.to_csv(out / 'window_coverage_30d_1d.csv', index=False)
    if not windows.admitted.all() or windows.has_unresolved_excursions.any():
        raise ValueError('A selected candidate window is not technically admitted')
    counts, reasons, gaps = [], [], []
    for sid, work in works.items():
        scope = work.loc[work.localdatetime.ge(begin) & work.localdatetime.lt(finish)]
        points = candidate_points(work, begin, finish)
        if points.doubtful.any() or not points.technically_retained.all():
            raise ValueError('Forbidden measurement in candidate input')
        points = points.rename(columns={'month_status': 'strict8_month_status', 'in_admitted_month': 'strict8_in_admitted_month'})
        statuses = months.loc[months.sensorid.eq(sid)].set_index('month').status
        points['candidate_month_status'] = points.month.map(statuses)
        if not points.candidate_month_status.eq('pass').all():
            raise ValueError('Candidate point from a non-pass month')
        points.to_parquet(local / f'{sid}_candidate_measurements.parquet', index=False)
        source = pd.read_parquet(strict.LOCAL / f'{sid}_source_decisions.parquet')
        source = source.loc[source.localdatetime.ge(begin) & source.localdatetime.lt(finish)]
        full_counts = next(r for r in frozen['channel_summary'] if r['sensorid'] == sid and r['period'] == 'full_history')
        counts.append(dict(sensorid=sid, level=strict.LEVELS[sid], raw_source_rows=len(source),
            unique_times=len(scope), point_excluded_unique=int((~scope.technically_retained).sum()),
            point_excluded_source_rows=int((~source.technically_retained).sum()),
            retained_unique=int(scope.technically_retained.sum()), doubtful_removed=int((scope.technically_retained & scope.doubtful).sum()),
            candidate_measurements=len(points), exact_duplicate_extra=int(scope.exact_duplicate_extra.sum()),
            remaining_plateau_review=int(points.plateau_review.sum()),
            full_history_raw_source_rows=int(full_counts['raw_source_rows']),
            occupied_bins_before_doubt_removal=int(previous[f'n_{sid}'].gt(0).sum()),
            occupied_bins_after_doubt_removal=int(grid[f'n_{sid}'].gt(0).sum())))
        for column in ('exclusion_reason', 'doubt_reason'):
            for reason, n in scope.loc[scope[column].ne(''), column].value_counts().items():
                reasons.append(dict(sensorid=sid, decision_field=column, reason=reason, unique_times=int(n)))
        times = points.localdatetime.sort_values()
        diffs = times.diff().dt.total_seconds().div(3600)
        index = diffs.idxmax()
        gaps.append(dict(sensorid=sid, max_observed_interval_hours=float(diffs.max()),
                         previous_observation=str(times.loc[:index].iloc[-2]), next_observation=str(times.loc[index]),
                         max_missing_2h_cells=longest_missing_cells(grid[f'n_{sid}'].gt(0))))
    pd.DataFrame(counts).to_csv(out / 'channel_counts.csv', index=False)
    pd.DataFrame(reasons).to_csv(out / 'measurement_reasons.csv', index=False)
    pd.DataFrame(gaps).to_csv(out / 'channel_gaps.csv', index=False)
    coverage = []
    labels = grid.index.strftime('%Y-%m')
    for month in selected:
        part = grid.loc[labels == month]
        coverage.append(dict(month=month, possible_days=part.index.normalize().nunique(),
            common_days=part.loc[part.complete].index.normalize().nunique(), possible_2h_bins=len(part),
            common_2h_bins=int(part.complete.sum()), before_doubt_removal_bins=int(previous.loc[labels == month, 'complete'].sum()),
            max_missing_2h_cells=longest_missing_cells(part.complete)))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(out / 'month_coverage.csv', index=False)
    seasons = []
    for label, part in grid.groupby(grid.index.map(season), sort=False):
        wins = windows.loc[windows.center_season.eq(label)]
        seasons.append(dict(season=label, observed_period_start=str(part.index[0]), observed_period_end_exclusive=str(part.index[-1]+pd.Timedelta(hours=2)),
            possible_days=part.index.normalize().nunique(), common_days=part.loc[part.complete].index.normalize().nunique(),
            possible_2h_bins=len(part), common_2h_bins=int(part.complete.sum()), windows_by_center=int(wins.admitted.sum()),
            max_missing_2h_cells=longest_missing_cells(part.complete)))
    pd.DataFrame(seasons).to_csv(out / 'season_coverage.csv', index=False)
    inventory = pd.read_csv(strict.ARCHIVE / 'diagnostics/vertical_candidates.csv')
    distances = pd.read_csv(strict.ARCHIVE / 'diagnostics/air_distances.csv')
    metadata = {c['sensorid']: c for c in manifest['channels']}
    seen_basalt = set()
    comparisons = []
    for record in inventory.itertuples():
        nearest = distances.loc[distances.vertical.eq(record.vertical) & distances.nearest, 'air_sensorid']
        if len(nearest) != 1 or int(nearest.iloc[0]) != record.air_sensorid:
            raise ValueError('Ambiguous or inconsistent documented nearest air channel')
        basalt_ids = list(map(int, record.basalt_sensorids.split(',')))
        seen_basalt.update(basalt_ids)
        actual_depths = []
        for sid in basalt_ids:
            c = metadata[sid]
            if (c['kind'], c['table'], c['variableid'], c['units'], c['x'], c['y']) != (
                    'GMM222', 'LEO_WEST.DATAVALUES', 9, 'ppm', record.x, record.y):
                raise ValueError('Basalt inventory/geometry mismatch')
            actual_depths.append(round(-100 * c['z']))
        if actual_depths != list(map(int, record.depths_cm.split(','))):
            raise ValueError('Actual depth mismatch')
        air = [c for c in metadata.values() if c['kind'] == 'LI-COR']
        actual_distances = {c['sensorid']: np.hypot(c['x']-record.x, c['y']-record.y) for c in air}
        if actual_distances[record.air_sensorid] != min(actual_distances.values()):
            raise ValueError('A more distant air sensor has been substituted')
        if not np.isclose(actual_distances[record.air_sensorid], record.air_distance_m, rtol=0, atol=1e-12):
            raise ValueError('Documented distance mismatch')
        base = dict(vertical=record.vertical, basalt_sensorids=record.basalt_sensorids, depths_cm=record.depths_cm,
            air_sensorid=int(record.air_sensorid), air_distance_m=record.air_distance_m,
            raw_common_first_day=record.raw_first_day, raw_common_last_day=record.raw_last_day,
            retained_common_first_day=record.retained_first_day, retained_common_last_day=record.retained_last_day,
            original_retained_common_days=record.common_retained_days,
            original_retained_common_bins=record.common_retained_2h_bins)
        if record.vertical == 'W_x1_y4':
            base.update(joint_pass_months=int(joint.eligible.sum()), max_consecutive_months=len(selected),
                common_2h_bins=int(grid.complete.sum()), candidate_windows=int(windows.admitted.sum()),
                decision='selected', monthly_evaluation_scope='Full 158-month archive calendar; cell/window counts are for selected run')
        elif record.common_retained_days == 0:
            base.update(joint_pass_months=0, max_consecutive_months=0, common_2h_bins=0, candidate_windows=0,
                decision='no_real_common_observations_after_original_point_cleaning',
                monthly_evaluation_scope='Screened: zero months can satisfy the required real-common-data condition; independent month gates not fitted')
        else:
            alternative, result = inspect_short_alternative(record, metadata)
            alternative.to_csv(out / f'{record.vertical}_channel_months.csv', index=False)
            base.update(result)
        comparisons.append(base)
    pd.DataFrame(comparisons).to_csv(out / 'vertical_comparison.csv', index=False)
    if len(comparisons) != 16 or seen_basalt != {c['sensorid'] for c in metadata.values() if c['kind'] == 'GMM222'}:
        raise ValueError('Not all 16 verticals were screened')
    for rel, sha in hashes.items():
        if strict.digest(ROOT / rel) != sha:
            raise ValueError(f'Input changed during candidate calculation: {rel}')
    passport = dict(stage='0.2 candidate selection, no new Fourier calculation', source_commit=BASE_COMMIT,
        decision_date_arizona='2026-10-23', amplitude_min=THRESHOLD,
        threshold_authority='Dimitri Bolt candidate decision; NOT the original I.R. Gabitov 8-unit criterion',
        strict_result_preserved=str((strict.OUT / 'passport.json').relative_to(ROOT)), strict_reproduction=reproduction,
        metadata=frozen['metadata'], point_and_daily_parameters=frozen['numerical_parameters'],
        monthly_parameter_override={'monthly_amplitude_min': THRESHOLD},
        full_history_start=str(start), full_history_end_exclusive=str(end),
        verticals_checked=len(comparisons), selected_vertical='W_x1_y4',
        selected_start_inclusive=str(begin), selected_end_exclusive=str(finish), selected_months=selected,
        selected_days=len(grid)//12, common_days=int(grid.loc[grid.complete].index.normalize().nunique()),
        possible_2h_bins=len(grid), common_2h_bins=int(grid.complete.sum()),
        common_2h_before_doubt_removal=before_bins, common_bins_lost_to_doubts=before_bins-int(grid.complete.sum()),
        candidate_windows=int(windows.admitted.sum()), minimum_window_common_bins=int(windows.common_bins.min()),
        maximum_window_common_bins=int(windows.common_bins.max()),
        maximum_missing_common_2h_cells=longest_missing_cells(grid.complete),
        selector='technically_retained & ~doubtful',
        binning='Actual means in fixed calendar 2h bins, real centers; NaN for absence; no interpolation, zero concentration or carry-forward',
        window_precheck='Existing >=2 finite common bins, 30d/1d; no coefficients or phase reliability evaluated',
        comparison_semantics='Pass-month counts include the real-common-data requirement. 14 zero-overlap configurations need no monthly fits; the short alternative cannot satisfy 13 months. Windows count only within >=13-month qualifying runs.',
        seasonal_window_assignment='Calendar window center; overlapping windows are not independent samples; edge seasons are partial',
        monthly_uncertainty='Original retained daily metrics and all-subset uncertainty bounds; candidate-point exclusion is a separate view',
        time_base=frozen['time_base'], height_note=frozen['height_note'],
        limitations=['Candidate threshold 2.5, not strict Gabitov admission at 8',
            'Technical computability does not establish nonzero Fourier modulus or stable phase',
            'Real missing days/cells and partial edge seasons remain; no new coverage cutoff',
            'Air is horizontally offset by 1m; unresolved height metadata retained',
            'Dimitri Bolt decides closure of 0.2; stage 0.5, irrigation record and I.G. approval remain'],
        input_sha256=hashes,
        local_derived_sha256={str(p.relative_to(ROOT)):strict.digest(p) for p in sorted(local.glob('*.parquet'))},
        implementation_sha256={str(Path(__file__).relative_to(ROOT)):strict.digest(__file__)})
    strict.save_json(out / 'passport.json', passport)
    print(json.dumps({k:passport[k] for k in ('verticals_checked','selected_vertical','selected_start_inclusive',
        'selected_end_exclusive','selected_days','common_days','common_2h_bins','common_bins_lost_to_doubts',
        'candidate_windows','minimum_window_common_bins','maximum_missing_common_2h_cells')}, indent=2))
    return passport


if __name__ == '__main__':
    run()

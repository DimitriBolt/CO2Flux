"""Offline technical preparation of W_R4_C-4; no fits, spectra or imputation.

Run once into a new directory: python3 -B stage02_prepare.py [--output PATH].
All dates are unchanged source clock labels; common bins are provisional.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from chapter02 import clean_admission_measurements
from stage02_coverage import sha, utc_now, window_metrics, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'data/ch02/archive/raw.parquet'
AIR = HERE / 'data/stage0/air_1275/raw.parquet'
EXCLUSIONS = HERE / 'output/ch02/W_R4_C-4_full_archive_final_checks.json'
SOURCE_SHA = {
    BASE: '89e69150734e8ae18ce00ebb554e6cd0dfc605172d54af2febf71c1c086ab40a',
    AIR: '40d3cb63c6f501567735d3f01eb44ff3edf376ba53aaa81472a4bb018ee921a7',
}
START, END = pd.Timestamp('2025-08-13'), pd.Timestamp('2026-10-01')
PILOT_START, PILOT_END = pd.Timestamp('2025-08-14'), pd.Timestamp('2025-09-13')
CHANNELS = {
    'C_air': dict(sensorid=1275, variableid=56, table='leo_west.datavalueslicor',
                  sensorcode='LEO-W_4_0_1_LI-7000', units='umol/mol', xyz=[0, 4, .25]),
    'C_5': dict(sensorid=994, variableid=9, table='leo_west.datavalues',
                sensorcode='LEO-W_4_-4_1_GMM222', units='ppm', xyz=[-4, 4, -.05]),
    'C_20': dict(sensorid=1010, variableid=9, table='leo_west.datavalues',
                 sensorcode='LEO-W_4_-4_2_GMM222', units='ppm', xyz=[-4, 4, -.20]),
    'C_35': dict(sensorid=1026, variableid=9, table='leo_west.datavalues',
                 sensorcode='LEO-W_4_-4_3_GMM222', units='ppm', xyz=[-4, 4, -.35]),
}
AIR_DOUBT = 'LI_COR_range_and_artifact_rules_unconfirmed'


def clip(frame, start, end):
    return frame.loc[frame.localdatetime.ge(start) & frame.localdatetime.lt(end)].copy()


def with_context(frame, start, end):
    """Four unique neighbours on each edge suffice for seven-point MAD + adjacency.

    Not the old +/-12 h gate. Output remains strictly [start,end).
    Constant duration at the outer context edge is only a diagnostic lower bound.
    """
    times = np.sort(frame.localdatetime.unique())
    lo, hi = np.searchsorted(times, [start.to_datetime64(), end.to_datetime64()])
    selected = times[max(0, lo-4):min(len(times), hi+4)]
    return frame.loc[frame.localdatetime.isin(selected)].copy()


def flags(work):
    work['excluded'] = work.exclusion_reason.ne('')
    work['doubtful'] = work.doubt_reason.ne('') & ~work.excluded
    work['technically_retained'] = ~work.excluded
    work['status'] = np.select([work.excluded, work.doubtful],
                              ['excluded', 'doubtful_retained'], default='retained')
    return work


def prepare_basalt(raw, start, end, exclusions):
    """Call only the approved pointwise cleaner, never monthly admission/H/A."""
    views, audits = {}, {}
    for level, name in zip(('D1', 'D2', 'D3'), ('C_5', 'C_20', 'C_35')):
        source = raw.loc[raw.sensorid.eq(CHANNELS[name]['sensorid'])].copy()
        context = with_context(source, start, end)
        work = clean_admission_measurements(context, {level: CHANNELS[name]['sensorid']}, exclusions)
        work = flags(clip(work, start, end))
        work['channel'] = name
        work['above_3000'] = work.original_value.gt(3000)
        # Link every raw row (including conflicting values) to its decision.
        audit = clip(source, start, end).merge(
            work[['localdatetime', 'exclusion_reason', 'doubt_reason', 'excluded',
                  'doubtful', 'technically_retained', 'status']],
            on='localdatetime', how='left', validate='many_to_one')
        audit['above_3000'] = audit.datavalue.gt(3000)
        audit['exact_duplicate_extra'] = audit.duplicated(['localdatetime', 'datavalue'])
        if audit.excluded.isna().any():
            raise AssertionError('Unmapped raw observation')
        views[name], audits[name] = work, audit
    return views, audits


def prepare_air(raw, start, end):
    """Keep Decimal values and every timestamp/ID. No borrowed basalt thresholds.

    No numeric LI-COR valid band or artefact detector is justified locally.
    Remaining finite observations are retained with an explicit review flag;
    this is not a claim that every such reading is anomalous. Negative/zero
    readings and repeated clocks get additional reasons, not guessed cutoffs.
    """
    work = clip(raw, start, end).sort_values('localdatetime', kind='stable')
    if work.localdatetime.isna().any():
        raise ValueError('Undefined LI-COR timestamp')
    work['original_value'] = work.datavalue
    finite = work.datavalue.map(lambda v: False if pd.isna(v) else
                               bool(v.is_finite()) if hasattr(v, 'is_finite') else bool(np.isfinite(v))).astype(bool)
    service = work.datavalue.map(lambda v: False if pd.isna(v) else v <= -9999).astype(bool) & finite
    work['exclusion_reason'] = np.select([service, ~finite], ['service_code', 'nonfinite'], default='')
    work['doubt_reason'] = np.where(work.exclusion_reason.eq(''), AIR_DOUBT, '')
    for condition, reason in (
        (finite & work.datavalue.le(0) & ~service, 'nonpositive_air_value'),
        (work.localdatetime.duplicated(keep=False), 'repeated_air_timestamp_unresolved'),
    ):
        use = condition & work.exclusion_reason.eq('')
        work.loc[use, 'doubt_reason'] += ';' + reason
    work = flags(work)
    work.loc[work.excluded, 'datavalue'] = None
    work['channel'] = 'C_air'
    audit = work.drop(columns=['original_value', 'channel']).copy()
    audit['datavalue'] = work.original_value
    return work, audit


def prepare(raw_basalt, raw_air, start, end, exclusions):
    views, audits = prepare_basalt(raw_basalt, start, end, exclusions)
    views['C_air'], audits['C_air'] = prepare_air(raw_air, start, end)
    return views, audits


def count_grid(views, audits, start, end, hours):
    """Counts, not concentrations. No aggregation of values or synthetic samples."""
    step = pd.Timedelta(hours=hours)
    index = pd.date_range(start, end, freq=step, inclusive='left')
    out = pd.DataFrame({'start': index, 'end_exclusive': (index+step).where(index+step <= end, end)})
    for name in CHANNELS:
        for kind, frame in (('raw', audits[name]), ('retained', views[name].loc[views[name].technically_retained]),
                            ('without_doubts', views[name].loc[views[name].technically_retained & ~views[name].doubtful]),
                            ('doubtful', views[name].loc[views[name].doubtful])):
            bucket = ((frame.localdatetime-start) // step).astype(int)
            out[name+'_'+kind] = bucket.value_counts().reindex(range(len(index)), fill_value=0).to_numpy()
        # Separate unique clock labels from raw records (duplicates are not extra coverage).
        unique = audits[name].drop_duplicates('localdatetime')
        bucket = ((unique.localdatetime-start) // step).astype(int)
        out[name+'_raw_unique_times'] = bucket.value_counts().reindex(range(len(index)), fill_value=0).to_numpy()
    for kind in ('raw', 'retained', 'without_doubts'):
        out['common_'+kind] = out[[n+'_'+kind for n in CHANNELS]].gt(0).all(axis=1)
    return out


def grid_runs(grid, kind):
    active = grid['common_'+kind]
    labels = active.ne(active.shift()).cumsum()
    result = []
    for _, part in grid.loc[active].groupby(labels[active]):
        result.append(dict(start=part.start.iloc[0], end_exclusive=part.end_exclusive.iloc[-1], bins=len(part)))
    return pd.DataFrame(result, columns=['start', 'end_exclusive', 'bins'])


def summarize(views, audits, start, end):
    rows, steps, gaps = [], [], []
    for name in CHANNELS:
        work, source = views[name], audits[name]
        retained = work.loc[work.technically_retained]
        for state, frame in (('raw', source), ('retained', retained)):
            times = frame.localdatetime.sort_values().drop_duplicates()
            dt = times.diff().dt.total_seconds().dropna()
            counts = dt[dt.gt(0)].value_counts()
            metrics = window_metrics(frame, start, end)
            rows.append(dict(channel=name, state=state, **metrics,
                             unique_times=len(times), days_with_records=times.dt.normalize().nunique(),
                             median_step_s=dt.median(), modal_step_s=counts.index[0] if len(counts) else None,
                             min_value=frame.datavalue.min(), max_value=frame.datavalue.max(),
                             excluded_unique_times=work.loc[work.excluded, 'localdatetime'].nunique(),
                             doubtful_rows=int(work.doubtful.sum()),
                             source_rows_above_3000=int(source.datavalue.gt(3000).sum()) if name != 'C_air' else None))
            steps.extend(dict(channel=name, state=state, seconds=float(t), count=int(n)) for t,n in counts.items())
            # Keep ALL positive spacings; a "gap" label does not impose an admission threshold.
            gaps.extend(dict(channel=name, state=state, previous=prev, following=nxt,
                             seconds=(nxt-prev).total_seconds()) for prev,nxt in zip(times.iloc[:-1],times.iloc[1:]))
    return pd.DataFrame(rows), pd.DataFrame(steps), pd.DataFrame(gaps)


def candidate_windows(grids, views):
    """30-day daily-start candidates ranked only by simultaneous real-bin presence.

    Equal coverage gets an equal dense rank; chronology only orders display.
    No validity, amplitude or spectral threshold is introduced.
    """
    rows = []
    for start in pd.date_range(START, END-pd.Timedelta(days=30), freq='D'):
        end = start+pd.Timedelta(days=30)
        row = dict(start=start, end_exclusive=end)
        for hours, grid in grids.items():
            part = grid.loc[grid.start.ge(start) & grid.end_exclusive.le(end)]
            for kind in ('raw', 'retained', 'without_doubts'):
                row[f'common_{kind}_{hours}h_bins'] = int(part['common_'+kind].sum())
            for name in CHANNELS:
                row[f'{name}_occupied_{hours}h_bins'] = int(part[name+'_retained'].gt(0).sum())
        for name, frame in views.items():
            part = clip(frame, start, end)
            row[name+'_retained_rows'] = int(part.technically_retained.sum())
            row[name+'_excluded_rows'] = int(part.excluded.sum())
            row[name+'_doubtful_rows'] = int(part.doubtful.sum())
        rows.append(row)
    out = pd.DataFrame(rows)
    # Both diagnostics are displayed, with 2h then 1h presence as ordering keys.
    scores = list(zip(out.common_retained_2h_bins, out.common_retained_1h_bins))
    ranks = {score:i+1 for i,score in enumerate(sorted(set(scores), reverse=True))}
    out['coverage_rank'] = [ranks[s] for s in scores]
    out['has_any_common_retained_bin'] = out.common_retained_2h_bins.gt(0)
    return out.sort_values(['coverage_rank', 'start'])


def save_period(output, views, audits, start, end):
    output.mkdir()
    for name in CHANNELS:
        for prefix, frame in (('', views[name]), ('source_flags_', audits[name]),
                              ('retained_', views[name].loc[views[name].technically_retained])):
            path = output/(prefix+name+'.parquet')
            frame.to_parquet(path, index=False)
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_parquet(path))
    stats, steps, gaps = summarize(views, audits, start, end)
    stats.to_csv(output/'channels.csv', index=False)
    steps.to_csv(output/'positive_intervals.csv', index=False)
    gaps.to_csv(output/'observation_spacings.csv', index=False)
    reasons = []
    for name, work in views.items():
        for kind in ('exclusion_reason', 'doubt_reason'):
            reasons.extend(dict(channel=name, kind=kind, reason=reason, rows=int(n))
                           for reason,n in work[kind].value_counts().items() if reason)
    pd.DataFrame(reasons).to_csv(output/'reasons.csv', index=False)
    grids = {}
    for hours in (1,2):
        grid = count_grid(views, audits, start, end, hours)
        grid.to_csv(output/f'grid_{hours}h_counts.csv', index=False)
        for kind in ('raw','retained','without_doubts'):
            grid_runs(grid,kind).to_csv(output/f'common_{kind}_{hours}h_runs.csv', index=False)
        grids[hours] = grid
    return grids, stats


def run(output):
    output = Path(output)
    for path, expected in SOURCE_SHA.items():
        if sha(path) != expected:
            raise ValueError(f'Input checksum mismatch: {path}')
    basalt, air = pd.read_parquet(BASE), pd.read_parquet(AIR)
    for frame, sensors, var in ((basalt, {994,1010,1026},9),(air,{1275},56)):
        if set(frame.sensorid.unique()) != sensors or not frame.variableid.eq(var).all():
            raise ValueError('Unexpected channel identity')
        if frame.localdatetime.isna().any() or frame.localdatetime.dt.tz is not None:
            raise ValueError('Unexpected timestamp basis')
        frame['source_row'] = np.arange(len(frame))
    exclusions = pd.DataFrame(json.loads(EXCLUSIONS.read_text())['approved_exclusions'])
    for col in ('start','end'):
        exclusions[col] = pd.to_datetime(exclusions[col])
    output.mkdir(parents=True, exist_ok=False)
    pilot, pilot_audit = prepare(basalt,air,PILOT_START,PILOT_END,exclusions)
    pilot_grids, pilot_stats = save_period(output/'pilot',pilot,pilot_audit,PILOT_START,PILOT_END)
    # Broaden only if the pilot has no common retained 2h bin; this is an
    # unambiguous absence of data, not an arbitrary sufficiency threshold.
    fallback = not pilot_grids[2].common_retained.any()
    if fallback:
        views, audits = prepare(basalt,air,START,END,exclusions)
        grids, stats = save_period(output/'search_period',views,audits,START,END)
        candidates = candidate_windows(grids,views)
        candidates.to_csv(output/'candidate_windows_30d.csv',index=False)
    else:
        candidates = pd.DataFrame()
    manifest = dict(created_at_utc=utc_now(), stage='0.2 technical preparation only',
        source_commit='87b1827e6c78413f71e48365c3f18eeb98cbb31a',
        source_sha256={str(p.relative_to(ROOT)):h for p,h in SOURCE_SHA.items()},
        clock_documentation_checked=[
            'Sensors_Description/workflow_memory.md',
            'Project_description/sensorDB/ideal_vertical_period_workflow.md',
            'Project_description/sensorDB/air_co2_catalog.py',
            'Project_description/Research_log/data/ch02/archive/provenance.json',
            'Project_description/Research_log/data/stage0/air_1275/provenance.json'],
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),HERE/'chapter02.py',HERE/'stage02_coverage.py',EXCLUSIONS)},
        channels=CHANNELS, air_horizontal_offset_m=4, pilot=[str(PILOT_START),str(PILOT_END)],
        timezone_status='unconfirmed; calendar comparison provisional; no clock conversion',
        air_policy='exclude service <=-9999/nonfinite only; keep other values including >3000 with unresolved numeric-QC flag; no new range or spike cutoff',
        basalt_policy='chapter02.clean_admission_measurements; no monthly or H/A calls; four unique edge neighbours',
        count_policy='raw counts source records; retained counts work rows; doubtful is subset of retained; zero is an empty count, never a concentration',
        grid_policy='1h/2h left-closed right-open calendar bins from midnight; counts only, no aligned concentration values',
        fallback_used=fallback, candidate_windows=len(candidates),
        candidates_with_common_retained_bins=int(candidates.has_any_common_retained_bin.sum()) if fallback else None,
        spectral_admission='not established',
        files_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    for path, expected in SOURCE_SHA.items():
        if sha(path) != expected:
            raise AssertionError('Source modified during preparation')
    write_json(output/'manifest.json',manifest)
    print(pilot_stats.to_string(index=False))
    print(json.dumps({k:manifest[k] for k in ('fallback_used','candidate_windows','candidates_with_common_retained_bins')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'data/stage0/prepared_2025_08_14')
    run(parser.parse_args().output)

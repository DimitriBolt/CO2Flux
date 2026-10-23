"""Final stage 0.2 admission of four channels; offline, no Fourier calculation.

Reuses the approved pointwise/monthly numerical implementation in chapter02.
Original archive/decisions and historical stages 0.3/0.4 are read-only inputs.
Run from repository root: python3 Project_description/sensorDB/stage02_final_admission.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'Project_description/Research_log'))
from chapter02 import ADMISSION_PARAMETERS, clean_admission_measurements, monthly_admission
from preliminary_fourier_phases import window_slices

BASE_COMMIT = '484a4fd01ca1e12b0702026076a3004509d9c283'
ARCHIVE = ROOT/'Project_description/sensorDB/Row_data/LEO_West_full/20261002T061358Z-264980'
OUT = ROOT/'Project_description/Research_log/output/stage02_final_admission'
LOCAL = ROOT/'Project_description/Research_log/data/stage02_final_admission'
IDS = (1275, 996, 1012, 1035)
LEVELS = {1275:'C_air', 996:'C_5', 1012:'C_20', 1035:'C_50'}
START, END = pd.Timestamp('2025-04-29'), pd.Timestamp('2026-10-02')
EMPTY_EXCLUSIONS = pd.DataFrame(columns=['level','start','end'])


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def joint_months(monthly, sensor_ids=IDS):
    """Fail dominates uncertainty; only four positive channel decisions pass."""
    required={'month','sensorid','status','reason'}
    if not required.issubset(monthly.columns) or monthly.duplicated(['month','sensorid']).any():
        raise ValueError('Missing or duplicate sensor-month decisions')
    if monthly[list(required)].isna().any().any():
        raise ValueError('Undefined sensor-month metadata')
    if not monthly.status.isin(['pass','fail','undetermined']).all():
        raise ValueError('Unknown monthly admission status')
    rows=[]
    for month,group in monthly.groupby('month',sort=True):
        if set(group.sensorid)!=set(sensor_ids) or len(group)!=len(sensor_ids):
            raise ValueError(f'Incomplete four-channel monthly decisions: {month}')
        by=group.set_index('sensorid')
        statuses=by.status.tolist()
        status='fail' if 'fail' in statuses else ('undetermined' if 'undetermined' in statuses else 'pass')
        row=dict(month=str(month),status=status,reason='all_four_channels_pass' if status=='pass' else
                 ';'.join(f'{sid}:{by.loc[sid,"status"]}:{by.loc[sid,"reason"]}'
                          for sid in sensor_ids if by.loc[sid,'status']!='pass'))
        for sid in sensor_ids:
            row[f'status_{sid}']=by.loc[sid,'status'];row[f'reason_{sid}']=by.loc[sid,'reason']
        rows.append(row)
    return pd.DataFrame(rows)


def make_grid(works, start, end):
    """Actual occupancy only; no concentration resampling or Fourier coefficients.

    Exact duplicate times have already been combined in the work view. An
    occupied 2h bin has at least one finite retained observation of each channel.
    Zero counts are absence markers, never zero concentration observations.
    """
    index=pd.date_range(start,end,freq='2h',inclusive='left')
    result=pd.DataFrame(index=index);result.index.name='bin_start'
    for sid in IDS:
        frame=works[sid]
        frame=frame[frame.technically_retained & frame.localdatetime.ge(start) & frame.localdatetime.lt(end)]
        if not np.isfinite(frame.datavalue.to_numpy(float)).all():
            raise ValueError('Retained nonfinite observation')
        groups=frame.groupby(frame.localdatetime.dt.floor('2h'))
        result[f'n_{sid}']=groups.size().reindex(index,fill_value=0).astype(int)
        result[f'doubtful_{sid}']=groups.doubtful.any().reindex(index,fill_value=False).astype(bool)
        result[f'plateau_review_{sid}']=groups.plateau_review.any().reindex(index,fill_value=False).astype(bool)
    result['complete']=result[[f'n_{sid}' for sid in IDS]].gt(0).all(axis=1)
    return result


def classify_windows(grid, monthly, window_days=30, step_days=1):
    """Apply existing 0.4 calendar and >=2-bin computability, never compute F.

    Month gates refer to ALL common bins after cleaning, without selectively
    deleting non-pass months to rescue an otherwise non-admitted window.
    An unresolved excursion in an otherwise joint-pass window is explicitly
    pending the author's decision, not resolved by a new rejection threshold.
    """
    if not isinstance(grid.index,pd.DatetimeIndex) or grid.index.has_duplicates:
        raise ValueError('Unique chronological 2h grid required')
    if len(grid)>1 and not np.all(np.diff(grid.index.asi8)==pd.Timedelta(hours=2).value):
        raise ValueError('The original contiguous calendar grid must be retained')
    joint=joint_months(monthly).set_index('month')
    month_labels=grid.index.strftime('%Y-%m')
    if not set(month_labels).issubset(joint.index):
        raise ValueError('Missing monthly decisions for window calendar')
    rows=[]
    for k,(lo,hi) in enumerate(window_slices(len(grid),window_days,step_days)):
        part=grid.iloc[lo:hi];common=part.complete.astype(bool)
        used=part.loc[common];months=list(dict.fromkeys(used.index.strftime('%Y-%m')))
        calendar_months=list(dict.fromkeys(part.index.strftime('%Y-%m')))
        statuses=joint.loc[months,'status'].tolist() if months else []
        month_pass=np.asarray(part.index.strftime('%Y-%m').map(joint.status)=='pass')
        ig_bins=int(np.count_nonzero(common.to_numpy() & month_pass))
        flags=int(used[[f'doubtful_{sid}' for sid in IDS]].any(axis=1).sum())
        technical=len(used)>=2
        month_status='fail' if not months or 'fail' in statuses else ('undetermined' if 'undetermined' in statuses else 'pass')
        reasons=[];requires_decision=False
        if not technical:reasons.append('fewer_than_2_common_bins_for_existing_coefficient_precheck')
        if month_status!='pass':
            reasons.extend(f'{m}:{joint.loc[m,"status"]}:{joint.loc[m,"reason"]}'
                           for m in months if joint.loc[m,'status']!='pass')
            if not months:reasons.append('no_contributing_common_bins')
        if not technical or month_status=='fail':status='fail'
        elif month_status=='undetermined':status='undetermined'
        elif flags:
            status='undetermined';requires_decision=True
            reasons.append('unresolved_excursions_in_joint_pass_window_require_decision')
        else:
            status='pass';reasons.append('all_used_months_joint_pass_and_existing_computability_met')
        row=dict(window=k,start=part.index[0],end_exclusive=part.index[-1]+pd.Timedelta(hours=2),
                 possible_bins=len(part),common_bins=len(used),ig_common_bins=ig_bins,
                 technical_computable=technical,joint_month_status=month_status,
                 has_unresolved_excursions=bool(flags),unresolved_common_bins=flags,
                 status=status,admitted=status=='pass',reason=';'.join(reasons),requires_decision=requires_decision,
                 contributing_months=json.dumps(months),calendar_month_statuses=json.dumps(
                    {m:{str(sid):joint.loc[m,f'status_{sid}'] for sid in IDS} for m in calendar_months}),
                 calendar_month_reasons=json.dumps(
                    {m:{str(sid):joint.loc[m,f'reason_{sid}'] for sid in IDS} for m in calendar_months}))
        for sid in IDS:
            row[f'retained_observations_{sid}']=int(part[f'n_{sid}'].sum())
            row[f'unresolved_bins_{sid}']=int(used[f'doubtful_{sid}'].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def contiguous_runs(dates):
    dates=pd.DatetimeIndex(dates).normalize().unique().sort_values()
    if not len(dates):return []
    group=np.cumsum(np.r_[True,np.diff(dates.asi8)!=pd.Timedelta(days=1).value])
    result=[]
    for key in np.unique(group):
        part=dates[group==key]
        result.append(dict(start=str(part[0].date()),end_exclusive=str((part[-1]+pd.Timedelta(days=1)).date()),days=len(part)))
    return result


def read_sources():
    manifest=json.loads((ARCHIVE/'manifest.json').read_text())
    summary=json.loads((ARCHIVE/'diagnostics/summary.json').read_text())
    if manifest['status']!='complete':raise ValueError('Archive is not complete')
    expected={1275:(56,'umol/mol','LEO_WEST.DATAVALUESLICOR'),
              **{sid:(9,'ppm','LEO_WEST.DATAVALUES') for sid in IDS[1:]}}
    metadata={c['sensorid']:c for c in manifest['channels'] if c['sensorid'] in IDS}
    if set(metadata)!=set(IDS):raise ValueError('Wrong selected channels')
    hashes={str(p.relative_to(ROOT)):digest(p) for p in [ARCHIVE/'manifest.json',ARCHIVE/'diagnostics/summary.json']}
    # Verify immutable raw parts as recorded by the completed archive, without
    # rebuilding caches or any existing decisions.
    for block in manifest['blocks']:
        for part in block['parts']:
            path=ARCHIVE/part['file']
            actual=digest(path)
            if actual!=part['sha256']:raise ValueError(f'Raw archive checksum mismatch: {path}')
            hashes[str(path.relative_to(ROOT))]=actual
    frames={}
    for sid in IDS:
        c=metadata[sid]
        if (c['variableid'],c['units'],c['table'])!=expected[sid]:raise ValueError(f'Unexpected metadata: {sid}')
        for suffix in ('source_decisions','working'):
            rel=f'sensors/{sid}_{suffix}.parquet';path=ARCHIVE/'diagnostics'/rel
            actual=digest(path)
            if actual!=summary['files_sha256'][rel]:raise ValueError(f'Original decision checksum mismatch: {rel}')
            hashes[str(path.relative_to(ROOT))]=actual
        frame=pd.read_parquet(ARCHIVE/f'diagnostics/sensors/{sid}_source_decisions.parquet')
        if set(frame.sensorid)!={sid} or set(frame.variableid)!={c['variableid']}:
            raise ValueError('Source sensor/variable identity mismatch')
        required=['valueid','source_part','source_row_in_part','localdatetime']
        if frame[required].isna().any().any() or frame.duplicated(['source_part','source_row_in_part']).any():
            raise ValueError('Invalid or duplicate source-row provenance')
        frame['source_value']=frame.original_value if 'original_value' in frame else frame.datavalue
        frames[sid]=frame
    return frames,metadata,hashes


def summarize_channel(source, work, sid, period, start, end):
    source=source[source.localdatetime.ge(start)&source.localdatetime.lt(end)]
    work=work[work.localdatetime.ge(start)&work.localdatetime.lt(end)]
    counts=dict(sensorid=sid,period=period,raw_source_rows=len(source),unique_timestamps=len(work),
        exact_duplicate_extra=int(work.exact_duplicate_extra.sum()),
        retained_source_rows=int(source.technically_retained.sum()),
        excluded_source_rows=int((~source.technically_retained).sum()),
        retained_measurements=int(work.technically_retained.sum()),
        excluded_unique_times=int((~work.technically_retained).sum()),
        doubtful_measurements=int(work.doubtful.sum()),
        retained_in_pass_months=int((work.technically_retained&work.month_status.eq('pass')).sum()),
        retained_in_fail_months=int((work.technically_retained&work.month_status.eq('fail')).sum()),
        retained_in_undetermined_months=int((work.technically_retained&work.month_status.eq('undetermined')).sum()),
        retained_in_nonpass_months=int((work.technically_retained&work.month_status.ne('pass')).sum()),
        previous_retained_source_rows=int(source.previous_technically_retained.sum()),
        previous_doubtful_source_rows=int((source.previous_technically_retained&source.previous_doubtful).sum()))
    assert counts['raw_source_rows']==counts['retained_source_rows']+counts['excluded_source_rows']
    assert counts['unique_timestamps']==counts['retained_measurements']+counts['excluded_unique_times']
    assert counts['retained_measurements']==counts['retained_in_pass_months']+counts['retained_in_nonpass_months']
    return counts


def complete_passport(passport, monthly, out=OUT):
    """Attach precise decision locations and closure limits, without new analysis."""
    passport['pass_months_by_channel']={str(s):monthly.loc[monthly.sensorid.eq(s)&monthly.status.eq('pass'),'month'].tolist() for s in IDS}
    passport['undetermined_months_by_channel']={str(s):monthly.loc[monthly.sensorid.eq(s)&monthly.status.eq('undetermined'),'month'].tolist() for s in IDS}
    passport['decision_tables']={name:str((out/name).relative_to(ROOT)) for name in (
        'channel_summary.csv','measurement_reasons.csv','channel_months.csv','channel_days.csv',
        'joint_months.csv','joint_days.csv','joint_admitted_intervals.csv','window_admission_30d_1d.csv')}
    passport['measurement_decisions']={str(s):dict(
        unique_timestamps=str((LOCAL/f'{s}_measurements.parquet').relative_to(ROOT)),
        every_original_source_row=str((LOCAL/f'{s}_source_decisions.parquet').relative_to(ROOT)),
        retained_mask='technically_retained',excluded_mask='not technically_retained',
        individual_reason='exclusion_reason',remaining_doubt='doubt_reason',
        retained_in_positive_month='in_admitted_month') for s in IDS}
    passport['admission_levels']={
        'measurements':'Pointwise retained/excluded decisions; flagged excursions remain retained with specific reasons.',
        'months':'Independent I.G. monthly gate, including uncertainty bounds; undetermined never counts as pass.',
        'four_channel_windows':'All used common bins must come from four-pass months; no new percentage, gap, constant-run or amplitude cutoffs.'}
    passport['ready_for_stage02_closure_review']=passport['windows']['requires_author_decision']==0
    passport['sufficient_for_continuing_phase_study']=passport['windows']['pass_count']>0 and passport['ready_for_stage02_closure_review']
    passport['closure_authority']='Dimitri Bolt; stage 0.5 and the overall stage 0 are not automatically closed.'
    if passport['joint_month_status_counts'].get('pass',0)==0:
        passport['full_history_window_consequence']='No measurement-using window anywhere in the full history can pass the joint month gate: there is no joint-pass month. The 492 explicitly enumerated windows retain the original stage 0.4 calendar.'
    references=[ROOT/'Project_description/Papers/LEO_CO2_sensor_audit_2026-06-30.pdf',
                ROOT/'Project_description/Research_log/README.md',ROOT/'Project_description/Research_log/PLAN.md']
    references += [ROOT/'Project_description/Research_log/output'/directory/name
                   for directory in ('preliminary_spectra','preliminary_fourier_phases')
                   for name in ('passport.json','validation.json')]
    passport['reference_input_sha256']={str(p.relative_to(ROOT)):digest(p) for p in references}
    passport['implementation_sha256']={str(p.relative_to(ROOT)):digest(p) for p in
        [Path(__file__),ROOT/'Project_description/Research_log/chapter02.py']}
    return passport


def run(out=OUT, local=LOCAL):
    out.mkdir(parents=True,exist_ok=True);local.mkdir(parents=True,exist_ok=True)
    frames,metadata,input_hashes=read_sources()
    first=min(f.localdatetime.min() for f in frames.values()).normalize()
    end=max(f.localdatetime.max() for f in frames.values()).normalize()+pd.Timedelta(days=1)
    works={};monthly_parts=[];daily_parts=[]
    for sid in IDS:
        c=metadata[sid];frame=frames[sid]
        numerical=frame[['sensorid','variableid','localdatetime']].copy()
        numerical['datavalue']=frame.source_value.to_numpy(float)
        work=clean_admission_measurements(numerical,{LEVELS[sid]:sid},EMPTY_EXCLUSIONS,
              channel_metadata={LEVELS[sid]:{key:c[key] for key in ('variableid','units','table','sensorcode')}})
        work['technically_retained']=work.exclusion_reason.eq('')
        work['doubtful']=work.technically_retained&work.doubt_reason.ne('')
        work['plateau_review']=work.technically_retained&work.constant_run_hours.ge(1) # diagnostic only
        # The old three basalt cleanings must remain numerically identical.
        old=pd.read_parquet(ARCHIVE/f'diagnostics/sensors/{sid}_working.parquet')
        if sid!=1275:
            for column in ('localdatetime','datavalue','exclusion_reason','doubt_reason'):
                pd.testing.assert_series_equal(work[column].reset_index(drop=True),old[column].reset_index(drop=True),check_names=False)
        monthly,daily=monthly_admission(work,{LEVELS[sid]:sid},first,end)
        for key in ('variableid','units','table','sensorcode'):
            monthly[key]=c[key];daily[key]=c[key]
        month_status=monthly.set_index('month').status
        work['month']=work.localdatetime.dt.strftime('%Y-%m')
        work['month_status']=work.month.map(month_status)
        work['in_admitted_month']=work.technically_retained&work.month_status.eq('pass')
        work.to_parquet(local/f'{sid}_measurements.parquet',index=False)
        # Keep every original source row and previous decisions as provenance.
        source=frame.rename(columns={name:'previous_'+name for name in frame.columns
                     if name in {'datavalue','original_value','exclusion_reason','doubt_reason','technically_retained',
                                 'doubtful','excluded','status','plateau_review'}}).copy()
        fields=['localdatetime','datavalue','exclusion_reason','doubt_reason','technically_retained',
                'doubtful','constant_run_hours','month','month_status','in_admitted_month']
        source=source.merge(work[fields],on='localdatetime',how='left',validate='many_to_one')
        for key in ('units','table','sensorcode'):source[key]=c[key]
        assert len(source)==len(frame) and source.technically_retained.notna().all()
        source.to_parquet(local/f'{sid}_source_decisions.parquet',index=False)
        works[sid]=work;frames[sid]=source
        monthly['raw_source_rows']=monthly.month.map(source.groupby('month').size()).fillna(0).astype(int)
        monthly['retained_source_rows']=monthly.month.map(source.groupby('month').technically_retained.sum()).fillna(0).astype(int)
        monthly['retained_measurements']=monthly.month.map(work.groupby('month').technically_retained.sum()).fillna(0).astype(int)
        monthly['doubtful_measurements']=monthly.month.map(work.groupby('month').doubtful.sum()).fillna(0).astype(int)
        monthly_parts.append(monthly);daily_parts.append(daily)
        print(f'{sid}: cleaned {len(frame)} source rows -> {int(work.technically_retained.sum())} retained times; monthly {monthly.status.value_counts().to_dict()}',flush=True)
    monthly=pd.concat(monthly_parts,ignore_index=True);daily=pd.concat(daily_parts,ignore_index=True)
    joint=joint_months(monthly)
    monthly.to_csv(out/'channel_months.csv',index=False)
    daily.to_csv(out/'channel_days.csv',index=False)
    joint.to_csv(out/'joint_months.csv',index=False)
    summaries=[];reasons=[]
    for sid in IDS:
        for period,lo,hi in [('full_history',first,end),('stage04_calendar',START,END)]:
            summaries.append(summarize_channel(frames[sid],works[sid],sid,period,lo,hi))
            w=works[sid];w=w[w.localdatetime.ge(lo)&w.localdatetime.lt(hi)]
            s=frames[sid];s=s[s.localdatetime.ge(lo)&s.localdatetime.lt(hi)]
            for reason,n in w.loc[~w.technically_retained,'exclusion_reason'].value_counts().items():
                reasons.append(dict(sensorid=sid,period=period,kind='exclusion',reason=reason,unique_times=int(n),
                               source_rows=int(s.exclusion_reason.eq(reason).sum())))
            for reason,n in w.loc[w.doubtful,'doubt_reason'].value_counts().items():
                reasons.append(dict(sensorid=sid,period=period,kind='retained_doubt',reason=reason,unique_times=int(n),
                               source_rows=int((s.doubtful&s.doubt_reason.eq(reason)).sum())))
    pd.DataFrame(summaries).to_csv(out/'channel_summary.csv',index=False)
    pd.DataFrame(reasons).to_csv(out/'measurement_reasons.csv',index=False)
    grid=make_grid(works,first,end)
    grid['joint_month_status']=grid.index.strftime('%Y-%m').map(joint.set_index('month').status)
    grid['ig_complete']=grid.complete&grid.joint_month_status.eq('pass')
    grid.to_parquet(local/'joint_2h_grid.parquet')
    days=pd.DataFrame(index=pd.date_range(first,end,freq='D',inclusive='left'))
    for sid in IDS:
        w=works[sid];ret=w[w.technically_retained]
        days[f'n_{sid}']=ret.groupby(ret.localdatetime.dt.normalize()).size().reindex(days.index,fill_value=0)
    days['common_after_cleaning']=days[[f'n_{sid}' for sid in IDS]].gt(0).all(axis=1)
    days['joint_month_status']=days.index.strftime('%Y-%m').map(joint.set_index('month').status)
    days['common_ig_admitted']=days.common_after_cleaning&days.joint_month_status.eq('pass')
    days['common_2h_bins']=grid.complete.resample('D').sum().reindex(days.index,fill_value=0)
    days['ig_common_2h_bins']=grid.ig_complete.resample('D').sum().reindex(days.index,fill_value=0)
    days.to_csv(out/'joint_days.csv',index_label='date')
    windows=classify_windows(grid.loc[(grid.index>=START)&(grid.index<END)],monthly)
    windows.to_csv(out/'window_admission_30d_1d.csv',index=False)
    assert len(windows)==492
    common=[]
    for period,lo,hi in [('full_history',first,end),('stage04_calendar',START,END)]:
        part=days[(days.index>=lo)&(days.index<hi)]
        common.append(dict(period=period,common_days_after_cleaning=int(part.common_after_cleaning.sum()),
            common_bins_after_cleaning=int(part.common_2h_bins.sum()),
            common_days_in_joint_pass_months=int(part.common_ig_admitted.sum()),
            common_bins_in_joint_pass_months=int(part.ig_common_2h_bins.sum())))
    periods=contiguous_runs(days.index[days.common_ig_admitted])
    pd.DataFrame(periods,columns=['start','end_exclusive','days']).to_csv(out/'joint_admitted_intervals.csv',index=False)
    old_spectrum=json.loads((ROOT/'Project_description/Research_log/output/preliminary_spectra/passport.json').read_text())
    old_phase=json.loads((ROOT/'Project_description/Research_log/output/preliminary_fourier_phases/validation.json').read_text())
    assert old_spectrum['common_days']==520 and old_phase['common_bins']==6184 and old_phase['total_windows']==492
    local_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted(local.glob('*.parquet'))}
    conflicts=windows[windows.requires_decision].to_dict('records')
    for record in conflicts:
        record['start']=str(record['start']);record['end_exclusive']=str(record['end_exclusive'])
    passport=dict(stage='0.2',source_commit=BASE_COMMIT,decision_date_arizona='2026-10-22',
        archive=str(ARCHIVE.relative_to(ROOT)),metadata=[metadata[s] for s in IDS],
        numerical_parameters=ADMISSION_PARAMETERS,
        original_audit='GMM222 only; pages 3 and 5 of LEO_CO2_sensor_audit_2026-06-30.pdf',
        application_to_licor='Separate explicit Dimitri Bolt decision, 22 October 2026; identical numeric thresholds in umol/mol without concentration conversion',
        time_base='Database time confirmed by current PLAN.md; stored LOCALDATETIME without transformation',
        height_note='Air Excel Z=0.25 m vs Oracle BOXZ=0.05 m remains a metadata discrepancy; not resolved by this admission calculation',
        full_history_start=str(first),full_history_end_exclusive=str(end),
        monthly_calendar='Full available calendar months; no 29 April truncation before monthly gate. Months before a channel has data and partial edge months are reported explicitly.',
        monthly_status_counts={str(s):monthly.loc[monthly.sensorid.eq(s),'status'].value_counts().to_dict() for s in IDS},
        joint_month_status_counts=joint.status.value_counts().to_dict(),
        channel_summary=summaries,common_coverage=common,joint_admitted_intervals=periods,
        previous_stage03=dict(common_days=520,common_bins=6184),
        previous_stage04=dict(calendar_windows=492,preliminary_computed_phase_windows=492,scientific_admission='not established'),
        window_parameters=dict(window_days=30,step_days=1,f0_cycles_per_day=1),
        windows=dict(total=len(windows),technical_computable=int(windows.technical_computable.sum()),
             pass_count=int(windows.status.eq('pass').sum()),fail_count=int(windows.status.eq('fail').sum()),
             undetermined_count=int(windows.status.eq('undetermined').sum()),requires_author_decision=len(conflicts)),
        unresolved_window_cases=conflicts,
        exclusions='Individual reasons retained in per-timestamp and per-source-row local Parquets; no original masks changed',
        spectral_sufficiency=('Not established: author decision required for flagged joint-pass windows' if conflicts else
             ('No admitted 30d/1d windows; cannot continue with the historical candidate calendar under these rules' if not windows.admitted.any() else
              'Admitted windows listed; this is not phase reliability or permission to proceed beyond stage 0.5')),
        completion='requires_author_decision' if conflicts else 'admission_passport_complete_for_author_review',
        input_sha256=input_hashes,derived_local_sha256=local_hashes,
        implementation_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'Project_description/Research_log/chapter02.py']})
    passport=complete_passport(passport,monthly,out)
    save_json(out/'passport.json',passport)
    for name,expected in {**input_hashes,**passport['reference_input_sha256']}.items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Input changed during processing: {name}')
    print(json.dumps(dict(windows=passport['windows'],joint_months=passport['joint_month_status_counts'],common_coverage=common),indent=2),flush=True)
    return passport


if __name__=='__main__':
    run()

"""Read-only SensorDB audit for stage 0.1; no event detector or CO2 access.

Uses the existing SensorDB thick-mode connection. All source timestamps are
preserved. Output paths are confined to the stage01_irrigation snapshot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'Project_description/Research_log/data/stage01_irrigation'
START, END = datetime(2025, 5, 1), datetime(2026, 10, 1)

def channels():
    sensors = pd.read_csv(OUT/'sensors.csv')
    locations = pd.read_csv(OUT/'locations.csv')
    spatial = sensors.merge(locations, on='locationid', suffixes=('', '_loc'))
    # A complete local neighbourhood, not a signal/amplitude selection.
    near = spatial[(spatial.sensortypeid == 1) &
                   spatial.localx.between(-1, 3) & spatial.localy.between(2, 6)]
    pairs = [(993,v) for v in (13,14)]
    pairs += [(s,v) for s in range(1066,1076) for v in (15,17)]
    pairs += [(s,v) for s in [*range(1086,1098),1258] for v in (13,14)]
    pairs += [(int(s),6) for s in near.sensorid]
    pairs += [(s,7) for s in (668,889)]
    c = pd.DataFrame(pairs,columns=['sensorid','variableid'])
    c = c.merge(spatial, on='sensorid').merge(pd.read_csv(OUT/'variables.csv'),on='variableid')
    return c

def predicate(c):
    terms = []
    for v, g in c.groupby('variableid'):
        terms.append(f"(variableid={int(v)} AND sensorid IN ({','.join(str(int(s)) for s in sorted(g.sensorid))}))")
    return '(' + ' OR '.join(terms) + ')'

def load_raw():
    files = sorted((OUT/'raw').glob('measurements_*.parquet'))
    if len(files) != 17:
        raise RuntimeError(f'Incomplete snapshot: {len(files)}/17 months')
    return pd.concat([pd.read_parquet(f) for f in files],ignore_index=True).sort_values(['sensorid','variableid','localdatetime','valueid'])

def intervals(mask, index):
    """Consecutive True slots, with exclusive end; descriptive, no detection cutoff."""
    pos = np.flatnonzero(mask)
    if not len(pos):
        return []
    pieces = np.split(pos, np.flatnonzero(np.diff(pos) != 1)+1)
    return [(index[g[0]], index[g[-1]]+pd.Timedelta(minutes=15),len(g)) for g in pieces]

def analyze():
    (OUT/'.gitignore').write_text('raw/\nmassbalance.csv\nmeasurements_*.csv\npilot_series.parquet\n')
    d = load_raw()
    c = channels()
    d['slot'] = d.localdatetime.dt.floor('15min')  # coverage only, original time retained
    d['usable'] = np.isfinite(d.datavalue) & d.datavalue.ne(-9999)
    grid = pd.date_range(START,END,freq='15min',inclusive='left')
    groups = {k:g for k,g in d.groupby(['sensorid','variableid'],sort=False)}
    summary, monthly, gaps, cadence, duplicates, value_counts = [],[],[],[],[],[]
    for r in c.itertuples():
        s,v=int(r.sensorid),int(r.variableid)
        g=groups.get((s,v),d.iloc[:0])
        valid=g[g.usable]
        dt=g.localdatetime.diff().dt.total_seconds().dropna()
        const = (valid.datavalue.ne(valid.datavalue.shift()) |
                 valid.slot.diff().ne(pd.Timedelta(minutes=15))).cumsum()
        runs=valid.groupby(const).size()
        summary.append(dict(sensorid=s,variableid=v,sensorcode=r.sensorcode,source_table='LEO_WEST.DATAVALUES',
            rows=len(g),usable_rows=len(valid),service_rows=int(g.datavalue.eq(-9999).sum()),null_or_nonfinite=int((~np.isfinite(g.datavalue)).sum()),
            first_time=g.localdatetime.min(),last_time=g.localdatetime.max(),first_usable=valid.localdatetime.min(),last_usable=valid.localdatetime.max(),
            valid_min=valid.datavalue.min(),valid_max=valid.datavalue.max(),valid_unique=valid.datavalue.nunique(),
            longest_equal_value_run_rows=int(runs.max()) if len(runs) else 0,
            exact_duplicate_rows=int(g.duplicated('localdatetime',keep=False).sum()),
            duplicate_slot_rows=int(g.duplicated('slot',keep=False).sum()),
            nominal_step_seconds=900,modal_step_seconds=dt.mode().iloc[0] if len(dt) else np.nan,
            maximum_gap_seconds=dt.max(),expected_slots=len(grid),observed_slots=g.slot.nunique(),usable_slots=valid.slot.nunique(),
            localx=r.localx,localy=r.localy,depth_m=r.boxz))
        for a,b in zip(pd.date_range(START,END,freq='MS')[:-1],pd.date_range(START,END,freq='MS')[1:]):
            x=g[g.localdatetime.ge(a)&g.localdatetime.lt(b)];u=x[x.usable]
            expected=int((b-a).total_seconds()/900)
            monthly.append(dict(sensorid=s,variableid=v,month=a.strftime('%Y-%m'),rows=len(x),usable_rows=len(u),
                service_rows=int(x.datavalue.eq(-9999).sum()),observed_slots=x.slot.nunique(),usable_slots=u.slot.nunique(),
                expected_slots=expected,missing_observation_slots=expected-x.slot.nunique(),unusable_or_missing_slots=expected-u.slot.nunique(),
                first_time=x.localdatetime.min(),last_time=x.localdatetime.max(),first_usable=u.localdatetime.min(),last_usable=u.localdatetime.max()))
        for kind, values in [('no_observation',g.slot),('no_usable_value',valid.slot)]:
            for a,b,n in intervals(~grid.isin(values),grid):
                gaps.append(dict(sensorid=s,variableid=v,kind=kind,start=a,end_exclusive=b,slots=n,hours=n/4))
        for sec,n in dt.value_counts().items():
            cadence.append(dict(sensorid=s,variableid=v,step_seconds=sec,count=int(n)))
        for t,x in g[g.duplicated('localdatetime',keep=False)].groupby('localdatetime'):
            duplicates.append(dict(sensorid=s,variableid=v,time=t,rows=len(x),distinct_values=x.datavalue.nunique(),valueids=';'.join(map(str,x.valueid))))
        for val,n in g.datavalue.value_counts(dropna=False).head(5).items():
            value_counts.append(dict(sensorid=s,variableid=v,datavalue=val,rows=int(n)))
    for name,rows in [('channel_summary',summary),('monthly_coverage',monthly),('gaps',gaps),('cadence',cadence),('duplicate_times',duplicates),('common_values',value_counts)]:
        frame=pd.DataFrame(rows)
        if name=='duplicate_times' and not len(frame):
            frame=pd.DataFrame(columns=['sensorid','variableid','time','rows','distinct_values','valueids'])
        frame.to_csv(OUT/(name+'.csv'),index=False)
    # Output exact counter differences, never a sum of cumulative readings.
    resets=[]; counter_summary=[]; ratios=[]
    for s in [993,*range(1086,1098),1258]:
        x=groups[(s,13)]; y=groups[(s,14)]
        pair=x.merge(y,on='localdatetime',suffixes=('_cnt','_vol'))
        z=pair[pair.usable_cnt & pair.usable_vol]
        nonzero=z[z.datavalue_cnt.ne(0)]
        ratio=nonzero.datavalue_vol/nonzero.datavalue_cnt
        ratios.append(dict(sensorid=s,pairs=len(z),nonzero_pairs=len(nonzero),volume_per_count_min=ratio.min(),volume_per_count_max=ratio.max()))
        valid=y[y.usable].copy();valid['diff']=valid.datavalue.diff();valid['elapsed_seconds']=valid.localdatetime.diff().dt.total_seconds()
        counter_summary.append(dict(sensorid=s,rows=len(y),valid_rows=len(valid),positive_differences=int(valid['diff'].gt(0).sum()),
            zero_differences=int(valid['diff'].eq(0).sum()),negative_differences=int(valid['diff'].lt(0).sum()),
            maximum_positive_difference=valid['diff'].max(),minimum_difference=valid['diff'].min()))
        for i in np.flatnonzero(valid['diff'].lt(0)):
            a,b=valid.iloc[i-1],valid.iloc[i]
            resets.append(dict(sensorid=s,previous_time=a.localdatetime,time=b.localdatetime,previous_value=a.datavalue,value=b.datavalue,
                               difference=b['diff'],elapsed_seconds=b.elapsed_seconds,previous_valueid=int(a.valueid),valueid=int(b.valueid)))
    pd.DataFrame(ratios).to_csv(OUT/'counter_conversion_checks.csv',index=False)
    pd.DataFrame(counter_summary).to_csv(OUT/'counter_summary.csv',index=False)
    pd.DataFrame(resets).to_csv(OUT/'counter_decreases.csv',index=False)
    # Check the documented factory factors against the available stored kg series.
    cp=pd.read_csv(OUT/'calibrationparameters.csv')
    mass_checks=[]
    for s in range(1066,1076):
        pair=groups[(s,15)].merge(groups[(s,17)],on='localdatetime',suffixes=('_raw','_kg'))
        z=pair[pair.usable_raw & pair.usable_kg]
        slope,offset=np.polyfit(z.datavalue_raw,z.datavalue_kg,1)
        coef=cp[cp.sensorid.eq(s)].set_index('parametercode').parametervalue
        residual=z.datavalue_kg-(slope*z.datavalue_raw+offset)
        mass_checks.append(dict(sensorid=s,pairs=len(z),k=coef['k'],MaxLoad=coef['MaxLoad'],
            empirical_kg_per_raw_unit=slope,empirical_offset_kg=offset,max_abs_fit_residual_kg=abs(residual).max(),
            raw_min=z.datavalue_raw.min(),raw_max=z.datavalue_raw.max(),
            metadata_direct_factor_MaxLoad_over_k=coef['MaxLoad']/coef['k'],
            empirical_factor_divided_by_MaxLoad_over_k=slope/(coef['MaxLoad']/coef['k'])))
    pd.DataFrame(mass_checks).to_csv(OUT/'mass_conversion_checks.csv',index=False)
    # A source-provided candidate register. P>0 is a field selection, not a new
    # recognition threshold. It is NOT independently documented irrigation.
    mb=pd.read_csv(OUT/'massbalance.csv',parse_dates=['localdatetime'])
    h=mb[mb.note.eq('hr')].sort_values('localdatetime')
    h['balance_residual']=h.et_kg-(h.p_kg-h.q_kg-h.dm_kg)
    pd.DataFrame([dict(rows=len(g),note=n,first=g.localdatetime.min(),last=g.localdatetime.max(),
        mass_service_rows=int(g.mass.eq(-9999).sum()),p_service_rows=int(g.p_kg.eq(-9999).sum()),
        positive_p_rows=int(g.p_kg.gt(0).sum()),negative_q_rows=int((g.q_kg.lt(0)&g.q_kg.ne(-9999)).sum()),
        same_note_duplicate_times=int(g.localdatetime.duplicated(keep=False).sum())) for n,g in mb.groupby('note')]).to_csv(OUT/'massbalance_quality.csv',index=False)
    pos=h[h.p_kg.gt(0)].copy()
    blocks=(pos.localdatetime.diff().ne(pd.Timedelta(hours=1))).cumsum()
    candidates=[]; evidence=[]
    for k,g in pos.groupby(blocks):
        first,last=g.localdatetime.min(),g.localdatetime.max()
        before=first-pd.Timedelta(hours=1)
        event=dict(candidate_id=f'MB{k:02}',first_positive_p_record=first,last_positive_p_record=last,
            comparison_start=before,comparison_end=last,positive_hour_records=len(g),
            stored_p_sum_kg=g.p_kg.sum(),stored_p_sum_mm=g.p_mm.sum(),massbalance_ids=';'.join(map(str,g.mbid)),
            status='source_derived_candidate_not_a_confirmed_event',onset='not_established',end='not_established',
            delivered_volume='not_established',slope_part='not_documented',experiment_type='not_documented')
        for s,v in [(16,6),(19,6),(317,6),(668,7),(889,7)]:
            x=groups[(s,v)]
            a=x[x.slot.eq(before)&x.usable];b=x[x.slot.eq(last)&x.usable]
            row=dict(candidate_id=event['candidate_id'],sensorid=s,variableid=v)
            if len(a)==len(b)==1:
                aa,bb=a.iloc[0],b.iloc[0]
                row.update(before_time=aa.localdatetime,after_time=bb.localdatetime,before_value=aa.datavalue,after_value=bb.datavalue,
                    change=bb.datavalue-aa.datavalue,before_valueid=int(aa.valueid),after_valueid=int(bb.valueid))
            evidence.append(row)
        candidates.append(event)
    pd.DataFrame(candidates).to_csv(OUT/'massbalance_candidates.csv',index=False)
    pd.DataFrame(evidence).to_csv(OUT/'candidate_independent_evidence.csv',index=False)
    # Source-ready data for checking hydrologic agreement, not interpolation.
    mass=d[d.variableid.eq(17)].copy()
    pivot=mass.pivot_table(index='slot',columns='sensorid',values='datavalue',aggfunc='first')
    counts=mass.groupby('slot').size()
    pivot=pivot.reindex(grid)
    total=pivot.where(pivot.ne(-9999)).sum(axis=1,min_count=10)
    total[counts.reindex(grid,fill_value=0).ne(10)]=np.nan
    total.rename('stored_total_mass_kg').to_csv(OUT/'total_mass_15min.csv',index_label='slot_for_comparison')
    daily=[]
    for day in pd.date_range(START,END,freq='D',inclusive='left'):
        tomorrow=day+pd.Timedelta(days=1)
        z=total.loc[(total.index>=day)&(total.index<tomorrow)]
        row=dict(date=day.date(),mass_slots=z.notna().sum(),mass_first=z.dropna().iloc[0] if z.notna().any() else np.nan,
                 mass_last=z.dropna().iloc[-1] if z.notna().any() else np.nan)
        row['mass_change_between_available_endpoints']=row['mass_last']-row['mass_first']
        for s in [16,19,317,668,889]:
            x=groups[(s,6 if s<500 else 7)]
            x=x[x.usable & x.localdatetime.ge(day) & x.localdatetime.lt(tomorrow)]
            row[f's{s}_rows']=len(x)
            row[f's{s}_change']=x.datavalue.iloc[-1]-x.datavalue.iloc[0] if len(x) else np.nan
        daily.append(row)
    pd.DataFrame(daily).to_csv(OUT/'daily_independent_changes.csv',index=False)
    month_overview=[]
    for a,b in zip(pd.date_range(START,END,freq='MS')[:-1],pd.date_range(START,END,freq='MS')[1:]):
        row=dict(month=a.strftime('%Y-%m'),expected_15min_slots=int((b-a).total_seconds()/900),
                 complete_10cell_mass_slots=int(total.loc[(total.index>=a)&(total.index<b)].notna().sum()),
                 positive_P_candidate_blocks=sum(a<=e['first_positive_p_record']<b for e in candidates))
        for s,v in [(993,14),(16,6),(19,6),(317,6),(668,7),(889,7),(1258,14)]:
            x=groups[(s,v)]
            row[f's{s}_usable_slots']=x.loc[x.usable & x.localdatetime.ge(a) & x.localdatetime.lt(b),'slot'].nunique()
        month_overview.append(row)
    pd.DataFrame(month_overview).to_csv(OUT/'monthly_overview.csv',index=False)
    proposed=c[c.sensortypeid.isin([3,7,8,9])].copy()
    proposed=proposed[['sensorid','sensorcode','variableid','variablecode','variablename','variableunitsid','localx','localy','boxz']]
    units=pd.read_csv(OUT/'units.csv')[['unitsid','unitsabbreviation']]
    proposed=proposed.merge(units,left_on='variableunitsid',right_on='unitsid').drop(columns=['unitsid'])
    proposed['source_table']='LEO_WEST.DATAVALUES'
    proposed['selector']=proposed.apply(lambda r:f'sensorid={r.sensorid} AND variableid={r.variableid}',axis=1)
    proposed['time_column']='LOCALDATETIME';proposed['value_column']='DATAVALUE'
    proposed['availability_scope']='2025-05-01 inclusive to 2026-10-01 exclusive; not full-history bounds'
    proposed.to_csv(OUT/'workbook_additions_proposed.csv',index=False)
    # MASSBALANCE rows and raw 10-cell sums must be distinguished from rainfall.
    check=h.set_index('localdatetime')[['mass','balance_residual']].join(total.rename('raw_10cell_sum'))
    check=check[(check.mass.ne(-9999)) & check.raw_10cell_sum.notna()]
    identity=h.loc[h[['mass','dm_kg','q_kg','p_kg','et_kg']].ne(-9999).all(axis=1),'balance_residual']
    dump_json(OUT/'massbalance_consistency.json',dict(compared_hours=len(check),
        max_abs_mass_difference_kg=abs(check.mass-check.raw_10cell_sum).max(),
        median_abs_mass_difference_kg=abs(check.mass-check.raw_10cell_sum).median(),
        identity_compared_rows=len(identity),max_abs_ET_identity_residual_kg=abs(identity).max(),
        warning='ET=P-Q-DM is an algebraic identity, not independent confirmation of P'))
    wetting_dates(d,pd.DataFrame(daily),h,total)
    draw_overview(d,total,candidates)
    report_tables()
    print('analyzed',len(d),'rows',len(c),'channels',len(candidates),'source candidates',flush=True)

def wetting_dates(d,daily,h,total):
    """Register sign concordance, not a calibrated irrigation event detector.

    All daily changes are published, including the unselected days. The physical
    zero is the only comparison; no amplitude, duration or cadence-of-irrigation
    cutoff is introduced. Event edges and applied water remain unestablished.
    """
    days=daily[(daily.mass_change_between_available_endpoints>0) &
               (daily.s19_change>0) & (daily.s317_change>0)]
    records=[];source_evidence=[];outflow_evidence=[];examples=[]
    for i,r in enumerate(days.itertuples(),1):
        day=pd.Timestamp(r.date);end=day+pd.Timedelta(days=1);key=f'W{i:02}'
        b=h[h.localdatetime.ge(day)&h.localdatetime.lt(end)]
        record=dict(date_id=key,date=str(r.date),observation_start=day,observation_end_exclusive=end,
            event_start='',event_end='',event_duration_hours='',delivered_volume_l='',intensity_mm_h='',
            slope_part='local wetting supported near x=2,y=4 and total-slope mass gain; exact irrigation footprint unknown',
            experiment_type='not_documented',sensorids='1066-1075 (17); 19,317 (6); ancillary 16 (6),668,889 (7),993 (13,14),1086-1097,1258 (13,14)',
            method='positive observed daily change in stored 10-cell mass and both independent VWC channels 19 and 317; no magnitude cutoff',
            status='concordant_wetting_date_only_irrigation_inferred',
            limits='calendar-day localization only; neither event boundaries nor event count within day established; completeness unproved',
            mass_change_between_available_daily_endpoints_kg=r.mass_change_between_available_endpoints,
            vwc19_change_percentage_points=r.s19_change,vwc317_change_percentage_points=r.s317_change,
            source_positive_P_hours=int(b.p_kg.gt(0).sum()),source_mass_unusable_hours=int(b.mass.eq(-9999).sum()),
            massbalance_ids=';'.join(map(str,b.mbid)))
        records.append(record)
        for s,v in [(16,6),(19,6),(317,6),(668,7),(889,7)]+[(s,17) for s in range(1066,1076)]:
            g=d[d.sensorid.eq(s)&d.variableid.eq(v)&d.localdatetime.ge(day)&d.localdatetime.lt(end)&d.usable]
            evidence=dict(date_id=key,sensorid=s,variableid=v,usable_rows=len(g))
            if len(g):
                a,z=g.iloc[0],g.iloc[-1]
                evidence.update(first_time=a.localdatetime,last_time=z.localdatetime,first_value=a.datavalue,last_value=z.datavalue,
                    change=z.datavalue-a.datavalue,first_valueid=int(a.valueid),last_valueid=int(z.valueid))
            source_evidence.append(evidence)
        for s in [*range(1086,1098),1258]:
            g=d[d.sensorid.eq(s)&d.variableid.eq(14)&d.localdatetime.ge(day)&d.localdatetime.lt(end)&d.usable]
            a,z=g.iloc[0],g.iloc[-1]
            outflow_evidence.append(dict(date_id=key,sensorid=s,usable_rows=len(g),first_time=a.localdatetime,last_time=z.localdatetime,
                first_valueid=int(a.valueid),last_valueid=int(z.valueid),counter_change_l=z.datavalue-a.datavalue,
                negative_differences=int(g.datavalue.diff().lt(0).sum()),
                limitation='endpoint change of one cumulative channel; not automatically total drainage or irrigation input'))
        if not b.p_kg.gt(0).any():
            examples.append(b)
    pd.DataFrame(records).to_csv(OUT/'wetting_dates.csv',index=False)
    evidence_frame=pd.DataFrame(source_evidence)
    for col in ['first_valueid','last_valueid']:
        evidence_frame[col]=evidence_frame[col].astype('Int64')
    evidence_frame.to_csv(OUT/'wetting_date_source_evidence.csv',index=False)
    pd.DataFrame(outflow_evidence).to_csv(OUT/'wetting_date_outflow_evidence.csv',index=False)
    if examples:
        pd.concat(examples).to_csv(OUT/'massbalance_counterexamples.csv',index=False)
    mo=pd.read_csv(OUT/'monthly_overview.csv')
    dates=pd.to_datetime(pd.DataFrame(records).date)
    mo['concordant_wetting_dates']=[int(dates.dt.strftime('%Y-%m').eq(m).sum()) for m in mo.month]
    mo.to_csv(OUT/'monthly_overview.csv',index=False)

def draw_overview(d,total,candidates,*,window=None,output_stem='hydrology_overview'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    output=ROOT/'Project_description/Research_log/output/stage01_irrigation'
    output.mkdir(exist_ok=True)
    if window is not None:
        start,end=window
        d=d[d.localdatetime.between(start,end)]
        total=total.loc[start:end]
        candidates=[e for e in candidates if start<=e['first_positive_p_record']<=end]
    fig,axs=plt.subplots(4,1,figsize=(16,11),sharex=True,constrained_layout=True)
    axs[0].plot(total.index,total/1000,lw=.5,label='10 stored Mass channels (VariableID 17)')
    axs[0].set_ylabel('Stored mass, t');axs[0].legend(loc='upper right')
    for s in [16,19,317]:
        x=d[(d.sensorid==s)&(d.variableid==6)]
        axs[1].plot(x.localdatetime,x.datavalue.where(x.usable),lw=.55,label=str(s))
    axs[1].set_ylabel('VWC, %');axs[1].legend(loc='upper right',ncol=3)
    for s in [668,889]:
        x=d[(d.sensorid==s)&(d.variableid==7)]
        axs[2].plot(x.localdatetime,x.datavalue.where(x.usable),lw=.5,label=str(s))
    axs[2].set_ylabel('Water potential, kPa');axs[2].legend(loc='upper right',ncol=2)
    x=d[(d.sensorid==993)&(d.variableid==14)]
    axs[3].plot(x.localdatetime,x.datavalue.where(x.usable),lw=.4,color='grey')
    axs[3].set_ylabel('EX81 stored volume, L\nreported noise; cause unverified')
    for ax in axs:
        for e in candidates:
            ax.axvline(e['first_positive_p_record'],color='tab:red',
                       alpha=.3 if window is None else .7,lw=.6,clip_on=window is None,
                       zorder=2 if window is None else 4)
        ax.grid(alpha=.2)
    if window is None:
        axs[-1].xaxis.set_major_locator(mdates.MonthLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axs[-1].set_xlim(START,END)
        title='LEO West 2025-05-01 to 2026-09-30: independent records and source-derived candidates'
    else:
        axs[-1].xaxis.set_major_locator(mdates.DayLocator())
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%b'))
        axs[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))
        axs[-1].set_xlim(start,end)
        axs[-1].set_xlabel('Source LOCALDATETIME (unchanged); daily ticks, minor ticks every 6 hours')
        title=f'LEO West: {start:%d.%b.%Y %H:%M} to {end:%d.%b.%Y %H:%M} ({(end-start).total_seconds()/86400:g} days)'
        for ax in axs:
            for line in ax.lines:
                line.set_linewidth(1)
    axs[-1].tick_params(axis='x',rotation=45)
    fig.suptitle(title+'\nRed lines: positive P in MASSBALANCE, not a verified irrigation calendar')
    fig.savefig(output/(output_stem+'.png'),dpi=150)
    if window is not None:
        fig.savefig(output/(output_stem+'.svg'))
    plt.close(fig)

def draw_saved_interval(first_candidate_id='MB02'):
    """Zoom between adjacent saved overview markers without rerunning the audit."""
    candidate_path=OUT/'massbalance_candidates.csv'
    candidates=pd.read_csv(candidate_path,parse_dates=['first_positive_p_record'])
    candidates=candidates.sort_values('first_positive_p_record').reset_index(drop=True)
    matches=candidates.index[candidates.candidate_id.eq(first_candidate_id)]
    if len(matches)!=1 or matches[0]+1>=len(candidates):
        raise ValueError('Choose a saved candidate with a following marker')
    pair=candidates.iloc[matches[0]:matches[0]+2]
    start,end=pair.first_positive_p_record.tolist()
    months=pd.date_range(start.to_period('M').start_time,end.to_period('M').start_time,freq='MS')
    raw_paths=[OUT/'raw'/f'measurements_{month:%Y-%m}.parquet' for month in months]
    d=pd.concat([pd.read_parquet(path,filters=[('localdatetime','>=',start),
                                               ('localdatetime','<=',end)])
                 for path in raw_paths],ignore_index=True)
    d=d.sort_values(['sensorid','variableid','localdatetime','valueid'])
    # Exactly the existing overview mask; native timestamps/values are retained.
    d['usable']=np.isfinite(d.datavalue) & d.datavalue.ne(-9999)
    total_path=OUT/'total_mass_15min.csv'
    total=pd.read_csv(total_path,index_col='slot_for_comparison',parse_dates=True)['stored_total_mass_kg']
    stem=f'hydrology_interval_{start:%Y-%m-%d}_to_{end:%Y-%m-%d}'
    draw_overview(d,total,pair.to_dict('records'),window=(start,end),output_stem=stem)
    output=ROOT/'Project_description/Research_log/output/stage01_irrigation'
    dump_json(output/(stem+'.json'),dict(
        start_inclusive=start,end_inclusive=end,candidate_ids=pair.candidate_id.tolist(),
        duration_days=(end-start).total_seconds()/86400,
        marker_meaning='First positive P record of consecutive saved MASSBALANCE blocks; not established irrigation boundaries',
        display='Same four overview panels; native sensor timestamps/values and existing usability mask; saved total mass / 1000 in tonnes',
        processing='No new averaging, interpolation, smoothing, event detection or source queries',
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [candidate_path,total_path,*raw_paths]},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(output/(stem+'.png'))

def report_tables():
    target=ROOT/'Project_description/Research_log/details/00_irrigation_audit.md'
    body=target.read_text()
    def put(name,table):
        nonlocal body
        block=f'<!-- BEGIN {name} -->\n'+table+f'\n<!-- END {name} -->'
        pattern=rf'<!-- BEGIN {name} -->.*?<!-- END {name} -->|<!-- {name} -->'
        body,n=re.subn(pattern,lambda _:block,body,flags=re.S)
        if n!=1:raise ValueError(f'Missing or repeated report marker: {name}')
    s=pd.read_csv(OUT/'channel_summary.csv').set_index('sensorid')
    lines=['| SensorID / VariableID | Строк | Неслужебных | Первое неслужебное время | Последнее неслужебное время |',
           '|---|---:|---:|---|---|']
    for key in [18,172,393,16,19,317,668,889]:
        r=s.loc[key]
        lines.append(f'| {key}/{int(r.variableid)} | {int(r.rows)} | {int(r.usable_rows)} | {r.first_usable if pd.notna(r.first_usable) else "нет"} | {r.last_usable if pd.notna(r.last_usable) else "нет"} |')
    put('CHANNEL_TABLE','\n'.join(lines))
    w=pd.read_csv(OUT/'wetting_dates.csv')
    lines=['| Месяц | Даты согласованного увлажнения |','|---|---|']
    for month,g in w.groupby(w.date.str[:7],sort=True):
        lines.append('| '+month+' | '+', '.join(g.date.str[-2:])+' |')
    put('DATE_TABLE','\n'.join(lines))
    m=pd.read_csv(OUT/'monthly_overview.csv')
    lines=['| Месяц | Всего ячеек | Масса 10 | EX81 | VWC 16 | VWC 19 | VWC 317 | Каждый MPS-2 | Дат увлажнения |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in m.itertuples():
        lines.append(f'| {r.month} | {r.expected_15min_slots} | {r.complete_10cell_mass_slots} | {r.s993_usable_slots} | {r.s16_usable_slots} | {r.s19_usable_slots} | {r.s317_usable_slots} | {r.s668_usable_slots} | {r.concordant_wetting_dates} |')
    put('MONTH_TABLE','\n'.join(lines))
    target.write_text(body)

def validate():
    d=load_raw()
    s=pd.read_csv(OUT/'channel_summary.csv')
    mo=pd.read_csv(OUT/'monthly_coverage.csv')
    w=pd.read_csv(OUT/'wetting_dates.csv')
    checks={}
    checks['17_months']=d.localdatetime.dt.to_period('M').nunique()==17
    checks['timestamps_inside_authorized_period']=bool(d.localdatetime.ge(START).all() and d.localdatetime.lt(END).all())
    checks['only_hydrologic_variables']=set(d.variableid)=={6,7,13,14,15,17}
    checks['raw_valueids_unique']=d.valueid.nunique()==len(d)
    checks['100_channels_1700_month_rows']=len(s)==100 and len(mo)==1700
    checks['row_counts_reconcile']=int(s.rows.sum())==len(d)==int(mo.rows.sum())
    checks['service_counts_reconcile']=int(s.service_rows.sum())==int(d.datavalue.eq(-9999).sum())
    checks['MPS_negative_values_retained']=bool(s[s.variableid.eq(7)].usable_rows.gt(0).all())
    checks['no_fabricated_event_fields']=bool(w[['event_start','event_end','event_duration_hours','delivered_volume_l','intensity_mm_h']].isna().all().all())
    checks['all_37_dates_have_independent_sign_agreement']=bool((w.mass_change_between_available_daily_endpoints_kg.gt(0)&w.vwc19_change_percentage_points.gt(0)&w.vwc317_change_percentage_points.gt(0)).all())
    candidates=pd.read_csv(OUT/'massbalance_candidates.csv')
    checks['35_source_dates_are_in_37_wetting_dates']=set(candidates.first_positive_p_record.str[:10])<=set(w.date)
    raw_manifest=json.loads((OUT/'raw_manifest.json').read_text())
    checks['raw_sha256_unchanged']=all(hashlib.sha256((OUT/f).read_bytes()).hexdigest()==meta['sha256'] for f,meta in raw_manifest.items())
    query_records=[json.loads(line) for line in (OUT/'queries.jsonl').read_text().splitlines()]
    query_counts={r['name']:r.get('rows') for r in query_records if r['name'].startswith('measurements_') and 'rows' in r}
    checks['oracle_query_counts_match_parts']=all(meta['rows']==query_counts[Path(f).stem] for f,meta in raw_manifest.items())
    before=json.loads((OUT/'protected_before.json').read_text())
    changed=[f for f,sha in before.items() if not (ROOT/f).is_file() or hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=sha]
    allowed={'Project_description/Research_log/JOURNAL.md','Project_description/Research_log/README.md'}
    checks['protected_files_unchanged']=set(changed)<=allowed
    result=dict(checks=checks,passed=all(checks.values()),raw_rows=len(d),selected_channels=len(s),
        observed_channels=int(s.rows.gt(0).sum()),nonservice_channels=int(s.usable_rows.gt(0).sum()),
        wetting_dates=len(w),source_candidate_blocks=len(candidates),changed_preexisting_tracked_files=changed,
        checked_utc=datetime.now(timezone.utc).isoformat(),
        caution='Nonservice means finite and not -9999, not proven scientifically reliable')
    dump_json(OUT/'validation.json',result)
    provenance=dict(base_commit='5499cb9a4060e2e6480e0673ff047657119b4270',
        current_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
        origin_main_verified_by='git ls-remote origin refs/heads/main on 2026-10-23; same full SHA as base_commit',
        period_start=START,period_end_exclusive=END,time_correction='none',
        source_files={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['AGENTS.md','Project_description/Research_log/PLAN.md','Sensors_Description/LEO-West-Inventory.xlsx','Sensors_Description/variables_schema.xlsx','Sensors_Description/LEOSensorDBdescription.pdf','Sensors_Description/workflow_memory.md','Project_description/sensorDB/sensorDB.py']},
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        massbalance_source_sha256=hashlib.sha256((OUT/'massbalance.csv').read_bytes()).hexdigest(),
        source_snapshot='queries.jsonl and raw_manifest.json; raw/ and massbalance.csv excluded from Git',
        oracle_operations='SELECT only; date-indexed monthly DATAVALUES queries; no unbounded measurement-history scan',
        acquisition_isolation='sequential SELECTs, not a single Oracle SCN; row IDs and hashes freeze this local snapshot')
    dump_json(OUT/'provenance.json',provenance)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['passed']:
        raise RuntimeError('Validation failed')

def dump_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n')

class Audit:
    def __init__(self):
        from sensorDB import SensorDB
        OUT.mkdir(parents=True, exist_ok=True)
        self.db = SensorDB(call_timeout_ms=60000)
        self.log_path = OUT / 'queries.jsonl'

    def query(self, name, sql, binds=None):
        before = time.monotonic()
        entry = dict(name=name, sql=sql, binds=binds or {}, utc=datetime.now(timezone.utc).isoformat())
        try:
            with self.db.cursor() as cur:
                cur.execute(sql, binds or {})
                cols = [c[0].lower() for c in cur.description]
                rows = [[v.read() if hasattr(v, 'read') else v for v in row] for row in cur]
            df = pd.DataFrame(rows, columns=cols)
            df.to_csv(OUT / f'{name}.csv', index=False)
            entry['rows'] = len(df)
            print(name, len(df), flush=True)
            return df
        except Exception as exc:
            entry['error'] = str(exc)
            raise
        finally:
            entry['seconds'] = round(time.monotonic()-before, 3)
            with self.log_path.open('a') as f:
                f.write(json.dumps(entry, default=str) + '\n')

    def metadata(self):
        tables = self.query('tables', "SELECT table_name,num_rows,last_analyzed FROM all_tables WHERE owner='LEO_WEST' ORDER BY table_name")
        self.query('columns', "SELECT table_name,column_name,data_type,column_id FROM all_tab_columns WHERE owner='LEO_WEST' ORDER BY table_name,column_id")
        self.query('indexes', "SELECT i.table_name,i.index_name,i.uniqueness,c.column_name,c.column_position FROM all_indexes i JOIN all_ind_columns c ON i.owner=c.index_owner AND i.index_name=c.index_name WHERE i.owner='LEO_WEST' ORDER BY i.table_name,i.index_name,c.column_position")
        for table in ['SENSORS','VARIABLES','LOCATIONS','UNITS','METHODS','SENSORTYPES','CALIBRATIONS','CALIBRATIONPARAMETERS','OUTPUTVARIABLES','SENSORLOG','SERIESCATALOG']:
            if table in set(tables.table_name):
                self.query(table.lower(), f'SELECT * FROM LEO_WEST.{table}')
        dump_json(OUT/'snapshot.json', {'period_start_inclusive': START, 'period_end_exclusive': END, 'time_correction': 'none', 'queried_utc':datetime.now(timezone.utc), 'scope':'LEO West stage 0.1 hydrology; metadata first'})

    def context(self):
        for table in ['SENSOROUTPUT','DAQMETHODS','EXPERIMENTLOG','QUALITYCONTROL','DATATYPECV']:
            self.query(table.lower(), f'SELECT * FROM LEO_WEST.{table}')
        self.query('table_comments', "SELECT table_name,comments FROM all_tab_comments WHERE owner='LEO_WEST' AND comments IS NOT NULL")
        self.query('column_comments', "SELECT table_name,column_name,comments FROM all_col_comments WHERE owner='LEO_WEST' AND comments IS NOT NULL")
        self.query('views', "SELECT view_name,text FROM all_views WHERE owner='LEO_WEST'")
        self.query('source_objects', "SELECT name,type,count(*) lines FROM all_source WHERE owner='LEO_WEST' GROUP BY name,type ORDER BY name,type")
        self.query('catalog_hydrology', "SELECT sensorid,variableid,min(dateday) first_day,max(dateday) last_day,sum(valuecnt) catalog_rows FROM LEO_WEST.DATACATALOG WHERE variableid IN (6,7,13,14,15,17,18) GROUP BY sensorid,variableid")
        self.query('massbalance', 'SELECT * FROM LEO_WEST.MASSBALANCE WHERE localdatetime >= :a AND localdatetime < :b ORDER BY localdatetime', {'a':START,'b':END})
        for table in ['OUTFLOWS_NEW','OUTFLOWS_OLD']:
            self.query(table.lower()+'_bounds', f'SELECT sensorid,variableid,count(*) n,min(localdatetime) first_time,max(localdatetime) last_time FROM LEO_WEST.{table} GROUP BY sensorid,variableid')

    def measurements(self):
        c = channels()
        c.to_csv(OUT/'channels.csv',index=False)
        raw = OUT/'raw'
        raw.mkdir(exist_ok=True)
        (OUT/'.gitignore').write_text('raw/\nmassbalance.csv\nmeasurements_*.csv\npilot_series.parquet\n')
        filt = predicate(c)
        for a,b in zip(pd.date_range(START,END,freq='MS')[:-1],pd.date_range(START,END,freq='MS')[1:]):
            name = f'measurements_{a:%Y-%m}'
            target = raw/(name+'.parquet')
            if target.exists():
                print('resume',name,flush=True)
                continue
            df = self.query(name, 'SELECT /*+ INDEX(d DATAVALUES_INDX1) */ valueid,localdatetime,sensorid,variableid,datavalue FROM LEO_WEST.DATAVALUES d WHERE localdatetime >= :a AND localdatetime < :b AND '+filt+' ORDER BY localdatetime,valueid', {'a':a.to_pydatetime(),'b':b.to_pydatetime()})
            df.to_parquet(target,index=False)
            (OUT/(name+'.csv')).unlink()
        dump_json(OUT/'raw_manifest.json', {str(f.relative_to(OUT)):{'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'rows':len(pd.read_parquet(f))} for f in sorted(raw.glob('*.parquet'))})

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=['metadata','context','measurements','analyze','validate','pilot','interval'])
    p.add_argument('--first-candidate',default='MB02',help='Left overview marker for interval; right marker is the next saved candidate')
    args = p.parse_args()
    if args.step == 'interval':
        draw_saved_interval(args.first_candidate)
        return
    if args.step == 'pilot':
        from stage01_hydrology_pilot import build
        build()
        return
    if args.step in ['analyze','validate']:
        globals()[args.step]()
        return
    a = Audit()
    try:
        getattr(a, args.step)()
    finally:
        a.db.disconnect()

if __name__ == '__main__':
    main()

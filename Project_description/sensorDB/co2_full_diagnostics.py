"""Automatic offline diagnosis of a COMPLETE 53-channel refresh; no FFT/phases.

Raw Decimal parts are immutable. Float64 conversion is restricted to diagnostic
working copies using the existing pointwise cleaner. Calendar overlap is only
provisional until the two sources' time bases are documented.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'Research_log'))
from chapter02 import clean_admission_measurements, STUCK_INTERVALS
from stage02_prepare import prepare_air
from co2_refresh import atomic_json,digest,now
LOG=logging.getLogger('diagnostics')


def periods(calendar):
    month=calendar.index.month
    season=np.select([np.isin(month,[12,1,2]),np.isin(month,[3,4,5]),np.isin(month,[6,7,8])],['DJF','MAM','JJA'],default='SON')
    season_year=calendar.index.year+(month==12)
    return {'year':calendar.index.year.astype(str),'season':np.char.add(season_year.astype(str),np.char.add('-',season))}


def runs(times):
    dates=pd.DatetimeIndex(times).normalize().unique().sort_values()
    if len(dates)==0:return pd.DataFrame(columns=['start','end_exclusive','days'])
    groups=np.cumsum(np.r_[True,np.diff(dates.asi8)!=pd.Timedelta(days=1).value])
    frame=pd.DataFrame({'date':dates,'group':groups})
    result=frame.groupby('group').date.agg(start='min',end_exclusive='max',days='size').reset_index(drop=True)
    result['end_exclusive']+=pd.Timedelta(days=1)
    return result


def build_cache(version,manifest,out):
    cache=out/'channel_input';cache.mkdir(exist_ok=True)
    inventory={c['sensorid'] for c in manifest['channels']}
    marker=cache/'complete.json'
    if marker.exists():return cache
    for block in manifest['blocks']:
        for part in block['parts']:
            file=version/part['file']
            if digest(file)!=part['sha256']:raise ValueError('Raw part checksum changed')
            table=pq.read_table(file).to_pandas()
            table['source_row_in_part']=np.arange(len(table))
            table['source_part']=part['file']
            # All source columns remain in raw Parquet; this is a working copy only.
            table['datavalue']=table.datavalue.map(lambda v:float(v) if v is not None else np.nan)
            table['sensorid']=table.sensorid.astype('int64');table['variableid']=table.variableid.astype('int64')
            table['valueid']=table.valueid.map(str)
            for sid,frame in table.groupby('sensorid'):
                if sid not in inventory:raise ValueError('Unknown raw channel')
                folder=cache/str(sid);folder.mkdir(exist_ok=True)
                path=folder/(block['id']+'-'+file.name)
                frame.to_parquet(path,index=False)
    atomic_json(marker,dict(at=now(),working_numeric_type='float64; raw Decimal parts unchanged'))
    return cache


def clean_sensor(raw,c):
    if raw.empty:
        # A channel with no Oracle records still has an explicit zero-coverage result.
        work=raw.copy()
        work['original_value']=pd.Series(dtype=float)
        for key in ['exclusion_reason','doubt_reason']:work[key]=pd.Series(dtype=str)
        for key in ['technically_retained','doubtful','plateau_review']:work[key]=pd.Series(dtype=bool)
        return work
    if c['kind']=='LI-COR':
        work,_=prepare_air(raw,raw.localdatetime.min(),raw.localdatetime.max()+pd.Timedelta(seconds=1))
        work['plateau_review']=False
    else:
        sid=c['sensorid'];level=str(sid)
        exclusions=pd.DataFrame([dict(level=level,start=pd.Timestamp(a),end=pd.Timestamp(b)) for a,b in STUCK_INTERVALS]
                               if sid in (994,1010,1026) else [],columns=['level','start','end'])
        work=clean_admission_measurements(raw,{level:sid},exclusions)
        work['technically_retained']=work.exclusion_reason.eq('')
        work['doubtful']=work.doubt_reason.ne('') & work.technically_retained
        # Reuse the previous >=1h plateau DIAGNOSTIC; never an automatic rejection.
        work['plateau_review']=work.constant_run_hours.ge(1) & work.technically_retained
    return work


def sensor_diagnostics(raw,work,c,out,start,end):
    sid=c['sensorid'];days=pd.date_range(start,end,freq='D',inclusive='left')
    calendar=pd.DataFrame(index=days)
    retained=work.loc[work.technically_retained.astype(bool)]
    certain=work.loc[work.technically_retained.astype(bool) & ~work.doubtful.astype(bool) & ~work.plateau_review.astype(bool)]
    summary=dict(sensorid=sid,sensorcode=c['sensorcode'],kind=c['kind'],x=c['x'],y=c['y'],z=c['z'],units=c['units'],
                 raw_rows=len(raw),retained_rows=len(retained),excluded_times=int((~work.technically_retained.astype(bool)).sum()),
                 doubtful_rows=int(work.doubtful.sum()),plateau_review_rows=int(work.plateau_review.sum()),
                 above_3000_rows=int(raw.datavalue.gt(3000).sum()) if c['kind']=='GMM222' else None)
    grids={};gaps=[];distributions=[]
    for state,frame in [('raw',raw),('retained',retained),('without_review',certain)]:
        t=frame.localdatetime.dropna().sort_values().drop_duplicates()
        calendar[state+'_rows']=frame.groupby(frame.localdatetime.dt.normalize()).size().reindex(days,fill_value=0)
        summary[state+'_days']=len(t.dt.normalize().unique())
        summary[state+'_first']=t.min();summary[state+'_last']=t.max()
        delta=t.diff().dt.total_seconds();positive=delta[delta.gt(0)]
        typical=positive.median()
        summary[state+'_median_step_s']=typical;summary[state+'_max_gap_s']=positive.max()
        if state!='without_review':
            distributions.extend(dict(sensorid=sid,state=state,seconds=float(step),count=int(n)) for step,n in positive.value_counts().items())
            previous=t.shift();selected=delta.gt(1.5*typical)
            gaps.extend(dict(sensorid=sid,state=state,start=a,end=b,seconds=float(d),diagnostic_limit_s=1.5*typical)
                        for a,b,d in zip(previous[selected],t[selected],delta[selected]))
            runs(t).to_csv(out/f'{sid}_{state}_daily_runs.csv',index=False)
        grid=pd.date_range(start,end,freq='2h',inclusive='left')
        grids[state]=frame.groupby(frame.localdatetime.dt.floor('2h')).size().reindex(grid,fill_value=0)
    calendar.to_csv(out/f'{sid}_daily.csv',index_label='date')
    pd.DataFrame(grids).to_parquet(out/f'{sid}_2h_counts.parquet')
    work.to_parquet(out/f'{sid}_working.parquet',index=False)
    # Map decisions back onto each source row without dropping conflicts/duplicates.
    if c['kind']=='GMM222' and len(raw):
        audit=raw.merge(work[['localdatetime','exclusion_reason','doubt_reason','technically_retained','doubtful','plateau_review']],
                        on='localdatetime',how='left',validate='many_to_one')
    else:audit=work
    audit.to_parquet(out/f'{sid}_source_decisions.parquet',index=False)
    reasons=[]
    for column in ['exclusion_reason','doubt_reason']:
        reasons.extend(dict(sensorid=sid,kind=column,reason=reason,rows=int(n)) for reason,n in work[column].value_counts().items() if reason)
    plateaus=[]
    if len(work) and c['kind']=='GMM222':
        selected=work.constant_run_hours.ge(1)
        boundary=~work.original_value.eq(work.original_value.shift()) | ~selected | ~selected.shift(fill_value=False)
        step=work.localdatetime.diff().dt.total_seconds()
        boundary |= step.gt(1.5*step[step.gt(0)].median())
        for _,part in work.loc[selected].groupby(boundary.cumsum()[selected]):
            plateaus.append(dict(sensorid=sid,start=part.localdatetime.min(),end=part.localdatetime.max(),
                                 rows=len(part),value=part.original_value.iloc[0],retained_rows=int(part.technically_retained.sum()),
                                 diagnostic_only=True))
    pd.DataFrame(plateaus,columns=['sensorid','start','end','rows','value','retained_rows','diagnostic_only']).to_csv(out/f'{sid}_plateaus.csv',index=False)
    return summary,reasons,gaps,distributions,calendar,grids


def aggregate(calendar,identity):
    results=[]
    for period_kind,labels in periods(calendar).items():
        for label,part in calendar.groupby(labels):
            row=dict(**identity,period_kind=period_kind,period=label,calendar_days_in_snapshot=len(part))
            for name in part.columns:
                row[name]=int(part[name].sum())
                if name.endswith('_rows'):row[name.replace('_rows','_days')]=int(part[name].gt(0).sum())
            results.append(row)
    return results


def diagnose(version):
    manifest=json.loads((version/'manifest.json').read_text())
    if not manifest.get('download_complete_at') or any(b['status']!='complete' for b in manifest['blocks']):
        raise ValueError('Full download is not complete')
    out=version/'diagnostics';out.mkdir(exist_ok=True)
    schemas=manifest['channels']
    roots=[b for b in manifest['blocks'] if not b.get('parent')]
    start=pd.Timestamp(min(b['start'] for b in roots)).normalize()
    end=pd.Timestamp(max(b['end'] for b in roots)).ceil('D')
    cache=build_cache(version,manifest,out)
    sensors=out/'sensors';sensors.mkdir(exist_ok=True)
    summaries=[];reasons=[];gaps=[];steps=[];yearly=[];calendars={};grids={}
    for c in schemas:
        sid=c['sensorid'];files=sorted((cache/str(sid)).glob('*.parquet'))
        raw=pd.concat([pd.read_parquet(p) for p in files],ignore_index=True) if files else pd.DataFrame(
            {'localdatetime':pd.Series(dtype='datetime64[ns]'),'datavalue':pd.Series(dtype=float),'sensorid':pd.Series(dtype=int),'variableid':pd.Series(dtype=int)})
        raw=raw.sort_values('localdatetime',kind='stable')
        work=clean_sensor(raw,c)
        s,r,g,d,calendar,grid=sensor_diagnostics(raw,work,c,sensors,start,end)
        summaries.append(s);reasons+=r;gaps+=g;steps+=d;yearly+=aggregate(calendar,{'sensorid':sid})
        calendars[sid]=calendar;grids[sid]=grid
        print('diagnosed',sid,'raw',len(raw),'retained',s['retained_rows'],flush=True)
    for name,rows in [('channels',summaries),('reasons',reasons),('gaps',gaps),('positive_intervals',steps),('channel_years_seasons',yearly)]:
        pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    basalt=[c for c in schemas if c['kind']=='GMM222'];air=[c for c in schemas if c['kind']=='LI-COR']
    candidates=[];distances=[];common_periods=[]
    for x,y in sorted({(c['x'],c['y']) for c in basalt}):
        triple=sorted([c for c in basalt if (c['x'],c['y'])==(x,y)],key=lambda c:-c['z'])
        if len(triple)!=3:raise ValueError('Vertical is not a triplet')
        measured=[((a['x']-x)**2+(a['y']-y)**2,a) for a in air]
        minimum=min(d for d,a in measured)
        vertical=f'W_x{x:g}_y{y:g}'
        for distance,a in measured:
            distances.append(dict(vertical=vertical,air_sensorid=a['sensorid'],distance_m=distance**.5,nearest=distance==minimum))
        for distance,a in measured:
            if distance!=minimum:continue  # All equal nearest candidates are retained.
            ids=[c['sensorid'] for c in triple]+[a['sensorid']]
            row=dict(vertical=vertical,x=x,y=y,basalt_sensorids=','.join(str(c['sensorid']) for c in triple),
                     depths_cm=','.join(str(round(-100*c['z'])) for c in triple),air_sensorid=a['sensorid'],
                     air_distance_m=distance**.5,time_status='provisional',spectral_admission='not established')
            common=pd.DataFrame(index=calendars[ids[0]].index)
            for state in ['raw','retained','without_review']:
                daily=pd.concat([calendars[sid][state+'_rows'] for sid in ids],axis=1)
                occupied=daily.gt(0).all(axis=1)
                common['common_'+state+'_days']=occupied.astype(int)
                common['common_'+state+'_2h_bins']=pd.concat([grids[sid][state] for sid in ids],axis=1).gt(0).all(axis=1).resample('D').sum()
                row['common_'+state+'_days']=int(occupied.sum())
                row['common_'+state+'_2h_bins']=int(common['common_'+state+'_2h_bins'].sum())
                row[state+'_years_with_any_common_day']=len(set(common.index[occupied].year))
                row[state+'_first_day']=common.index[occupied].min();row[state+'_last_day']=common.index[occupied].max()
                runs(common.index[occupied]).to_csv(out/f'{vertical}_air{a["sensorid"]}_{state}_common_daily_runs.csv',index=False)
            common.to_csv(out/f'{vertical}_air{a["sensorid"]}_common_daily.csv',index_label='date')
            common_periods+=aggregate(common,{'vertical':vertical,'air_sensorid':a['sensorid']})
            candidates.append(row)
    candidates=pd.DataFrame(candidates).sort_values(['common_retained_2h_bins','common_retained_days'],ascending=False,kind='stable')
    candidates.to_csv(out/'vertical_candidates.csv',index=False)
    pd.DataFrame(distances).to_csv(out/'air_distances.csv',index=False)
    pd.DataFrame(common_periods).to_csv(out/'vertical_years_seasons.csv',index=False)
    result=dict(created_at=now(),source_version=version.name,channels=53,verticals=16,
                timezone='unconfirmed',air_quality='LI-COR numeric QC unresolved; retained with review flag',
                plateau_policy='>=1h diagnostic from existing code, not automatic rejection',
                day_policy='at least one actual measurement per channel; not continuous or simultaneous coverage',
                seasonal_policy='DJF/MAM/JJA/SON, December assigned to following winter year; only snapshot calendar days in denominators',
                no_automatic_multiyear_admission=True,
                files_sha256={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and 'channel_input' not in p.parts})
    atomic_json(out/'summary.json',result)
    update_document(version,candidates,summaries)


def update_document(version,candidates,summaries):
    path=ROOT/'Project_description/Research_log/details/02_data_checks.md'
    begin='<!-- CO2-FULL-REFRESH-RESULT-BEGIN -->';end='<!-- CO2-FULL-REFRESH-RESULT-END -->'
    lines=[begin,'','#### Результат автоматической диагностики полной выгрузки',
        '',f'Версия `{version.name}`; выполнено {now()}. Проверены все 53 канала и 16 вертикалей.',
        'Исходные данные сохранены отдельно; прежние архивы не использованы. FFT и фазы не вычислялись.',
        'Календарное сопоставление предварительное: часовые пояса/часы не подтверждены.',
        'Все сохранённые LI-COR имеют флаг незавершённого численного контроля качества.',
        'Дни ниже означают наличие хотя бы одной реальной записи каждого канала; это не полный годовой цикл и не научный допуск.',
        '', '| Вертикаль | Глубины, см | LI-COR / расстояние, м | Общих исходных дней | После технических масок | Общих интервалов 2 ч после масок |',
        '|---|---|---|---:|---:|---:|']
    for r in candidates.itertuples():
        lines.append(f'| {r.vertical} | {r.depths_cm} | {r.air_sensorid} / {r.air_distance_m:.3f} | {r.common_raw_days} | {r.common_retained_days} | {r.common_retained_2h_bins} |')
    relative=version.relative_to(ROOT)
    lines += ['',f'Полные результаты: `{relative}/diagnostics/` — `channels.csv`, `reasons.csv`,',
        '`gaps.csv`, `channel_years_seasons.csv`, `vertical_years_seasons.csv`, `vertical_candidates.csv`,',
        '`air_distances.csv`, реальные календарные интервалы и рабочие журналы каждого датчика.',
        'Контрольные суммы результатов — `diagnostics/summary.json`; исходных блоков — `manifest.json`.',
        'Окончательная вертикаль не выбрана; необходимы решение Dimitri Bolt, разрешение сомнений LI-COR,',
        'подтверждение времени и отдельный спектральный допуск. Этап 0.3 не начат.', '',end]
    text=path.read_text();section='\n'.join(lines)+'\n'
    if begin in text:
        before,rest=text.split(begin,1);_,after=rest.split(end,1);text=before+section+after.lstrip('\n')
    else:text+='\n'+section
    temporary=path.with_suffix('.md.tmp');temporary.write_text(text);temporary.replace(path)


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    diagnose(Path(sys.argv[1]).resolve())

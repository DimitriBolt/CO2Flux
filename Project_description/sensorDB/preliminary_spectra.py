"""Read-only stage 0.2 demonstration: floating-mean GLS, never phase/admission.

Run from repository root: python3 Project_description/sensorDB/preliminary_spectra.py
Inputs are the immutable full-refresh diagnostic copies. No database code is used.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'Project_description/sensorDB/Row_data/LEO_West_full/20261002T061358Z-264980'
OUT = ROOT / 'Project_description/Research_log/output/preliminary_spectra'
IDS = (1275, 996, 1012, 1035)
LABELS = ('Air · LI-COR 1275', '5 cm · GMM222 996', '20 cm · GMM222 1012', '50 cm · GMM222 1035')
COLORS = ('#006d77', '#0072b2', '#b45f06', '#9b287b')
WARNING = 'PRELIMINARY — QC AND TIME BASE UNRESOLVED'
START, END = pd.Timestamp('2025-04-29'), pd.Timestamp('2026-10-02')
FMAX = 6.0  # half the 12/day slow modal air cadence; not an irregular Nyquist theorem


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def gls(t, y, frequency):
    """Unweighted floating-mean fit: y=c+a*cos(2pi*f*t)+b*sin(2pi*f*t).

    A=hypot(a,b) is sinusoidal semi-amplitude in input units, not power.
    Centered normal equations eliminate c; no resampling, detrending or taper.
    W=abs(mean(exp(2pi*i*f*t)))**2 is the dimensionless sampling window.
    Singular fits / fewer than four distinct times return NaN, never zero.
    """
    t, y, frequency = (np.asarray(v, dtype=float) for v in (t, y, frequency))
    if t.shape != y.shape or t.ndim != 1 or not np.all(np.isfinite(t) & np.isfinite(y)):
        raise ValueError('GLS requires matching finite one-dimensional observations')
    if np.any(~np.isfinite(frequency) | (frequency <= 0)):
        raise ValueError('Frequencies must be finite and positive')
    amp, window = np.full(len(frequency), np.nan), np.full(len(frequency), np.nan)
    if len(np.unique(t)) < 4:
        return amp, window
    t = t - t.min()  # improves precision; amplitude is invariant to translation
    centered_y = y - y.mean()
    for first in range(0, len(frequency), 24):
        f = frequency[first:first + 24]
        theta = 2 * np.pi * f[:, None] * t
        c, s = np.cos(theta), np.sin(theta)
        mc, ms = c.mean(axis=1), s.mean(axis=1)
        window[first:first + len(f)] = mc * mc + ms * ms
        c -= mc[:, None]
        s -= ms[:, None]
        cc = np.einsum('ij,ij->i', c, c)
        ss = np.einsum('ij,ij->i', s, s)
        cs = np.einsum('ij,ij->i', c, s)
        yc = np.einsum('ij,j->i', c, centered_y)
        ys = np.einsum('ij,j->i', s, centered_y)
        det = cc * ss - cs * cs
        good = det > 1e-12 * (cc + ss)**2  # numerical rank guard only
        a = np.divide(yc * ss - ys * cs, det, out=np.full(len(f), np.nan), where=good)
        b = np.divide(ys * cc - yc * cs, det, out=np.full(len(f), np.nan), where=good)
        amp[first:first + len(f)] = np.hypot(a, b)
    return amp, window


def grid(days):
    if not np.isfinite(days) or days <= 0:
        raise ValueError('Positive duration required')
    # Exclude exactly 6/day: 2-hour regular segments cannot determine both
    # sine and cosine there, and clock jitter can make a spurious huge fit.
    return np.arange(1, int(np.ceil(FMAX * days))) / days


def window_starts(total_days, window_days=30, step_days=1):
    if not (1 <= window_days <= total_days and step_days >= 1):
        raise ValueError('Require 1 <= window_days <= total_days and step_days >= 1')
    return np.arange(0, total_days - window_days + 1e-9, step_days)


def coverage(t, lo, hi):
    """Calendar occupancy, not regularized measurements; all gaps remain real."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return dict(n=0, days=0, bins_2h=0, median_step_s=None, modal_step_s=None,
                    max_gap_h=None, edge_start_h=(hi-lo)*24, edge_end_h=(hi-lo)*24,
                    observed_span_days=0, insufficient='no retained observations')
    delta = np.diff(t) * 86400
    step, n = np.unique(np.rint(delta).astype(int), return_counts=True)
    return dict(n=len(t), days=len(np.unique(np.floor(t))), bins_2h=len(np.unique(np.floor(t*12))),
                median_step_s=float(np.median(delta)) if len(delta) else None,
                modal_step_s=int(step[np.argmax(n)]) if len(delta) else None,
                max_gap_h=float(delta.max()/3600) if len(delta) else None,
                edge_start_h=float((t[0]-lo)*24), edge_end_h=float((hi-t[-1])*24),
                observed_span_days=float(t[-1]-t[0]),
                insufficient='fewer than four distinct timestamps' if len(np.unique(t)) < 4 else '')


def read_inputs():
    manifest = json.loads((ARCHIVE/'manifest.json').read_text())
    summary = json.loads((ARCHIVE/'diagnostics/summary.json').read_text())
    if manifest['status'] != 'complete' or summary['channels'] != 53 or summary['verticals'] != 16:
        raise ValueError('Unexpected source archive')
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ARCHIVE/'manifest.json', ARCHIVE/'diagnostics/summary.json')}
    for name in ('channels.csv','vertical_candidates.csv'):
        path=ARCHIVE/'diagnostics'/name
        checksum=digest(path)
        if checksum != summary['files_sha256'][name]:
            raise ValueError(f'Input checksum mismatch: {name}')
        hashes[str(path.relative_to(ROOT))]=checksum
    inventory=pd.read_csv(ARCHIVE/'diagnostics/channels.csv')
    candidates=pd.read_csv(ARCHIVE/'diagnostics/vertical_candidates.csv')
    assert len(inventory)==53 and inventory.raw_rows.sum()==41946851
    assert len(candidates)==16 and int(candidates.common_retained_days.gt(0).sum())==2
    work, raw = {}, {}
    for sid in IDS:
        for suffix, destination in [('working', work), ('source_decisions', raw)]:
            rel = f'sensors/{sid}_{suffix}.parquet'
            p = ARCHIVE/'diagnostics'/rel
            checksum = digest(p)
            if checksum != summary['files_sha256'][rel]:
                raise ValueError(f'Input checksum mismatch: {rel}')
            hashes[str(p.relative_to(ROOT))] = checksum
            destination[sid] = pd.read_parquet(p).sort_values('localdatetime')
    metadata = {c['sensorid']: c for c in manifest['channels'] if c['sensorid'] in IDS}
    return work, raw, metadata, hashes


def days_since(t):
    return ((t - START).dt.total_seconds()/86400).to_numpy()


def prepare(out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    work, raw, metadata, hashes = read_inputs()
    series, passport, monthly, gaps = [], [], [], []
    total = (END-START).days
    calendar = pd.date_range(START, END, freq='D', inclusive='left')
    common = np.ones(total, bool)
    for sid in IDS:
        w = work[sid]
        w = w[w.localdatetime.ge(START) & w.localdatetime.lt(END)]
        r = raw[sid]
        r = r[r.localdatetime.ge(START) & r.localdatetime.lt(END)]
        kept = w[w.technically_retained]
        t = days_since(kept.localdatetime)
        y = kept.datavalue.to_numpy(float)
        q = (kept.doubtful | kept.plateau_review).to_numpy(bool)
        series.append(dict(sid=sid, t=t, y=y, q=q, units=metadata[sid]['units']))
        p = dict(label=LABELS[IDS.index(sid)], **metadata[sid])
        p.update(raw_rows=len(r), unique_times=len(w), retained_rows=len(kept),
                 excluded_times=int((~w.technically_retained).sum()), doubtful_rows=int(kept.doubtful.sum()),
                 plateau_review_rows=int(kept.plateau_review.sum()), first=str(kept.localdatetime.min()),
                 last=str(kept.localdatetime.max()), **coverage(t, 0, total))
        p['exclusion_reasons']=w.loc[~w.technically_retained,'exclusion_reason'].value_counts().to_dict()
        p['doubt_reasons']=kept.loc[kept.doubtful,'doubt_reason'].value_counts().to_dict()
        passport.append(p)
        present = np.isin(calendar, kept.localdatetime.dt.normalize().unique())
        common &= present
        for month in pd.period_range(START, END-pd.Timedelta(seconds=1), freq='M'):
            lo, hi = max(START, month.start_time), min(END, (month+1).start_time)
            mask = kept.localdatetime.ge(lo) & kept.localdatetime.lt(hi)
            a = kept[mask]
            wm=w[w.localdatetime.ge(lo)&w.localdatetime.lt(hi)]
            rm=r[r.localdatetime.ge(lo)&r.localdatetime.lt(hi)]
            monthly.append(dict(sensorid=sid, month=str(month), calendar_days=(hi-lo).days,
                raw_rows=len(rm),excluded_unique_times=int((~wm.technically_retained).sum()),
                retained_rows=len(a), doubtful_rows=int(a.doubtful.sum()),
                occupied_days=a.localdatetime.dt.normalize().nunique(),
                occupied_2h_bins=a.localdatetime.dt.floor('2h').nunique(), available_2h_bins=(hi-lo).days*12))
        delta = np.diff(t)*86400
        # Match the existing archive's descriptive 1.5 x median gap diagnostic.
        limit = 1.5*np.median(delta)
        for i in np.flatnonzero(delta > limit):
            gaps.append(dict(sensorid=sid,start=str(START+pd.Timedelta(days=t[i])),
                end=str(START+pd.Timedelta(days=t[i+1])),seconds=float(delta[i]),diagnostic_limit_s=float(limit)))
    lo, hi = max(s['t'][0] for s in series), min(s['t'][-1] for s in series)
    frequencies = grid(hi-lo)
    rows, static = [], []
    for s, p in zip(series, passport):
        mask = (s['t'] >= lo) & (s['t'] <= hi)
        amp, sampling = gls(s['t'][mask], s['y'][mask], frequencies)
        p['static_n'] = int(mask.sum())
        p['static_doubtful'] = int(s['q'][mask].sum())
        static.append(dict(f=frequencies, amplitude=amp, sampling=sampling))
        rows.extend(dict(sensorid=s['sid'],frequency_per_day=float(f),amplitude=float(a),sampling_window=float(w))
                    for f,a,w in zip(frequencies,amp,sampling))
    changes = np.diff(np.r_[False, common, False].astype(int))
    runs = [dict(start=str(calendar[a].date()),end_exclusive=str((calendar[b-1]+pd.Timedelta(days=1)).date()),days=int(b-a))
            for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1))]
    report = dict(input_version=ARCHIVE.name,checked_start_commit='48cc2c96bc47069f703136aca8c8f0acb5aedbef',
        hashes=hashes,calendar_start=str(START),calendar_end_exclusive=str(END),calendar_days=total,
        common_days=int(common.sum()),common_daily_runs=runs,
        common_start=str(START+pd.Timedelta(days=lo)),common_last=str(START+pd.Timedelta(days=hi)),
        span_days=hi-lo,frequency_step=1/(hi-lo),frequency_max=FMAX,
        normalization='equal weights; floating mean; A=sqrt(a*a+b*b), sinusoidal semi-amplitude in source units',
        experimental_dataset=dict(channels=53,original_measurements=41946851,verticals=16,technical_candidates=2),
        warning=WARNING,channels=passport)
    assert report['common_days'] == 520
    pd.DataFrame(monthly).to_csv(out/'monthly_coverage.csv',index=False)
    pd.DataFrame(gaps).to_csv(out/'gaps.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'static_spectra.csv',index=False)
    (out/'passport.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    return series, static, report, work, raw


def plot_static(static, report, out=OUT):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig, axs = plt.subplots(4,3,figsize=(17,11),sharex='col',gridspec_kw={'width_ratios':[1.9,1,1]})
    ranges = [(0,6),(0,.18),(.85,1.15)]
    for i,(sp,p) in enumerate(zip(static,report['channels'])):
        for j,(low,high) in enumerate(ranges):
            ax = axs[i,j]
            mask = (sp['f']>=low)&(sp['f']<=high)
            ax.plot(sp['f'][mask],sp['amplitude'][mask],color=COLORS[i],lw=.85)
            ax.set_xlim(low,high); ax.set_ylim(bottom=0); ax.grid(alpha=.22)
            if low <= 1 <= high:
                # The daily reference stays on the axis and above the plot;
                # it must not cover the spectral curve or affect data limits.
                ax.plot(1,0,marker='v',markersize=4,color='#333333',linestyle='none',
                        transform=ax.get_xaxis_transform(),clip_on=False,scalex=False,scaley=False)
                ax.annotate('24 h',xy=(1,1.014),xycoords=ax.get_xaxis_transform(),
                            xytext=(1+.05*(high-low),1.055),textcoords=ax.get_xaxis_transform(),
                            ha='left',va='center',fontsize=8.5,color='#333333',
                            arrowprops=dict(arrowstyle='->',lw=.8,color='#333333',shrinkA=2,shrinkB=0),
                            annotation_clip=False)
            if j == 0:
                ax.set_ylabel(f"Amplitude [{p['units']}]")
                ax.legend([f"{LABELS[i]}   |   N={p['static_n']:,}"],loc='upper right',fontsize=10,framealpha=.93)
            if j == 1:
                ax.axvline(1/365.2425,color='#888888',ls=':',lw=.9)
            if i == 3:
                ax.set_xlabel('Frequency [cycles/day]')
    for ax,title in zip(axs[0],['Full display band · 24 h reference','Low frequencies · dotted: annual reference','Daily-frequency neighborhood']):
        ax.set_title(title,fontsize=12,pad=12)
    fig.suptitle('LEO West (x=1, y=4) · Four-channel floating-mean GLS amplitudes',fontsize=19,x=.52,y=.974)
    fig.text(.52,.929,WARNING,ha='center',color='#9e2926',fontsize=13,weight='bold')
    fig.text(.52,.904,f"{report['common_start']} – {report['common_last']}  |  T={report['span_days']:.3f} days  |  Δf=1/T={report['frequency_step']:.6f} cycles/day",ha='center',fontsize=11)
    fig.text(.065,.061,r'$C(t)=c+\operatorname{Re}\{Z_f e^{2\pi ift}\},\quad Z_f=a-ib,\quad A(f)=|Z_f|$; independent GLS fit at each frequency; original timestamps, equal weights, existing masks.',fontsize=10)
    fig.text(.065,.039,'Air: 1 m horizontal offset; source umol/mol. Excel Z=0.25 m vs Oracle BOXZ=0.05 m unresolved. No unit conversion or time-zone correction.',fontsize=10)
    fig.text(.065,.018,'Gaps and review flags retained. 520 occupied common days ≠ continuous validated data. Only 1.43 annual cycles; annual periodicity is not established.',fontsize=10)
    fig.subplots_adjust(left=.065,right=.985,top=.858,bottom=.125,hspace=.29,wspace=.21)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'four_channel_spectra_17x11.{ext}',dpi=300)
        if ext == 'svg':
            path=out/f'four_channel_spectra_17x11.{ext}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    fig,axs=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for i,(ax,sp) in enumerate(zip(axs,static)):
        ax.plot(sp['f'],sp['sampling'],color=COLORS[i],lw=.9,label=LABELS[i]);ax.legend(loc='upper right')
        ax.set_ylabel('W(f) [dimensionless]');ax.set_yscale('log');ax.set_ylim(1e-10,1);ax.grid(alpha=.2)
        ax.axvline(1,color='gray',ls='--',lw=.8)
    axs[-1].set_xlabel('Frequency offset [cycles/day]')
    fig.suptitle('Actual sampling window: W(f)=|mean exp(2πift)|²; W(0)=1\n'+WARNING,fontsize=12)
    fig.text(.1,.015,'Window sidelobes indicate possible aliases around a signal frequency; they do not prove that a measured peak is an artifact.',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.93));fig.savefig(out/'sampling_window.png',dpi=180);plt.close(fig)


def historical_diagnostic(work, raw, out=OUT):
    start = pd.Timestamp('2023-07-08')
    before = START
    history = {}
    statistics = []
    for sid in IDS[1:]:
        r = raw[sid]
        r = r[r.localdatetime.ge(start) & r.localdatetime.lt(before)]
        history[sid]=r
        positive = r[r.datavalue.gt(0)]
        for month,g in positive.groupby(positive.localdatetime.dt.to_period('M')):
            statistics.append(dict(sensorid=sid,month=str(month),n=len(g),minimum=g.datavalue.min(),
                q05=g.datavalue.quantile(.05),median=g.datavalue.median(),q95=g.datavalue.quantile(.95),maximum=g.datavalue.max()))
    old=history[1035]
    positive=old[old.datavalue.gt(0)]
    scenario=[]
    for relax in (False,True):
        days=[]
        for sid in IDS:
            w=work[sid]
            keep=w.technically_retained.copy()
            if relax and sid==1035:
                # Diagnostic counterfactual only. These points never passed downstream QC.
                keep |= w.exclusion_reason.eq('outside_0_3000') & w.original_value.gt(3000)
            sel=w[keep & w.localdatetime.ge(start) & w.localdatetime.lt(END)]
            days.append(set(sel.localdatetime.dt.normalize()))
        common=set.intersection(*days)
        scenario.append(dict(scenario='restore only >3000 at 1035; unreviewed counterfactual' if relax else 'existing masks',
                             common_days=len(common),first=str(min(common).date()),last=str(max(common).date())))
    result=dict(interval_start=str(start),interval_end_exclusive=str(before),raw_rows=len(old),
        raw_above_3000=int(old.datavalue.gt(3000).sum()),positive_min=float(positive.datavalue.min()),
        positive_max=float(positive.datavalue.max()),positive_median=float(positive.datavalue.median()),
        positive_q05=float(positive.datavalue.quantile(.05)),positive_q95=float(positive.datavalue.quantile(.95)),
        scenarios=scenario,warning='No threshold changed. Restored points are not scientifically admitted and have not passed downstream QC.')
    pd.DataFrame(statistics).to_csv(out/'historical_monthly_ranges.csv',index=False)
    (out/'historical_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axs=plt.subplots(4,1,figsize=(14,10),sharex=True)
    for ax,sid,color in zip(axs,IDS[1:],COLORS[1:]):
        r=history[sid]
        ax.scatter(r.localdatetime,r.datavalue,s=.15,color=color,rasterized=True,label=f'{sid} · original source rows')
        ax.axhline(3000,color='#333333',ls='--',lw=.8,label='Existing upper mask: 3000 ppm')
        ax.set_ylabel('Original CO₂ [ppm]');ax.legend(loc='lower left',fontsize=9);ax.grid(alpha=.2)
    axs[3].scatter(positive.localdatetime,positive.datavalue,s=.15,color=COLORS[3],rasterized=True)
    axs[3].set_ylabel('1035 zoom [ppm]');axs[3].grid(alpha=.2)
    axs[3].text(.02,.93,'Zoom: all positive 1035 source rows; every value exceeds 3000 ppm',transform=axs[3].transAxes,va='top',fontsize=9)
    fig.suptitle('Before 29 April 2025 · 1035 and neighboring depths (5 / 20 / 50 cm)\n'+WARNING,fontsize=12)
    fig.text(.08,.02,'All source rows plotted, including service codes and conflicting duplicates. Near-7000 plateau is descriptive; cause and admission unresolved.',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.93));fig.savefig(out/'1035_before_candidate.png',dpi=180);plt.close(fig)
    return result


def sliding(series, out=OUT, window_days=30, step_days=1):
    starts=window_starts((END-START).days,window_days,step_days)
    frequencies=grid(window_days)
    spectra=[]; stats=[]
    for k,lo in enumerate(starts):
        frame=[]
        for s in series:
            mask=(s['t']>=lo)&(s['t']<lo+window_days)
            t,y=s['t'][mask],s['y'][mask]
            info=coverage(t,lo,lo+window_days)
            amp,_=gls(t,y,frequencies)
            # A displayed frequency must span >=1 cycle in the actual observations.
            # This is a display sufficiency rule, never an admission criterion.
            amp[frequencies*info['observed_span_days'] < 1-1e-12]=np.nan
            frame.append(amp)
            stats.append(dict(window=k,sensorid=s['sid'],start=str(START+pd.Timedelta(days=lo)),
                end_exclusive=str(START+pd.Timedelta(days=lo+window_days)),review_rows=int(s['q'][mask].sum()),**info))
        spectra.append(frame)
        if k % 100 == 0:
            print(f'Sliding spectra: {k+1}/{len(starts)}',flush=True)
    pd.DataFrame(stats).to_csv(out/'sliding_30d_1d_coverage.csv',index=False)
    return starts,frequencies,np.asarray(spectra),stats


def export_gif(starts,frequencies,spectra,stats,out=OUT):
    fig,axs=plt.subplots(4,1,figsize=(10,7.4),sharex=True,dpi=105)
    lines=[];notes=[]
    for i,ax in enumerate(axs):
        line,=ax.plot(frequencies,spectra[0,i],color=COLORS[i],lw=1)
        ax.set_xlim(0,FMAX);ax.set_ylim(0,max(1,float(np.nanmax(spectra[:,i]))*1.1));ax.grid(alpha=.2)
        ax.set_ylabel('A [umol/mol]' if i==0 else 'A [ppm]',fontsize=9)
        ax.axvline(1,color='gray',ls='--',lw=.7)
        notes.append(ax.text(.98,.92,'',transform=ax.transAxes,ha='right',va='top',fontsize=8,
                             bbox=dict(facecolor='white',alpha=.9,edgecolor='none')))
        lines.append(line)
    axs[-1].set_xlabel('Frequency [cycles/day] · daily reference dashed')
    title=fig.suptitle('',fontsize=11)
    fig.text(.1,.023,'30-day windows, 1-day steps · no annual inference · original timestamps · gaps preserved',fontsize=9)
    fig.text(.1,.005,'INSUFFICIENT DATA: <4 unique times, singular fit, or <1 observed cycle at a frequency; not scientific admission.',fontsize=8)
    fig.subplots_adjust(top=.89,bottom=.13,left=.095,right=.985,hspace=.18)
    frames=[]
    for k,lo in enumerate(starts):
        title.set_text(f'{WARNING}\n{(START+pd.Timedelta(days=lo)).date()} to {(START+pd.Timedelta(days=lo+30)).date()} (end excluded)')
        for i,line in enumerate(lines):
            line.set_ydata(spectra[k,i]);s=stats[k*4+i]
            gap='n/a' if s['max_gap_h'] is None else f"{s['max_gap_h']:.2f} h"
            notes[i].set_text(f"{LABELS[i]} · N={s['n']} · days={s['days']}/30 · max gap={gap}"+
                             ('\nINSUFFICIENT DATA' if not np.isfinite(spectra[k,i]).any() else ''))
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3]).quantize(colors=96))
    frames[0].save(out/'sliding_spectra_30d_1d.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True,disposal=1)
    plt.close(fig)


def write_html(series,static,report,out=OUT):
    template=Path(__file__).with_name('preliminary_spectra.html').read_text()
    payload=dict(start=str(START),end=str(END),days=(END-START).days,warning=WARNING,
        labels=LABELS,colors=COLORS,passport=report,
        series=[dict(sid=s['sid'],units=s['units'],t=s['t'].tolist(),y=s['y'].tolist(),q=s['q'].astype(int).tolist()) for s in series],
        static=[{k:v.tolist() for k,v in s.items()} for s in static])
    packed=base64.b64encode(gzip.compress(json.dumps(payload,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode()
    for mode,name in [('static','interactive_spectra.html'),('sliding','sliding_spectra.html')]:
        (out/name).write_text(template.replace('__MODE__',mode).replace('__PAYLOAD__',packed))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-gif',action='store_true');args=parser.parse_args()
    series,static,report,work,raw=prepare()
    print('Static spectra computed',flush=True)
    plot_static(static,report)
    historical_diagnostic(work,raw)
    write_html(series,static,report)
    if not args.skip_gif:
        starts,f,amp,stats=sliding(series)
        export_gif(starts,f,amp,stats)
    for path,checksum in report['hashes'].items():
        if digest(ROOT/path)!=checksum:
            raise ValueError(f'Source changed during calculation: {path}')
    print('Artifacts complete; source checksums unchanged',flush=True)


if __name__=='__main__':
    main()

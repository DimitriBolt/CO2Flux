"""Stage 0.4: direct Fourier phase differences at 1/day; immutable local inputs.

Run this file, or run 00_preliminary_fourier_phases.ipynb from a clean kernel.
No Oracle, interpolation, compressed-calendar FFT, or scientific admission rule.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preliminary_spectra import ROOT, ARCHIVE, START, END, IDS, digest, read_inputs

OUT = ROOT / 'Project_description/Research_log/output/preliminary_fourier_phases'
SOURCE_SHA = '886733c80caea324d4154553e1b9e35ab387bef9'
F0 = 1.0
PAIRS = ('air_to_5', '5_to_20', '20_to_50')
LABELS = ('Воздух → 5 см', '5 → 20 см', '20 → 50 см')
REVIEW = 'ВЫЧИСЛЕНО / ТРЕБУЕТ ПРОВЕРКИ'
UNASSESSED = 'ВЫЧИСЛЕНО / НАУЧНАЯ НАДЁЖНОСТЬ НЕ ОЦЕНЕНА'


def wrap(phi):
    """Map radians to [-pi, pi), including +pi -> -pi."""
    return (phi + np.pi) % (2 * np.pi) - np.pi


def direct_fourier(t_days, x, f0=F0):
    """One coefficient per column; time in days from ONE shared origin.

    The caller provides only the common, actually occupied bin centers.
    No coverage/amplitude cutoff is applied. Zero coefficients have no phase.
    """
    t_days, x = np.asarray(t_days, float), np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    if len(t_days) != len(x) or x.ndim != 2:
        raise ValueError('Timestamp/value shape mismatch')
    if len(t_days) < 2:
        raise ValueError('INSUFFICIENT DATA: менее двух общих ячеек; после центрирования нет ненулевой фазы')
    if not np.isfinite(x).all() or not np.isfinite(t_days).all():
        raise ValueError('INSUFFICIENT DATA: нечисловое значение в общей сетке')
    centered = x - np.mean(x, axis=0)
    exponential = np.exp(-2j * np.pi * f0 * t_days)
    return np.sum(centered * exponential[:, None], axis=0) / len(t_days)


def phase_difference(f_upper, f_lower):
    """Positive means second sinusoid lags first, modulo one period."""
    if abs(f_upper) == 0 or abs(f_lower) == 0:
        return np.nan
    return wrap(np.angle(f_upper * np.conj(f_lower)))


def aggregate_bins(frames, start=START, end=END):
    """Arithmetic mean of retained SOURCE observations, not means of means.

    Frames are existing source_decisions tables with the existing masks.
    Missing concentrations stay NaN. Count/flag zero denotes no observation,
    not an invented concentration. Global midnight-aligned bins can be reused
    in any integer-day window without changing that window's aggregation.
    """
    index = pd.date_range(start, end, freq='2h', inclusive='left')
    grid = pd.DataFrame(index=index)
    grid.index.name = 'bin_start'
    grid['bin_center'] = grid.index + pd.Timedelta(hours=1)
    grid['t_days'] = (grid['bin_center'] - start).dt.total_seconds() / 86400
    for sid in IDS:
        frame = frames[sid]
        keep = frame.technically_retained & frame.localdatetime.ge(start) & frame.localdatetime.lt(end)
        retained = frame.loc[keep].copy()
        if not np.isfinite(retained.datavalue.to_numpy(float)).all():
            raise ValueError(f'Retained nonfinite source value: {sid}')
        retained['bin_start'] = retained.localdatetime.dt.floor('2h')
        retained['review'] = retained.doubtful | retained.plateau_review
        grouped = retained.groupby('bin_start').agg(
            mean=('datavalue', 'mean'), n=('datavalue', 'size'),
            doubtful=('doubtful', 'any'), plateau_review=('plateau_review', 'any'),
            review=('review', 'any'), review_n=('review', 'sum'))
        grid[f'mean_{sid}'] = grouped['mean'].reindex(index)
        for col in ('n', 'review_n'):
            grid[f'{col}_{sid}'] = grouped[col].reindex(index, fill_value=0).astype(int)
        for col in ('doubtful', 'plateau_review', 'review'):
            grid[f'{col}_{sid}'] = grouped[col].reindex(index, fill_value=False).astype(bool)
    grid['complete'] = grid[[f'mean_{s}' for s in IDS]].notna().all(axis=1)
    grid['any_review'] = grid[[f'review_{s}' for s in IDS]].any(axis=1)
    return grid


def longest_missing_run(complete):
    """Maximum consecutive absent common bins, INCLUDING window edges."""
    best = run = 0
    for occupied in complete:
        run = 0 if occupied else run + 1
        best = max(best, run)
    return best


def window_slices(total_bins, window_days=30, step_days=1):
    for value in (window_days, step_days):
        if not np.isfinite(value) or value < 1 or int(value) != value:
            raise ValueError('Длина и шаг должны быть положительными целыми числами суток')
    width, step = int(window_days) * 12, int(step_days) * 12
    if width > total_bins:
        raise ValueError('INSUFFICIENT DATA: окно длиннее кандидатного календаря')
    return [(lo, lo + width) for lo in range(0, total_bins - width + 1, step)]


def compute_windows(grid, window_days=30, step_days=1):
    phases, coverage = [], []
    for k, (lo, hi) in enumerate(window_slices(len(grid), window_days, step_days)):
        window = grid.iloc[lo:hi]
        common = window['complete'].to_numpy(bool)
        used = window.loc[common]
        n = len(used)
        start, end = window.index[0], window.index[-1] + pd.Timedelta(hours=2)
        info = dict(window=k, start=start, end_exclusive=end, center=start+(end-start)/2,
                    possible_bins=len(window), complete_bins=n, fraction_complete=n/len(window),
                    coverage_percent=100*n/len(window),
                    longest_gap_bins=longest_missing_run(common),
                    longest_gap_hours=2*longest_missing_run(common),
                    doubtful_bins=int(used.any_review.sum()),
                    doubtful_bins_all=int(window.any_review.sum()),
                    window_days=window_days, step_days=step_days)
        for sid in IDS:
            info[f'retained_observations_{sid}'] = int(window[f'n_{sid}'].sum())
            info[f'used_observations_{sid}'] = int(used[f'n_{sid}'].sum())
            info[f'review_bins_{sid}'] = int(used[f'review_{sid}'].sum())
        row = {name: info[name] for name in ('window', 'start', 'end_exclusive', 'center')}
        row.update(f0_cycles_per_day=F0, **{p+'_rad': np.nan for p in PAIRS},
                   **{p+'_deg': np.nan for p in PAIRS})
        reason = ''
        try:
            coefficients = direct_fourier(used.t_days.to_numpy(), used[[f'mean_{s}' for s in IDS]].to_numpy())
            for j, sid in enumerate(IDS):
                row[f'F_real_{sid}'], row[f'F_imag_{sid}'] = coefficients[j].real, coefficients[j].imag
                row[f'F_abs_{sid}'] = abs(coefficients[j])
            for j, pair in enumerate(PAIRS):
                angle = phase_difference(coefficients[j], coefficients[j+1])
                row[pair+'_rad'], row[pair+'_deg'] = angle, angle * 180 / np.pi
                row[pair+'_reason'] = '' if np.isfinite(angle) else 'Нулевой Fourier-коэффициент; аргумент не определён'
            if not all(np.isfinite(row[p+'_rad']) for p in PAIRS):
                reason = 'INSUFFICIENT DATA: нулевой коэффициент хотя бы одного канала; часть фаз не определена'
        except ValueError as error:
            reason = str(error)
            for pair in PAIRS:
                row[pair+'_reason'] = reason
        status = 'INSUFFICIENT DATA' if reason else (REVIEW if info['doubtful_bins'] else UNASSESSED)
        row.update(status=status, reason=reason, scientific_reliability='НЕ УСТАНОВЛЕНА')
        info.update(status=status, reason=reason)
        phases.append(row)
        coverage.append(info)
    return pd.DataFrame(phases), pd.DataFrame(coverage)


def broken_line(times, values):
    """Insert NaN at branch crossings; never discard either endpoint."""
    x, y = [], []
    previous = np.nan
    for time, value in zip(times, values):
        if np.isfinite(previous) and np.isfinite(value) and abs(value-previous) > 180:
            x.append(time)
            y.append(np.nan)
        x.append(time)
        y.append(value)
        previous = value
    return x, y


def print_figure(phases, coverage, out=OUT):
    fig, axes = plt.subplots(3, 1, figsize=(17, 11), sharex=True)
    fig.subplots_adjust(left=.095, right=.985, top=.83, bottom=.15, hspace=.23)
    fig.suptitle('CO₂: временная эволюция трёх Fourier-разностей фаз', fontsize=23, y=.965)
    fig.text(.095, .915, 'LEO West · (x, y) = (1, 4) · 1275 / 996 / 1012 / 1035 · f₀ = 1 цикл/сутки (24 ч)', fontsize=13)
    fig.text(.095, .880, f'Окно {int(coverage.window_days.iloc[0])} суток / шаг {int(coverage.step_days.iloc[0])} сутки · общая сетка 2 ч · центры окон', fontsize=13)
    colors = ['#b56418', '#b56418', '#b56418']
    for ax, pair, label, color in zip(axes, PAIRS, LABELS, colors):
        values = phases[pair+'_deg'].to_numpy()
        tx, vy = broken_line(phases.center, values)
        ax.plot(tx, vy, color='#aaa49c', lw=.7)
        review = coverage.doubtful_bins.to_numpy() > 0
        ax.scatter(phases.center[review], values[review], c=color, s=9, label='Есть флаги проверки')
        if (~review).any():
            ax.scatter(phases.center[~review], values[~review], c='#186c7b', s=9, label='Без флагов; надёжность не установлена')
        absent = ~np.isfinite(values)
        if absent.any():
            ax.scatter(phases.center[absent], np.full(absent.sum(), -172), marker='x', color='#b32025', s=25, label='Недостаточно данных')
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.axhline(0, color='#757575', lw=.5)
        ax.grid(alpha=.20)
        ax.set_ylabel('Разность фаз, °', fontsize=11)
        ax.set_title(label, loc='left', fontsize=14, fontweight='bold', pad=6)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.985,.912), fontsize=10, frameon=False)
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%Y'))
    axes[-1].set_xlabel('Календарная дата центра окна', fontsize=11)
    cmin, cmax = coverage.complete_bins.min(), coverage.complete_bins.max()
    pctmin, pctmax = coverage.coverage_percent.min(), coverage.coverage_percent.max()
    fig.text(.095, .080, f'{len(phases)} окон · общих ячеек {cmin}–{cmax} из {coverage.possible_bins.iloc[0]} · покрытие {pctmin:.1f}–{pctmax:.1f}% · максимальный пробел {coverage.longest_gap_hours.max()} ч', fontsize=11)
    fig.text(.095, .052, 'ПРЕДВАРИТЕЛЬНО: контроль LI-COR и временные шкалы не подтверждены. Оранжевые точки требуют проверки.', color='#91490a', fontsize=11)
    fig.text(.095, .026, 'Ветвь [−180°, 180°); линии разорваны при переходе через границу. Положительный знак: второй сигнал запаздывает в синтетической синусоиде.', fontsize=10)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out/f'phase_evolution_17x11.{extension}', dpi=300)
    plt.close(fig)


def trajectory_figure(phases, out=OUT):
    valid = phases[[p+'_deg' for p in PAIRS]].notna().all(axis=1)
    points = phases.loc[valid]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    t = mdates.date2num(points.center)
    artist = ax.scatter(*(points[p+'_deg'] for p in PAIRS), c=t, cmap='viridis', s=12)
    for setter, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), LABELS):
        setter(label + ', °', labelpad=10)
    ax.set(xlim=(-180, 180), ylim=(-180, 180), zlim=(-180, 180))
    bar = fig.colorbar(artist, ax=ax, shrink=.55, pad=.14)
    ticks = np.linspace(t.min(), t.max(), 4)
    bar.set_ticks(ticks, labels=[mdates.num2date(d).strftime('%d.%m.%Y') for d in ticks])
    bar.set_label('Дата центра окна')
    fig.suptitle('Три циклические разности фаз: предварительные точки', y=.96, fontsize=15)
    fig.text(.07, .06, 'Пространство фаз: T³ = S¹ × S¹ × S¹. Грани куба отождествляются; точки не соединены.\nВсе реальные окна имеют флаги проверки; научная надёжность не установлена.', fontsize=10)
    fig.savefig(out/'phase_trajectory_3d.png', dpi=180)
    plt.close(fig)


def validation_examples():
    """Numerical evidence; independent unittest assertions live in tests/."""
    t = (np.arange(360)+.5)/12
    u, v = np.cos(2*np.pi*t), np.cos(2*np.pi*(t-4/24))
    def delta(ti, a, b):
        f = direct_fourier(ti, np.column_stack([a, b]))
        return float(phase_difference(*f)*180/np.pi)
    records = []
    for name, a, b, expected in [('A',u,u,0), ('B',u,v,60), ('C',v,u,-60),
                               ('D',u,np.cos(2*np.pi*(t-14/24)),-150)]:
        actual = delta(t,a,b)
        records.append(dict(test=name,expected_deg=expected,actual_deg=actual,error_deg=actual-expected))
    whole_days = ~((t>=10)&(t<13))
    irregular = np.ones(360, dtype=bool)
    irregular[np.array([7,8,9,27,48,121,122,203,278,301,339])] = False
    for name, mask in [('E_whole_days',whole_days), ('E_irregular',irregular)]:
        actual = delta(t[mask],u[mask],v[mask])
        records.append(dict(test=name,expected_deg=60,actual_deg=actual,error_deg=actual-60,
                            retained_bins=int(mask.sum()), removed_indices=np.flatnonzero(~mask).tolist()))
    f = direct_fourier(t, np.column_stack([u,v]))
    fft = np.fft.fft(np.column_stack([u,v])-np.mean(np.column_stack([u,v]),axis=0),axis=0)[30]/len(t)
    fft *= np.exp(-2j*np.pi*t[0])
    records.append(dict(test='F',max_complex_error=float(np.max(np.abs(f-fft))),
                        note='FFT mode 30; center-origin correction exp(-2πi f0 t_first)'))
    return records


def html_payload(grid, passport):
    def finite_list(values):
        return [float(v) if np.isfinite(v) else None for v in values]
    return dict(start=str(START.date()), end=str(END.date()), source_sha=SOURCE_SHA,
                t=grid.t_days.tolist(),
                y=[finite_list(grid[f'mean_{s}']) for s in IDS],
                n=[grid[f'n_{s}'].tolist() for s in IDS],
                q=[grid[f'review_{s}'].astype(int).tolist() for s in IDS],
                passport=passport)


def write_html(grid, passport, out=OUT):
    from plotly.offline import get_plotlyjs
    template = Path(__file__).with_suffix('.html').read_text()
    engine = Path(__file__).with_suffix('.js').read_text()
    common = template.replace('__PLOTLY__', get_plotlyjs()).replace('__ENGINE__', engine)
    common = common.replace('__DATA__', json.dumps(html_payload(grid,passport), ensure_ascii=False, allow_nan=False).replace('</','<\\/'))
    for name, mode in [('phase_evolution.html','time'), ('phase_trajectory_3d.html','3d')]:
        (out/name).write_text(common.replace('__MODE__',mode))


def run_tests():
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).parent/'tests'), '-p', 'test_preliminary_fourier_phases.py', '-v'],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def browser_cross_check(grid):
    """Numerical validation of the actual embedded data at two UI settings."""
    engine = Path(__file__).with_suffix('.js')
    records = []
    for width, step in ((30,1), (60,7)):
        script = ('const fs=require("fs"),e=require('+json.dumps(str(engine))+');'
                  'const d=JSON.parse(fs.readFileSync(0,"utf8"));'
                  f'process.stdout.write(JSON.stringify(e.calculatePhases(d,{width},{step})));')
        actual = json.loads(subprocess.check_output(['node','-e',script],
                            input=json.dumps(html_payload(grid,{}),allow_nan=False),text=True))
        phases, coverage = compute_windows(grid,width,step)
        js_phase = np.array([r['phase'] for r in actual],float)
        py_phase = phases[[p+'_deg' for p in PAIRS]].to_numpy()
        np.testing.assert_allclose(js_phase,py_phase,atol=1e-9,rtol=0,equal_nan=True)
        assert [r['n'] for r in actual] == coverage.complete_bins.tolist()
        assert [r['doubtful'] for r in actual] == coverage.doubtful_bins.tolist()
        assert [r['longestGapHours'] for r in actual] == coverage.longest_gap_hours.tolist()
        records.append(dict(window_days=width,step_days=step,windows=len(actual),
                            maximum_phase_difference_deg=float(np.nanmax(np.abs(js_phase-py_phase)))))
    return records


def generate(out=OUT, window_days=30, step_days=1):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    test_log = run_tests()
    work, raw, metadata, hashes = read_inputs()
    grid = aggregate_bins(raw)
    phases, coverage = compute_windows(grid, window_days, step_days)
    browser_checks = browser_cross_check(grid)
    passport = dict(source_sha=SOURCE_SHA, stage='0.4', priority='temporal phase evolution',
                    archive=str(ARCHIVE.relative_to(ROOT)), input_hashes=hashes,
                    calendar_start=str(START), calendar_end_exclusive=str(END),
                    time_origin=str(START), bin_width_hours=2, bin_center_offset_hours=1,
                    f0_cycles_per_day=F0, channels=[metadata[s] for s in IDS],
                    source='source_decisions.parquet; existing technically_retained mask',
                    review_reasons={str(s): raw[s].loc[raw[s].technically_retained & raw[s].localdatetime.ge(START) & raw[s].localdatetime.lt(END), 'doubt_reason'].value_counts().to_dict() for s in IDS},
                    limitations=['LI-COR QC unresolved', 'subsurface review flags retained',
'no scientific coverage/amplitude threshold',
                                 'missing bins may bias direct Fourier phase',
                                 'near-zero coefficient phase can be unstable',
                                 'overlapping windows are dependent'])
    grid.to_csv(out/'common_2h_bins.csv', float_format='%.17g')
    phases.to_csv(out/'phase_timeseries.csv', index=False, float_format='%.17g')
    coverage.to_csv(out/'window_coverage.csv', index=False, float_format='%.17g')
    (out/'passport.json').write_text(json.dumps(passport,ensure_ascii=False,indent=2)+'\n')
    print_figure(phases, coverage, out)
    trajectory_figure(phases, out)
    write_html(grid, passport, out)
    unchanged = all(digest(ROOT/path) == expected for path,expected in hashes.items())
    if not unchanged:
        raise RuntimeError('Input changed during phase calculation')
    from PIL import Image
    from pypdf import PdfReader
    box = PdfReader(out/'phase_evolution_17x11.pdf').pages[0].mediabox
    pdf_inches = [float(box.width)/72, float(box.height)/72]
    png_size = list(Image.open(out/'phase_evolution_17x11.png').size)
    assert pdf_inches == [17,11] and png_size == [5100,3300]
    valid = phases[[p+'_deg' for p in PAIRS]].notna().all(axis=1)
    validation = dict(source_sha=SOURCE_SHA, stage='0.4', tests_passed=True, test_log=test_log,
        synthetic=validation_examples(), browser_python_cross_check=browser_checks, input_hashes_unchanged=unchanged,
        total_windows=len(phases), all_three_phases_computable=int(valid.sum()),
        windows_with_review_flags=int(coverage.doubtful_bins.gt(0).sum()),
        insufficient_windows=int((~valid).sum()), scientifically_reliable_windows=None,
        scientific_reliability='NOT ESTABLISHED; no admission criterion introduced',
        common_bins=int(grid.complete.sum()), possible_calendar_bins=len(grid),
        coverage_percent_range=[float(coverage.coverage_percent.min()),float(coverage.coverage_percent.max())],
        complete_bins_range=[int(coverage.complete_bins.min()),int(coverage.complete_bins.max())],
        longest_gap_hours=int(coverage.longest_gap_hours.max()),
        center_range=[str(phases.center.min()),str(phases.center.max())],
        pdf_inches=pdf_inches,png_pixels=png_size,
        python_versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__),
        visual_inspection='See verified delivery record in details/02_data_checks.md')
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    return grid, phases, coverage, validation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-days', type=int, default=30)
    parser.add_argument('--step-days', type=int, default=1)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    _, _, _, report = generate(args.output,args.window_days,args.step_days)
    print(json.dumps({k:v for k,v in report.items() if k not in ('test_log','synthetic')},ensure_ascii=False,indent=2))

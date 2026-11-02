"""Stage 0.4: direct Fourier phase differences at 1/day; immutable local inputs.

Run this file, or run 00_preliminary_fourier_phases.ipynb from a clean kernel.
The notebook also diagnoses one saved 5->20 cm phase jump (Stage 0.5 / Q1).
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
        row.update(f0_cycles_per_day=F0, **{p+'_rad': np.nan for p in PAIRS})
        reason = ''
        try:
            coefficients = direct_fourier(used.t_days.to_numpy(), used[[f'mean_{s}' for s in IDS]].to_numpy())
            for j, sid in enumerate(IDS):
                row[f'F_real_{sid}'], row[f'F_imag_{sid}'] = coefficients[j].real, coefficients[j].imag
                row[f'F_abs_{sid}'] = abs(coefficients[j])
            for j, pair in enumerate(PAIRS):
                angle = phase_difference(coefficients[j], coefficients[j+1])
                row[pair+'_rad'] = angle
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
        if np.isfinite(previous) and np.isfinite(value) and abs(value-previous) > np.pi:
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
        values = phases[pair+'_rad'].to_numpy()
        tx, vy = broken_line(phases.center, values)
        ax.plot(tx, vy, color='#aaa49c', lw=.7)
        review = coverage.doubtful_bins.to_numpy() > 0
        ax.scatter(phases.center[review], values[review], c=color, s=9, label='Есть флаги проверки')
        if (~review).any():
            ax.scatter(phases.center[~review], values[~review], c='#186c7b', s=9, label='Без флагов; надёжность не установлена')
        absent = ~np.isfinite(values)
        if absent.any():
            ax.scatter(phases.center[absent], np.full(absent.sum(), -np.pi*.95), marker='x', color='#b32025', s=25, label='Недостаточно данных')
        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                      labels=['−π', '−π/2', '0', 'π/2', 'π'])
        ax.axhline(0, color='#757575', lw=.5)
        ax.grid(alpha=.20)
        ax.set_ylabel('Разность фаз Δφ', fontsize=11)
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
    fig.text(.095, .052, 'ПРЕДВАРИТЕЛЬНО: контроль LI-COR не завершён. Оранжевые точки требуют проверки.', color='#91490a', fontsize=11)
    fig.text(.095, .026, 'Ветвь [−π, π); линии разорваны при переходе через границу. Положительный знак: второй сигнал запаздывает в синтетической синусоиде.', fontsize=10)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out/f'phase_evolution_17x11.{extension}', dpi=300)
    plt.close(fig)


def trajectory_figure(phases, out=OUT):
    valid = phases[[p+'_rad' for p in PAIRS]].notna().all(axis=1)
    points = phases.loc[valid]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    t = mdates.date2num(points.center)
    artist = ax.scatter(*(points[p+'_rad'] for p in PAIRS), c=t, cmap='viridis', s=12)
    for setter, label in zip((ax.set_xlabel, ax.set_ylabel, ax.set_zlabel), LABELS):
        setter(label + ', рад', labelpad=10)
    ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), zlim=(-np.pi, np.pi))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        axis.set_ticklabels(['−π', '−π/2', '0', 'π/2', 'π'])
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
        return float(phase_difference(*f))
    records = []
    for name, a, b, expected in [('A',u,u,0), ('B',u,v,np.pi/3), ('C',v,u,-np.pi/3),
                               ('D',u,np.cos(2*np.pi*(t-14/24)),-5*np.pi/6)]:
        actual = delta(t,a,b)
        records.append(dict(test=name,expected_rad=expected,actual_rad=actual,error_rad=actual-expected))
    whole_days = ~((t>=10)&(t<13))
    irregular = np.ones(360, dtype=bool)
    irregular[np.array([7,8,9,27,48,121,122,203,278,301,339])] = False
    for name, mask in [('E_whole_days',whole_days), ('E_irregular',irregular)]:
        actual = delta(t[mask],u[mask],v[mask])
        records.append(dict(test=name,expected_rad=np.pi/3,actual_rad=actual,error_rad=actual-np.pi/3,
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
    (out/'phase_evolution.html').write_text(common)


def run_tests():
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s',
        str(Path(__file__).parent/'tests'), '-p', 'test_preliminary_fourier_phases.py', '-v'],
        capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout + result.stderr


def load_saved_results(out=OUT):
    """Read the published 30/1 results without regenerating Stage 0.4 artifacts."""
    out = Path(out)
    dates = ['start', 'end_exclusive', 'center']
    grid = pd.read_csv(out/'common_2h_bins.csv', index_col='bin_start',
                       parse_dates=['bin_start', 'bin_center'], float_precision='round_trip')
    phases = pd.read_csv(out/'phase_timeseries.csv', parse_dates=dates, float_precision='round_trip')
    coverage = pd.read_csv(out/'window_coverage.csv', parse_dates=dates, float_precision='round_trip')
    return grid, phases, coverage, json.loads((out/'validation.json').read_text())


def select_phase_jump(phases):
    """Largest |raw phase difference| for 5->20 on adjacent daily centers.

    Earliest pair wins an exact tie. This is an example-selection rule, not
    a quality threshold; only this pair is subsequently diagnosed.
    """
    if not phases.center.is_monotonic_increasing or phases.center.duplicated().any():
        raise ValueError('Expected unique chronological window centers')
    delta = phases['5_to_20_rad'].diff().to_numpy()
    adjacent = phases.center.diff().eq(pd.Timedelta(days=1)).to_numpy()
    eligible = adjacent & np.isfinite(delta)
    if not eligible.any():
        raise ValueError('No finite adjacent daily phase pair')
    right = int(np.argmax(np.where(eligible, np.abs(delta), -np.inf)))
    return right - 1, right


def complex_step(z_before, z_after):
    """Discrete endpoint diagnostics; infer no path between windows.

    Numerical zero only; no near-zero or scientific admission threshold.
    The circular increment uses the SAME phase_difference convention.
    """
    defined = (np.isfinite(z_before) and np.isfinite(z_after)
               and abs(z_before) > 0 and abs(z_after) > 0)
    before = wrap(np.angle(z_before)) if np.isfinite(z_before) and abs(z_before) > 0 else np.nan
    after = wrap(np.angle(z_after)) if np.isfinite(z_after) and abs(z_after) > 0 else np.nan
    raw = after - before
    circular = phase_difference(z_after, z_before) if defined else np.nan
    turns = int(np.rint((raw-circular)/(2*np.pi))) if defined else None
    return dict(abs_before=float(abs(z_before)), abs_after=float(abs(z_after)),
                modulus_ratio=float(abs(z_after)/abs(z_before)) if abs(z_before) > 0 else np.nan,
                relative_complex_change=float(abs(z_after-z_before)/abs(z_before)) if abs(z_before) > 0 else np.nan,
                arg_before_rad=float(before), arg_after_rad=float(after),
                raw_increment_rad=float(raw), circular_increment_rad=float(circular),
                branch_turns=turns, phase_defined=bool(defined))


def phase_jump_tables(phases, coverage, neighbors=3):
    """One selected pair plus local context, using saved complex coefficients."""
    if not isinstance(neighbors, int) or neighbors < 0:
        raise ValueError('neighbors must be a nonnegative integer')
    keys = ['window', 'start', 'end_exclusive', 'center']
    pd.testing.assert_frame_equal(phases[keys], coverage[keys])
    if not (phases.f0_cycles_per_day.eq(1).all() and coverage.window_days.eq(30).all()
            and coverage.step_days.eq(1).all()):
        raise ValueError('This diagnostic requires the unchanged 1/day, 30/1 protocol')
    left, right = select_phase_jump(phases)
    columns = keys + [f'F_{part}_{sid}' for sid in (996, 1012) for part in ('real', 'imag', 'abs')]
    columns += ['5_to_20_rad', 'status', 'scientific_reliability']
    local = phases.iloc[max(0, left-neighbors):right+neighbors+1][columns].copy()
    for sid in (996, 1012):
        z = local[f'F_real_{sid}'] + 1j*local[f'F_imag_{sid}']
        np.testing.assert_allclose(np.abs(z), local[f'F_abs_{sid}'], rtol=1e-13, atol=1e-13)
    upper = local.F_real_996.to_numpy() + 1j*local.F_imag_996.to_numpy()
    lower = local.F_real_1012.to_numpy() + 1j*local.F_imag_1012.to_numpy()
    q = upper * np.conj(lower)
    local['Q_real'], local['Q_imag'], local['Q_abs'] = q.real, q.imag, np.abs(q)
    local['Q_arg_rad'] = [phase_difference(u, v) for u, v in zip(upper, lower)]
    np.testing.assert_allclose(local.Q_arg_rad, local['5_to_20_rad'], rtol=0, atol=2e-14)
    local['selected_endpoint'] = local.window.isin(phases.iloc[[left, right]].window)
    quality = coverage.loc[local.index].copy()
    records = []
    for name, z in [('F5', upper), ('F20', lower), ('Q', q)]:
        a, b = z[local.selected_endpoint.to_numpy()]
        records.append(dict(coefficient=name, **complex_step(a, b)))
    return local.reset_index(drop=True), quality.reset_index(drop=True), pd.DataFrame(records)


def phase_jump_figures(local, quality, out):
    """Static reusable figures. Points only: no interpolated coefficient path."""
    selected = local.loc[local.selected_endpoint]
    month = local.center.iloc[0].strftime('%B %Y')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    series = [('5_to_20_rad', 'Relative phase: 5 → 20 cm', 'Phase (rad)'),
              ('F_abs_996', '|F₅|', 'ppm'), ('F_abs_1012', '|F₂₀|', 'ppm'),
              ('Q_abs', '|Q| = |F₅| |F₂₀|', 'ppm²')]
    for ax, (column, title, unit) in zip(axes.flat, series):
        ax.scatter(local.center, local[column], color='#537782', s=35)
        for (_, row), color, marker in zip(selected.iterrows(), ('#b44b13', '#8c238b'), ('s', 'D')):
            ax.scatter(row.center, row[column], c=color, marker=marker, s=65, zorder=3,
                       label=row.center.strftime('%d %b'))
        ax.set(title=title, ylabel=unit, xlabel=f'Window center ({month})')
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylim(-np.pi, np.pi)
    axes[0, 0].set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                         labels=['−π', '−π/2', '0', 'π/2', 'π'])
    axes[0, 0].legend(loc='lower left', frameon=False)
    nmin, nmax = quality.complete_bins.min(), quality.complete_bins.max()
    fig.suptitle('One discrete phase jump · 1 cycle/day · 30-day windows · 1-day step\n'
                 f'{len(local)} windows: {nmin}–{nmax}/{quality.possible_bins.iloc[0]} common bins; '
                 'review flags retained', fontsize=14)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'phase_jump_local.{extension}', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax, real, imag, title, unit in zip(axes,
            ('F_real_996', 'F_real_1012', 'Q_real'),
            ('F_imag_996', 'F_imag_1012', 'Q_imag'),
            ('F₅', 'F₂₀', 'Q = F₅ conjugate(F₂₀)'), ('ppm', 'ppm', 'ppm²')):
        ax.axhline(0, color='#cccccc', lw=.7)
        ax.axvline(0, color='#cccccc', lw=.7)
        ax.scatter(0, 0, marker='+', color='black', s=80)
        ax.scatter(local[real], local[imag], c='#7e878a', s=25)
        for (_, row), color, marker in zip(selected.iterrows(), ('#b44b13', '#8c238b'), ('s', 'D')):
            ax.scatter(row[real], row[imag], c=color, marker=marker, s=70, zorder=3)
            ax.annotate(row.center.strftime('%d'), (row[real], row[imag]), xytext=(5, 3),
                        textcoords='offset points', fontsize=9)
        ax.set(title=title, xlabel=f'Re ({unit})', ylabel=f'Im ({unit})', aspect='equal')
        # Square panels retain equal geometric scale and keep the origin visible.
        xmin, xmax = min(0, local[real].min()), max(0, local[real].max())
        ymin, ymax = min(0, local[imag].min()), max(0, local[imag].max())
        span = max(xmax-xmin, ymax-ymin) * 1.35
        ax.set_xlim((xmin+xmax-span)/2, (xmin+xmax+span)/2)
        ax.set_ylim((ymin+ymax-span)/2, (ymin+ymax+span)/2)
        ax.grid(alpha=.15)
    axes[2].plot([axes[2].get_xlim()[0], 0], [0, 0], color='#b44b13',
                 ls='--', label='Arg branch boundary')
    axes[2].legend(loc='upper left', fontsize=9, frameon=False)
    a, b = selected.center.dt.strftime('%d %b %Y')
    fig.suptitle(f'Complex coefficients · square: {a}; diamond: {b}\n'
                 f'{len(local)-2} neighboring windows in grey; discrete points, no path between windows inferred.', fontsize=13)
    for extension in ('png', 'svg'):
        fig.savefig(out/f'phase_jump_complex.{extension}', dpi=180, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    for name in ('phase_jump_local.svg', 'phase_jump_complex.svg'):
        path = out/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def generate_phase_jump(source=OUT):
    """Verify Stage 0.4; export only one Q1 diagnostic into its own subfolder."""
    source = Path(source)
    previous = {p.name: digest(p) for p in source.iterdir() if p.is_file()}
    grid, phases, coverage, _ = load_saved_results(source)
    passport = json.loads((source/'passport.json').read_text())
    _, raw, _, hashes = read_inputs()
    if hashes != passport['input_hashes']:
        raise ValueError('The Stage 0.4 input snapshot changed')
    rebuilt_grid = aggregate_bins(raw)
    rebuilt_phases, rebuilt_coverage = compute_windows(rebuilt_grid, 30, 1)
    differences = {}
    for name, saved, rebuilt in [('grid', grid, rebuilt_grid), ('phases', phases, rebuilt_phases),
                                  ('coverage', coverage, rebuilt_coverage)]:
        # CSV loses frequency metadata and integer/float distinctions, not values.
        pd.testing.assert_frame_equal(saved.fillna(''), rebuilt.fillna(''),
                                      check_dtype=False, check_freq=False, check_exact=False,
                                      rtol=1e-12, atol=1e-12)
        numeric = saved.select_dtypes(include='number').columns.intersection(
            rebuilt.select_dtypes(include='number').columns)
        differences[name] = float(np.nanmax(np.abs(saved[numeric].to_numpy()-rebuilt[numeric].to_numpy())))
    local, quality, steps = phase_jump_tables(phases, coverage)
    test_log = run_tests()
    out = source/'jump_diagnostic'
    out.mkdir(exist_ok=True)
    for name, frame in [('windows', local), ('coverage', quality), ('step', steps)]:
        frame.to_csv(out/f'phase_jump_{name}.csv', index=False, float_format='%.17g')
    phase_jump_figures(local, quality, out)
    if not all(digest(source/name) == h for name, h in previous.items()):
        raise RuntimeError('A pre-existing Stage 0.4 artifact changed')
    if not all(digest(ROOT/name) == h for name, h in hashes.items()):
        raise RuntimeError('An input changed during the diagnostic')
    captions = {
        'phase_jump_local': 'One selected 5–20 cm phase jump at window centers 17 and 18 December 2025, '
            'with three neighboring windows on either side. The original phase branch is [−π, π). '
            'Moduli use the unchanged 1/N normalization at 1 cycle/day. All windows have 360/360 '
            'common two-hour bins and review flags; scientific reliability is not established.',
        'phase_jump_complex': 'Saved complex Fourier coefficients F5, F20 and Q = F5 conjugate(F20). '
            'The labeled square and diamond mark window centers 17 and 18 December 2025; '
            'the other six windows are shown in grey. The origin and the negative-real-axis branch boundary for Q are shown. '
            'Points are discrete estimates, not an interpolated trajectory. No sampled coefficient '
            'is zero; an intervening zero cannot be established or excluded by these endpoints.'}
    report = dict(stage='0.5 / Q1: one diagnostic only',
        baseline_commit='9059ae9306c1fbebaa821ca746c93f057f2e402a',
        implementation_sha256=digest(Path(__file__)),
        selection='Largest absolute raw 5_to_20_rad difference between finite daily-adjacent centers; earliest exact tie.',
        selected_windows=local.loc[local.selected_endpoint, 'window'].tolist(), neighbors_each_side=3,
        f0_cycles_per_day=1, window_days=30, step_days=1,
        baseline_artifact_sha256=previous, input_sha256=hashes,
        baseline_artifacts_unchanged=True, input_hashes_unchanged=True,
        stage04_recomputed_max_abs_difference=differences, tests_passed=True, test_log=test_log,
        mathematical_checks=dict(
            raw_phase_increment_rad=float(np.diff(local.loc[local.selected_endpoint, '5_to_20_rad'])[0]),
            circular_phase_increment_rad=float(steps.loc[steps.coefficient.eq('Q'), 'circular_increment_rad'].iloc[0]),
            decomposition_residual_rad=float(wrap(steps.iloc[0].circular_increment_rad
                                                  - steps.iloc[1].circular_increment_rad
                                                  - steps.iloc[2].circular_increment_rad))),
        captions=captions,
        limitations=['No new near-zero or quality threshold', 'No interpolation or intervening zero inference',
                     'Full bin coverage does not remove review flags', 'Overlapping windows are dependent',
                     'No physical event or mechanism established'])
    report['artifact_sha256'] = {p.name: digest(p) for p in out.iterdir()
                                  if p.is_file() and p.name != 'phase_jump_validation.json'}
    (out/'phase_jump_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return local, quality, steps, report


def load_saved_phase_jump(source=OUT):
    """Read historical Q1 tables; do not recompute or rewrite Stage 0.4/Q1."""
    out = Path(source)/'jump_diagnostic'
    tables = [pd.read_csv(out/f'phase_jump_{name}.csv', float_precision='round_trip',
                         parse_dates=['start', 'end_exclusive', 'center'] if name != 'step' else None)
              for name in ('windows', 'coverage', 'step')]
    return (*tables, json.loads((out/'phase_jump_validation.json').read_text()))


def saved_phase_jump_agreement(local):
    """Maximum circular discrepancy of the two saved Q1 phase columns."""
    angles = local[['Q_arg_rad', '5_to_20_rad']].to_numpy(float)
    if not len(angles) or not np.isfinite(angles).all():
        raise ValueError('Expected finite saved Q1 phases')
    difference = np.angle(np.exp(1j*(angles[:, 0]-angles[:, 1])))
    return float(np.max(np.abs(difference)))


def plot_phase_jump_unit_circles(local, steps, out=None):
    """Illustrate saved Before/After arguments; never recompute Fourier data.

    Optional export creates only phase_jump_unit_circles.png/.svg. The returned
    figure lets callers inspect the drawing; ordinary notebook execution reads
    the saved PNG. All positions use unrounded angles from the Q1 tables.
    """
    from matplotlib.patches import Circle, Wedge

    endpoints = local.loc[local.selected_endpoint].sort_values('window')
    if endpoints.window.tolist() != [217, 218]:
        raise ValueError('This illustration is limited to the selected Q1 pair')
    saved = steps.set_index('coefficient')
    angles = saved.loc[['F5', 'F20'], ['arg_before_rad', 'arg_after_rad']].to_numpy(float)
    phases = endpoints['5_to_20_rad'].to_numpy(float)
    if not np.isfinite(angles).all() or not np.all((phases > 0) & (phases < np.pi)):
        raise ValueError('Expected the saved positive Q1 sectors')
    np.testing.assert_allclose(angles[0]-angles[1], phases, rtol=0, atol=1e-14)
    np.testing.assert_allclose(
        saved.loc['Q', ['arg_before_rad', 'arg_after_rad']].to_numpy(float),
        phases, rtol=0, atol=1e-14)
    months = ('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')
    colors = ('#126b98', '#c46518')
    sector_color = '#8067a9'
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 7.6))
    fig.subplots_adjust(left=.06, right=.95, bottom=.30, top=.90, wspace=.25)
    for k, (ax, label) in enumerate(zip(axes, ('Before', 'After'))):
        theta5, theta20 = angles[:, k]
        date = pd.Timestamp(endpoints.center.iloc[k])
        date_text = f'{date.day:02d}.{months[date.month-1]}.{date.year}'
        ax.set_title(f'{label}, {date_text}', fontsize=16, pad=20)
        ax.set(xlim=(-1.32, 1.32), ylim=(-1.32, 1.32), aspect='equal',
               xticks=[-1, 0, 1], yticks=[-1, 0, 1])
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_position('zero')
        ax.spines[['bottom', 'left']].set_color('#a5acb4')
        ax.tick_params(labelsize=10, colors='#58616b', length=3)
        ax.text(1.29, -.13, 'Re', ha='right', va='top', fontsize=12)
        ax.text(.07, 1.28, 'Im', ha='left', va='top', fontsize=12)
        ax.add_patch(Circle((0, 0), 1, fill=False, ec='#58616b', lw=1.3))
        # Wedge expects degrees; all saved and displayed phase values stay in radians.
        ax.add_patch(Wedge((0, 0), 1, np.degrees(theta20), np.degrees(theta5),
                           facecolor=sector_color, alpha=.20, edgecolor='none'))
        for theta, color in zip((theta5, theta20), colors):
            point = (np.cos(theta), np.sin(theta))
            ax.annotate('', xy=point, xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-|>', lw=2.4, color=color,
                                        mutation_scale=16, shrinkA=0, shrinkB=0))
        arc = np.linspace(theta20, theta5, 100)
        ax.plot(.62*np.cos(arc), .62*np.sin(arc), color=sector_color, lw=1.6)
        ax.annotate('', xy=(.62*np.cos(arc[-1]), .62*np.sin(arc[-1])),
                    xytext=(.62*np.cos(arc[-8]), .62*np.sin(arc[-8])),
                    arrowprops=dict(arrowstyle='-|>', color=sector_color, lw=1.6,
                                    mutation_scale=13, shrinkA=0, shrinkB=0))
        middle = (theta5+theta20)/2
        ax.text(.36*np.cos(middle), .36*np.sin(middle), r'$\Delta\phi$',
                color=sector_color, fontsize=15, ha='center', va='center')
        ax.text(.5, -.075, rf'$\arg F_5 = {theta5:.6f}$ rad',
                transform=ax.transAxes, ha='center', color=colors[0], fontsize=13)
        ax.text(.5, -.145, rf'$\arg F_{{20}} = {theta20:.6f}$ rad',
                transform=ax.transAxes, ha='center', color=colors[1], fontsize=13)
        ax.text(.5, -.23, rf'$\Delta\phi_{{\mathrm{{{label.lower()}}}}} = {phases[k]:.6f}$ rad',
                transform=ax.transAxes, ha='center', fontsize=14)
    change = float(saved.loc['Q', 'circular_increment_rad'])
    fig.text(.5, .105, f'Change in relative phase: {change:+.6f} rad',
             ha='center', fontsize=16, weight='bold')
    fig.text(.5, .045,
             'Unit circles show arguments only; actual Fourier magnitudes are not represented.',
             ha='center', fontsize=12, color='#58616b')
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        for extension in ('png', 'svg'):
            fig.savefig(out/f'phase_jump_unit_circles.{extension}', dpi=180,
                        bbox_inches='tight', pad_inches=.2,
                        metadata={'Description': 'Before/After unit-circle illustration from saved Q1 angles.'})
        svg = out/'phase_jump_unit_circles.svg'
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    return fig


GRID_START, GRID_END = pd.Timestamp('2025-05-01'), pd.Timestamp('2026-10-01')
SUBSURFACE_IDS = (996, 1012, 1035)
RESOLUTION_OUT = OUT/'grid_resolution'
CANDIDATE_PASSPORT = ROOT/'Project_description/Research_log/output/stage02_candidate_selection/passport.json'


def load_resolution_inputs():
    """Verify the accepted 0.2 passport and read its frozen candidate points.

    README/PLAN are historical reference hashes: verify their original Git
    versions explicitly and record their later revisions. Data mismatches fail.
    No cleaning, monthly fitting, old 0.4 masks or Oracle access is performed.
    """
    passport = json.loads(CANDIDATE_PASSPORT.read_text())
    if (passport['selected_vertical'] != 'W_x1_y4'
            or pd.Timestamp(passport['selected_start_inclusive']) != GRID_START
            or pd.Timestamp(passport['selected_end_exclusive']) != GRID_END
            or passport['selector'] != 'technically_retained & ~doubtful'):
        raise ValueError('Unexpected accepted Stage 0.2 input')
    expected = {str(CANDIDATE_PASSPORT.relative_to(ROOT)): digest(CANDIDATE_PASSPORT)}
    for key in ('input_sha256', 'local_derived_sha256', 'implementation_sha256'):
        expected.update(passport[key])
    current, reference_updates = {}, {}
    for rel, sha in expected.items():
        actual = digest(ROOT/rel)  # missing files fail; no fallback snapshot
        if actual != sha:
            if rel not in ('Project_description/Research_log/README.md', 'Project_description/Research_log/PLAN.md'):
                raise ValueError(f'Accepted input checksum mismatch: {rel}')
            import hashlib
            original = subprocess.check_output(['git', 'show', passport['source_commit']+':'+rel], cwd=ROOT)
            if hashlib.sha256(original).hexdigest() != sha:
                raise ValueError(f'Historical reference does not match its source commit: {rel}')
            reference_updates[rel] = dict(passport_sha256=sha, current_sha256=actual,
                                          historical_commit=passport['source_commit'])
        current[rel] = actual
    local = ROOT/'Project_description/Research_log/data/stage02_candidate_selection'
    strict_local = ROOT/'Project_description/Research_log/data/stage02_final_admission'
    from stage02_candidate_selection import candidate_points
    frames, used_hashes, point_checks = {}, {}, []
    for sid in IDS:
        path = local/f'{sid}_candidate_measurements.parquet'
        frame = pd.read_parquet(path)
        if (frame.doubtful.any() or not frame.technically_retained.all()
                or not frame.candidate_month_status.eq('pass').all()
                or frame.localdatetime.duplicated().any()
                or not frame.localdatetime.is_monotonic_increasing
                or not np.isfinite(frame.datavalue).all()):
            raise ValueError(f'Invalid accepted points: {sid}')
        strict_path = strict_local/f'{sid}_measurements.parquet'
        selected = candidate_points(pd.read_parquet(strict_path), GRID_START, GRID_END)
        cols = ['localdatetime', 'datavalue', 'sensorid', 'variableid', 'table', 'sensorcode', 'plateau_review']
        pd.testing.assert_frame_equal(frame[cols].reset_index(drop=True), selected[cols].reset_index(drop=True),
                                      check_exact=True)
        frames[sid] = frame
        for p in (path, strict_path):
            used_hashes[str(p.relative_to(ROOT))] = current[str(p.relative_to(ROOT))]
        point_checks.append(dict(sensorid=sid, points=len(frame), exact_time_value_and_metadata_match=True,
            fine_cells_with_multiple_points=int(frame.localdatetime.dt.floor('15min').duplicated().sum()),
            plateau_review_points=int(frame.plateau_review.sum())))
    return frames, passport, current, used_hashes, reference_updates, point_checks


def resolution_grid(frames, bin_minutes, start=GRID_START, end=GRID_END):
    """Calendar-cell means of the same accepted raw points, at either width.

    Coarse means are not means of fine-cell means, nor restricted by fine-cell
    commonality. Source timestamps stay in the input; Fourier times are real
    calendar-cell centers, measured from the unchanged Stage 0.4 origin START.
    """
    if bin_minutes not in (15, 120):
        raise ValueError('Only the authorized 15-minute and 2-hour grids are supported')
    index = pd.date_range(start, end, freq=f'{bin_minutes}min', inclusive='left', name='bin_start')
    grid = pd.DataFrame(index=index)
    grid['bin_center'] = index + pd.Timedelta(minutes=bin_minutes/2)
    grid['t_days'] = (grid.bin_center - START).dt.total_seconds()/86400
    for sid, frame in frames.items():
        points = frame.loc[frame.technically_retained & ~frame.doubtful
                           & frame.localdatetime.ge(start) & frame.localdatetime.lt(end)]
        if not np.isfinite(points.datavalue).all():
            raise ValueError('Nonfinite accepted concentration')
        groups = points.groupby(points.localdatetime.dt.floor(f'{bin_minutes}min'))
        grid[f'mean_{sid}'] = groups.datavalue.mean().reindex(index)
        grid[f'n_{sid}'] = groups.size().reindex(index, fill_value=0).astype(int)
        grid[f'plateau_review_{sid}'] = groups.plateau_review.any().reindex(index, fill_value=False)
    grid['complete_subsurface'] = grid[[f'mean_{s}' for s in SUBSURFACE_IDS]].notna().all(axis=1)
    if 1275 in frames:
        grid['complete_with_air'] = grid.complete_subsurface & grid.mean_1275.notna()
    return grid


def resolution_windows(grid, bin_minutes, support='subsurface', window_days=30, step_days=1):
    """Two subsurface phases with actual common support; no quality cutoff."""
    # Reuse the established integer-day window calendar (12 bins/day), then
    # map its boundaries to the denser grid. The Fourier sum itself is shared.
    if bin_minutes not in (15, 120) or (support == 'with_air' and bin_minutes != 120):
        raise ValueError('No 15-minute atmospheric estimate is authorized')
    factor = 120//bin_minutes
    if len(grid) % factor:
        raise ValueError('Incomplete calendar extent')
    slices = [(lo*factor, hi*factor) for lo,hi in window_slices(len(grid)//factor, window_days, step_days)]
    rows = []
    for number, (lo,hi) in enumerate(slices):
        window = grid.iloc[lo:hi]
        common = window[f'complete_{support}'].to_numpy(bool)
        used = window.loc[common]
        start, end = window.index[0], window.index[-1]+pd.Timedelta(minutes=bin_minutes)
        info = dict(window=number, start=start, end_exclusive=end, center=start+(end-start)/2,
                    bin_minutes=bin_minutes, support=support, possible_bins=len(window),
                    complete_bins=len(used), missing_bins=len(window)-len(used),
                    coverage_fraction=len(used)/len(window),
                    longest_gap_bins=longest_missing_run(common),
                    longest_gap_hours=longest_missing_run(common)*bin_minutes/60,
                    air_removed_bins=int(window.complete_subsurface.sum()-len(used)) if support == 'with_air' else 0)
        coefficients = np.full(3, np.nan+1j*np.nan)
        if len(used) >= 2:
            coefficients = direct_fourier(used.t_days.to_numpy(), used[[f'mean_{s}' for s in SUBSURFACE_IDS]].to_numpy())
        for j,sid in enumerate(SUBSURFACE_IDS):
            info.update({f'F_real_{sid}': coefficients[j].real, f'F_imag_{sid}': coefficients[j].imag,
                         f'F_abs_{sid}': abs(coefficients[j]),
                         f'used_observations_{sid}': int(used[f'n_{sid}'].sum()),
                         f'plateau_review_bins_{sid}': int(used[f'plateau_review_{sid}'].sum())})
        if support == 'with_air':
            info['plateau_review_bins_1275'] = int(used.plateau_review_1275.sum())
        for j,pair in enumerate(PAIRS[1:]):
            angle = phase_difference(coefficients[j], coefficients[j+1])
            info[pair+'_rad'] = angle
            info[pair+'_reason'] = ('fewer_than_two_common_cells' if len(used) < 2 else
                                    'zero_or_nonfinite_coefficient' if not np.isfinite(angle) else '')
        rows.append(info)
    return pd.DataFrame(rows)


def compare_resolution_windows(left, right, effect):
    """Join exact calendar windows; undefined pairs remain explicit non-comparisons."""
    keys = ['window', 'start', 'end_exclusive', 'center']
    pd.testing.assert_frame_equal(left[keys], right[keys])
    result = []
    for j,pair in enumerate(PAIRS[1:]):
        columns = keys+['complete_bins','possible_bins','missing_bins','coverage_fraction',
                        'longest_gap_hours', pair+'_rad', pair+'_reason']
        for sid in SUBSURFACE_IDS[j:j+2]:
            columns += [f'F_abs_{sid}', f'used_observations_{sid}', f'plateau_review_bins_{sid}']
        joined = left[columns].merge(right[columns], on=keys, suffixes=('_left','_right'), validate='one_to_one')
        a, b = joined[pair+'_rad_left'].to_numpy(), joined[pair+'_rad_right'].to_numpy()
        joined['comparable'] = np.isfinite(a) & np.isfinite(b)
        joined['delta_rad'] = np.where(joined.comparable, wrap(np.angle(np.exp(1j*(a-b)))), np.nan)
        joined['effect'], joined['pair'] = effect, pair
        joined['left_variant'] = f'{left.bin_minutes.iloc[0]}min_{left.support.iloc[0]}'
        joined['right_variant'] = f'{right.bin_minutes.iloc[0]}min_{right.support.iloc[0]}'
        result.append(joined)
    return pd.concat(result, ignore_index=True)


def resolution_summary(comparison):
    """Descriptive spread only; overlapping windows are not independent samples."""
    rows = []
    for (effect,pair), group in comparison.groupby(['effect','pair'], sort=False):
        d = group.loc[group.comparable, 'delta_rad']
        row = dict(effect=effect, pair=pair, total_windows=len(group), comparable_windows=len(d),
                   undefined_windows=int((~group.comparable).sum()))
        row.update({name: float(value) for name,value in dict(min_rad=d.min(), max_rad=d.max(),
            median_rad=d.median(), q25_rad=d.quantile(.25), q75_rad=d.quantile(.75),
            std_rad=d.std(ddof=0), median_abs_rad=d.abs().median(), q95_abs_rad=d.abs().quantile(.95),
            max_abs_rad=d.abs().max()).items()})
        if len(d):
            row['max_abs_center'] = str(group.loc[d.abs().idxmax(), 'center'])
        rows.append(row)
    return pd.DataFrame(rows)


def resolution_extrema(windows):
    """List observed minimum modules and worst coverage, without classifying them.

    Keep ties: a minimum is a descriptive rank, never an admission threshold.
    Undefined phases are retained separately in the comparisons and counts.
    """
    rows = []
    metrics = ['coverage_fraction', 'longest_gap_hours']+[f'F_abs_{s}' for s in SUBSURFACE_IDS]
    for variant, group in windows.groupby('variant', sort=False):
        for metric in metrics:
            target = group[metric].max() if metric == 'longest_gap_hours' else group[metric].min()
            if metric in ('coverage_fraction','longest_gap_hours') and group.missing_bins.eq(0).all():
                continue  # no incomplete windows; the range table reports full coverage
            for _,row in group.loc[group[metric].eq(target)].iterrows():
                rows.append(dict(variant=variant, metric=metric, value=float(target), center=row.center,
                    complete_bins=row.complete_bins, possible_bins=row.possible_bins,
                    longest_gap_hours=row.longest_gap_hours,
                    **{f'F_abs_{s}': row[f'F_abs_{s}'] for s in SUBSURFACE_IDS},
                    **{p+'_rad': row[p+'_rad'] for p in PAIRS[1:]}))
    return pd.DataFrame(rows)


def resolution_quality_summary(windows):
    """Coverage, gaps and undefined counts; no new quality categories."""
    rows = []
    for variant, group in windows.groupby('variant', sort=False):
        rows.append(dict(variant=variant, windows=len(group),
            common_bins_min=int(group.complete_bins.min()), common_bins_max=int(group.complete_bins.max()),
            possible_bins=int(group.possible_bins.iloc[0]),
            coverage_percent_min=float(100*group.coverage_fraction.min()),
            coverage_percent_max=float(100*group.coverage_fraction.max()),
            incomplete_windows=int(group.missing_bins.gt(0).sum()),
            max_gap_hours=float(group.longest_gap_hours.max()),
            **{p+'_undefined':int(group[p+'_rad'].isna().sum()) for p in PAIRS[1:]}))
    return pd.DataFrame(rows)


def resolution_figures(comparison, windows, out):
    """Two separate effect figures, plus module/coverage context; radians only."""
    colors = ['#186b9b', '#bd6217', '#8a3c82']
    variant_colors = dict(zip(('15min_subsurface','120min_subsurface','120min_with_air'),colors))
    variant_labels = {'15min_subsurface':'15 min · subsurface', '120min_subsurface':'2 h · subsurface',
                      '120min_with_air':'2 h · + LI-COR mask'}
    captions = {}
    for effect, group in comparison.groupby('effect', sort=False):
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, layout='constrained')
        for col,pair in enumerate(PAIRS[1:]):
            data = group.loc[group.pair.eq(pair)]
            for suffix in ('left','right'):
                variant=data[suffix+'_variant'].iloc[0]
                tx,vy = broken_line(data.center, data[pair+'_rad_'+suffix])
                axes[0,col].plot(tx,vy, color=variant_colors[variant], lw=.9, label=variant_labels[variant])
            axes[0,col].set(title=pair.replace('_to_', ' → ')+' cm', ylabel='Relative phase (rad)', ylim=(-np.pi,np.pi))
            axes[0,col].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi], labels=['−π','−π/2','0','π/2','π'])
            axes[0,col].legend(fontsize=9, frameon=False)
            tx,vy = broken_line(data.center, data.delta_rad)
            axes[1,col].plot(tx,vy,color='#323b43',lw=.9)
            axes[1,col].axhline(0,color='#888888',lw=.6)
            axes[1,col].set(ylabel='Circular difference: left − right (rad)', xlabel='Window center')
            for ax in axes[:,col]:
                ax.grid(alpha=.2)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        caption = ('Subsurface 15-minute versus 2-hour estimates from the accepted Stage 0.2 observations.'
                   if effect == 'aggregation' else
                   'Additional LI-COR availability restriction on the same subsurface 2-hour means; no air phase estimated.')
        fig.suptitle(caption+'\n1 cycle/day · 30-day windows · 1-day step; no sufficiency threshold', fontsize=13)
        name='grid_'+effect
        for extension in ('png','svg'):
            fig.savefig(out/f'{name}.{extension}',dpi=180,bbox_inches='tight',pad_inches=.15)
        plt.close(fig)
        captions[name]=caption+' The lower row uses Arg(exp(i(left − right))) on [−π, π). Windows overlap.'
    fig,axes=plt.subplots(4,1,figsize=(13,11),sharex=True,layout='constrained')
    for (variant,data),color in zip(windows.groupby('variant',sort=False),colors):
        axes[0].plot(data.center,100*data.coverage_fraction,color=color,lw=.9,label=variant_labels[variant])
        for ax,sid in zip(axes[1:],SUBSURFACE_IDS):
            ax.plot(data.center,data[f'F_abs_{sid}'],color=color,lw=.9)
            minimum=data.loc[data[f'F_abs_{sid}'].eq(data[f'F_abs_{sid}'].min())]
            ax.scatter(minimum.center, minimum[f'F_abs_{sid}'], color=color, s=25, zorder=3)
            ax.set_ylabel({996:'|F₅| (ppm)',1012:'|F₂₀| (ppm)',1035:'|F₅₀| (ppm)'}[sid])
            ax.set_ylim(bottom=0)
    axes[0].set(ylabel='Common cells (%)',ylim=(0,102))
    axes[0].legend(loc='lower left',ncol=3,frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].set_xlabel('Window center')
    fig.suptitle('Coverage and coefficient magnitudes · accepted Stage 0.2 input\nDots: observed module minima; no new quality cutoff',fontsize=14)
    for extension in ('png','svg'):
        fig.savefig(out/f'grid_context.{extension}',dpi=180,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    captions['grid_context']='Actual common-cell coverage and |F| of the three subsurface channels for each variant; no new amplitude or coverage threshold.'
    for path in out.glob('grid_*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    return captions


def generate_grid_resolution(out=RESOLUTION_OUT):
    """The bounded 0.5 diagnostic only: no full Stage 0.3/0.4/Q1 calculation."""
    out = Path(out)
    frames, accepted, hashes, used_hashes, reference_updates, point_checks = load_resolution_inputs()
    protected = {}
    for folder in ('stage02_candidate_selection','stage02_final_admission','preliminary_spectra','preliminary_fourier_phases'):
        for path in (OUT.parent/folder).rglob('*'):
            if path.is_file() and not path.is_relative_to(out):
                protected[str(path.relative_to(ROOT))]=digest(path)
    fine = resolution_grid({s:frames[s] for s in SUBSURFACE_IDS},15)
    coarse = resolution_grid(frames,120)
    saved_grid_path=ROOT/'Project_description/Research_log/data/stage02_candidate_selection/calendar_2h_grid.parquet'
    saved_grid=pd.read_parquet(saved_grid_path)
    for sid in IDS:
        np.testing.assert_array_equal(coarse[f'mean_{sid}'],saved_grid[f'C_{sid}'])
        np.testing.assert_array_equal(coarse[f'n_{sid}'],saved_grid[f'n_{sid}'])
    np.testing.assert_array_equal(coarse.complete_with_air,saved_grid.complete)
    variants = {
        '15min_subsurface': resolution_windows(fine,15),
        '120min_subsurface': resolution_windows(coarse,120),
        '120min_with_air': resolution_windows(coarse,120,'with_air')}
    comparison=pd.concat([
        compare_resolution_windows(variants['15min_subsurface'],variants['120min_subsurface'],'aggregation'),
        compare_resolution_windows(variants['120min_subsurface'],variants['120min_with_air'],'air_support')],ignore_index=True)
    windows=pd.concat([data.assign(variant=name) for name,data in variants.items()],ignore_index=True)
    summary=resolution_summary(comparison)
    extrema=resolution_extrema(windows)
    quality=resolution_quality_summary(windows)
    test_log=run_tests()
    out.mkdir(parents=True,exist_ok=True)
    for name,frame in [('windows',windows),('comparisons',comparison),('summary',summary),('extrema',extrema),('quality',quality)]:
        frame.to_csv(out/f'grid_{name}.csv',index=False,float_format='%.17g')
    # Save every calendar cell and missing run, not just windows selected for display.
    gap_rows=[]
    for variant,grid,mask,minutes in [('15min_subsurface',fine,'complete_subsurface',15),
            ('120min_subsurface',coarse,'complete_subsurface',120),('120min_with_air',coarse,'complete_with_air',120)]:
        missing=grid.index[~grid[mask]]
        groups=(missing.to_series().diff()!=pd.Timedelta(minutes=minutes)).cumsum()
        for _,part in missing.to_series().groupby(groups):
            gap_rows.append(dict(variant=variant,start=part.iloc[0],
                end_exclusive=part.iloc[-1]+pd.Timedelta(minutes=minutes),missing_bins=len(part),missing_hours=len(part)*minutes/60))
    pd.DataFrame(gap_rows,columns=['variant','start','end_exclusive','missing_bins','missing_hours']).to_csv(out/'grid_gap_runs.csv',index=False)
    fine.to_csv(out/'grid_15min.csv.gz',float_format='%.17g',compression={'method':'gzip','mtime':0})
    coarse.to_csv(out/'grid_2h.csv.gz',float_format='%.17g',compression={'method':'gzip','mtime':0})
    captions=resolution_figures(comparison,windows,out)
    for rel,sha in {**hashes,**protected}.items():
        if digest(ROOT/rel)!=sha:
            raise ValueError(f'Protected input/result changed: {rel}')
    used_hashes[str(CANDIDATE_PASSPORT.relative_to(ROOT))]=digest(CANDIDATE_PASSPORT)
    used_hashes[str(saved_grid_path.relative_to(ROOT))]=digest(saved_grid_path)
    passport=dict(stage='0.5: temporal-grid diagnostic', baseline_commit='0e33a2d951e572e57636452840da0a794c01e072',
        accepted_passport=str(CANDIDATE_PASSPORT.relative_to(ROOT)),selector=accepted['selector'],
        selected_vertical=accepted['selected_vertical'],metadata=accepted['metadata'],
        start=str(GRID_START),end_exclusive=str(GRID_END),time_origin=str(START),
        f0_cycles_per_day=F0,window_days=30,step_days=1,bin_minutes=[15,120],
        bin_time='Calendar centers (+7min30s or +1h); input timestamps unchanged',
        aggregation='Arithmetic mean of all accepted source observations in each cell; equal weight per common cell in Fourier sum',
        normalization='1/N after subtraction of each channel mean over the same common cells',
        phase='wrap(Arg(F_upper * conjugate(F_lower))) in [-pi,pi)',
        differences={'aggregation':'wrap(Arg(exp(i*(phase_15min_subsurface-phase_120min_subsurface))))',
                     'air_support':'wrap(Arg(exp(i*(phase_120min_subsurface-phase_120min_with_air))))'},
        input_sha256=used_hashes,upstream_hashes_verified=len(hashes),reference_document_updates=reference_updates,
        point_checks=point_checks,implementation_sha256=digest(Path(__file__)),captions=captions,
        limitations=['Overlapping windows are dependent; descriptive spread is not measurement uncertainty',
                     'No new small-module, coverage or sufficiency threshold; extrema are ranks only',
                     '2h support can retain source observations whose 15min cell lacks another subsurface channel',
                     'LI-COR only restricts 2h support here; no 15min air reconstruction or air phase',
                     'No fast-transport or physical-mechanism inference; sufficiency requires I.R. Gabitov decision'])
    (out/'grid_passport.json').write_text(json.dumps(passport,ensure_ascii=False,indent=2)+'\n')
    validation=dict(tests_passed=True,test_log=test_log,accepted_2h_means_counts_support_exact=True,
        upstream_checksums_verified=len(hashes),upstream_inputs_unchanged=True,
        previous_results_unchanged=True,previous_result_sha256=protected,
        windows_per_variant={k:len(v) for k,v in variants.items()},
        common_calendar_cells={'15min_subsurface':int(fine.complete_subsurface.sum()),
                              '120min_subsurface':int(coarse.complete_subsurface.sum()),
                              '120min_with_air':int(coarse.complete_with_air.sum())},
        artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='grid_validation.json'})
    (out/'grid_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    return windows,comparison,summary,extrema,validation


# Stage 0.5: concentration-only diagnostic. No Fourier functions are called.
INTRABIN_OUT = OUT/'intrabin_2h'
INTRABIN_COUNTS = {996: 49311, 1012: 49293, 1035: 49267}


def intrabin_statistics(frames, start=GRID_START, end=GRID_END):
    """Own-sensor observations in [s,s+2h); no new cleaning or common mask.

    Missingness counts empty nominal 15-minute cells (not assumed exact times).
    Adjacent pairs are consecutive accepted records within one 2-hour cell,
    including pairs across missing slots, with their actual elapsed seconds.
    """
    calendar = pd.date_range(start, end, freq='2h', inclusive='left')
    rows, pairs = [], []
    for sid, frame in frames.items():
        points = frame.loc[frame.technically_retained & ~frame.doubtful
            & frame.localdatetime.ge(start) & frame.localdatetime.lt(end)].sort_values('localdatetime')
        if points.localdatetime.duplicated().any() or not np.isfinite(points.datavalue).all():
            raise ValueError('Expected unique finite accepted observations')
        groups = {k: g for k, g in points.groupby(points.localdatetime.dt.floor('2h'))}
        # Same groupby reduction as the accepted input: preserve exact means.
        means = points.groupby(points.localdatetime.dt.floor('2h')).datavalue.mean()
        for s in calendar:
            g = groups.get(s, points.iloc[:0])
            n = len(g)
            t, y = g.localdatetime, g.datavalue.to_numpy(dtype=float)
            slots = t.dt.floor('15min')
            occupied = slots.nunique()
            missing = ~pd.date_range(s, periods=8, freq='15min').isin(slots)
            row = dict(sensorid=sid, bin_start=s, bin_end_exclusive=s+pd.Timedelta(hours=2),
                n=n, first_time=t.iloc[0] if n else pd.NaT, last_time=t.iloc[-1] if n else pd.NaT,
                mean_ppm=float(means.loc[s]) if n else np.nan,
                min_ppm=float(y.min()) if n else np.nan, max_ppm=float(y.max()) if n else np.nan,
                occupied_15min_cells=occupied, missing_15min_cells=8-occupied,
                duplicate_slot_points=n-occupied, complete_15min_coverage=occupied == 8,
                longest_missing_15min_cells=longest_missing_run(~missing),
                insufficient_variation=n < 2, plateau_review_points=int(g.plateau_review.sum()),
                leading_unobserved_seconds=(t.iloc[0]-s).total_seconds() if n else np.nan,
                trailing_unobserved_seconds=(s+pd.Timedelta(hours=2)-t.iloc[-1]).total_seconds() if n else np.nan)
            for key in ('range_ppm','max_abs_deviation_ppm','last_minus_first_ppm',
                        'max_adjacent_abs_change_ppm','max_change_dt_seconds','max_change_rate_ppm_h',
                        'max_observed_gap_seconds'):
                row[key] = np.nan
            row['max_change_from'] = row['max_change_to'] = pd.NaT
            if n >= 2:
                delta = np.diff(y)
                dt = t.diff().dt.total_seconds().iloc[1:].to_numpy()
                if np.any(dt <= 0):
                    raise ValueError('Nonpositive adjacent elapsed time')
                k = int(np.argmax(np.abs(delta)))  # earliest exact tie
                row.update(range_ppm=float(np.ptp(y)), max_abs_deviation_ppm=float(np.max(np.abs(y-row['mean_ppm']))),
                    last_minus_first_ppm=float(y[-1]-y[0]), max_adjacent_abs_change_ppm=float(abs(delta[k])),
                    max_change_dt_seconds=float(dt[k]), max_change_rate_ppm_h=float(abs(delta[k])*3600/dt[k]),
                    max_observed_gap_seconds=float(dt.max()), max_change_from=t.iloc[k], max_change_to=t.iloc[k+1])
                for k in range(n-1):
                    pairs.append(dict(sensorid=sid, bin_start=s, from_time=t.iloc[k], to_time=t.iloc[k+1],
                        from_ppm=y[k], to_ppm=y[k+1], signed_change_ppm=delta[k], abs_change_ppm=abs(delta[k]),
                        dt_seconds=dt[k], observed_rate_ppm_h=abs(delta[k])*3600/dt[k],
                        missing_15min_cells_between=max(0, int((slots.iloc[k+1]-slots.iloc[k]).total_seconds()/900)-1)))
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pairs, columns=['sensorid','bin_start','from_time','to_time',
        'from_ppm','to_ppm','signed_change_ppm','abs_change_ppm','dt_seconds','observed_rate_ppm_h','missing_15min_cells_between'])


def intrabin_summary(cells, pairs):
    """Per-sensor descriptive linear quantiles; no independence/CI claim."""
    rows, extrema = [], []
    for sid, g in cells.groupby('sensorid', sort=False):
        a = pairs.loc[pairs.sensorid.eq(sid)]
        row = dict(sensorid=int(sid), bins=len(g), points=int(g.n.sum()),
            full_bins=int(g.complete_15min_coverage.sum()), incomplete_bins=int((~g.complete_15min_coverage).sum()),
            insufficient_bins=int(g.insufficient_variation.sum()), adjacent_pairs=len(a))
        for data, metrics in [(g, ['range_ppm','max_abs_deviation_ppm']),
                              (a, ['abs_change_ppm','dt_seconds','observed_rate_ppm_h'])]:
            for metric in metrics:
                for name, value in [('min',data[metric].min()),('median',data[metric].median()),
                                    ('p95',data[metric].quantile(.95)),('max',data[metric].max())]:
                    row[f'{metric}_{name}'] = float(value)
                if data[metric].notna().any():
                    r = data.loc[data[metric].idxmax()]
                    extrema.append(dict(sensorid=int(sid), metric=metric, maximum=float(r[metric]),
                        bin_start=r.bin_start, bin_end_exclusive=r.bin_start+pd.Timedelta(hours=2),
                        from_time=r.get('from_time',r.get('first_time')), to_time=r.get('to_time',r.get('last_time'))))
        rows.append(row)
    counts = cells.groupby(['sensorid','n']).size().rename('bins').reset_index()
    return pd.DataFrame(rows), counts, pd.DataFrame(extrema)


def select_intrabin_examples(cells):
    """At most eight unique times, earliest exact ties, retain all reasons."""
    selected = {}
    def add(row, reason):
        selected.setdefault(row.bin_start, []).append(f'GMM222 {int(row.sensorid)}: {reason}')
    ordered = cells.sort_values(['bin_start','sensorid'])
    for sid, g in ordered.groupby('sensorid', sort=False):
        valid = g.loc[g.range_ppm.notna()]
        add(valid.loc[valid.range_ppm.idxmax()], 'maximum within-bin range; earliest tied bin')
        distance = (valid.range_ppm-valid.range_ppm.median()).abs()
        add(valid.loc[distance.idxmin()], 'range closest to sensor median; earliest tied bin')
    add(ordered.loc[ordered.max_adjacent_abs_change_ppm.idxmax()],
        'maximum absolute adjacent change across all three sensors; earliest bin, then lowest sensor ID')
    incomplete = ordered.loc[~ordered.complete_15min_coverage]
    if len(incomplete):
        add(incomplete.iloc[0], 'earliest incomplete bin across three sensors; lowest sensor ID tie')
    return pd.DataFrame([dict(example=f'example_{i:02d}', bin_start=s,
        bin_end_exclusive=s+pd.Timedelta(hours=2), selection_rules='; '.join(reasons))
        for i,(s,reasons) in enumerate(sorted(selected.items()),1)])


def intrabin_metadata(accepted):
    """Check archived Oracle geometry and the primary workbook, read-only."""
    import openpyxl
    path = ARCHIVE/'manifest.json'
    manifest = json.loads(path.read_text())
    workbook = ROOT/'Sensors_Description/variables_schema.xlsx'
    wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
    sheet = wb['CO2']
    metadata = []
    try:
        for sid in SUBSURFACE_IDS:
            c = next(c for c in manifest['channels'] if c['sensorid'] == sid)
            assert c == next(c for c in accepted['metadata'] if c['sensorid'] == sid)
            values = [sheet.cell(c['row'], j).value for j in (3,4,5,11,12,13,18)]
            assert values == [c['x'],c['y'],c['z'],'GMM222',sid,c['sensorcode'],'ppm']
            loc = c['oracle_location']
            assert (float(loc['boxx']),float(loc['boxy']),-float(loc['boxz'])) == (c['x'],c['y'],c['z'])
            metadata.append(c)
    finally:
        wb.close()
    return metadata, manifest['timezone'], {str(p.relative_to(ROOT)):digest(p) for p in (path,workbook)}


def intrabin_figures(cells, pairs, examples, frames, metadata, out):
    """English, spatially explicit concentration plots; do not bridge gaps."""
    captions = {}
    def location(sid):
        m = next(m for m in metadata if m['sensorid'] == sid)
        return f"LEO West W_x1_y4, x={m['x']:g} m, y={m['y']:g} m, depth {-100*m['z']:g} cm, GMM222 {sid}"
    def save(fig, name, caption):
        for ext in ('png','svg'):
            fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight',pad_inches=.18,
                        metadata={'Description':caption})
        plt.close(fig)
        captions[name] = caption
    colors = ['#17658a','#b05c16','#7954a0']
    fig, axes = plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for row,(sid,color) in enumerate(zip(SUBSURFACE_IDS,colors)):
        g = cells.loc[cells.sensorid.eq(sid)]
        for ax,metric,label in zip(axes[row],['range_ppm','max_abs_deviation_ppm'],
                ['Within-bin CO₂ range (ppm)','Maximum absolute deviation from 2 h mean (ppm)']):
            values = np.sort(g[metric].dropna())
            ax.step(values, np.arange(1,len(values)+1)/len(values),where='post',color=color,lw=1.6)
            ax.set(xlabel=label,ylabel='Empirical cumulative fraction',ylim=(0,1.02),xlim=(0,None))
            ax.set_title(location(sid).replace(', depth','\ndepth'),fontsize=10)
            ax.grid(alpha=.2)
    fig.suptitle('Observed CO₂ variability within calendar 2 h bins\n01.May.2025–30.Sep.2026 · accepted observations only',fontsize=15)
    save(fig,'intrabin_2h_distributions','I show per-sensor empirical distributions of observed within-bin CO₂ range and maximum absolute deviation from the own-sensor mean. Quantiles are descriptive, not confidence intervals.')
    fig, axes = plt.subplots(3,2,figsize=(13,11),layout='constrained')
    for row,(sid,color) in enumerate(zip(SUBSURFACE_IDS,colors)):
        a = pairs.loc[pairs.sensorid.eq(sid)]
        for ax,metric,label in zip(axes[row],['abs_change_ppm','dt_seconds'],
                ['Absolute adjacent CO₂ change (ppm)','Actual time between adjacent records (s)']):
            values=np.sort(a[metric].dropna())
            ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,lw=1.6)
            ax.set(xlabel=label,ylabel='Empirical cumulative fraction',ylim=(0,1.02))
            ax.set_title(location(sid).replace(', depth','\ndepth'),fontsize=10)
            ax.grid(alpha=.2)
    fig.suptitle('All adjacent accepted pairs within the same 2 h bin\nPairs across missing 15 min cells retain their actual elapsed time',fontsize=15)
    save(fig,'intrabin_2h_adjacent_distributions','I include every adjacent accepted pair within a 2 h bin, including pairs spanning missing cells; I retain the actual elapsed time. Changes are not measurements of physical process speed.')
    for e in examples.itertuples():
        fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True,layout='constrained')
        for ax,sid,color in zip(axes,SUBSURFACE_IDS,colors):
            g=frames[sid].loc[frames[sid].localdatetime.ge(e.bin_start)&frames[sid].localdatetime.lt(e.bin_end_exclusive)]
            r=cells.loc[cells.sensorid.eq(sid)&cells.bin_start.eq(e.bin_start)].iloc[0]
            slots=g.localdatetime.dt.floor('15min')
            for i in range(len(g)-1):
                if slots.iloc[i+1]-slots.iloc[i] == pd.Timedelta(minutes=15):
                    ax.plot(g.localdatetime.iloc[i:i+2],g.datavalue.iloc[i:i+2],color=color,lw=1)
            ax.scatter(g.localdatetime,g.datavalue,color=color,s=30,zorder=3,label='Accepted observations')
            ax.axhline(r.mean_ppm,color='#4d4d4d',ls='--',lw=1.3,label='Own-sensor 2 h mean')
            for slot in pd.date_range(e.bin_start,periods=8,freq='15min'):
                if slot not in set(slots): ax.axvspan(slot,slot+pd.Timedelta(minutes=15),color='#dddddd',alpha=.5)
            ax.set_title('CO₂ concentration (ppm) — '+location(sid)+'\n'+
                f'n={r.n}; occupied cells={r.occupied_15min_cells}/8; min={r.min_ppm:.3f}; max={r.max_ppm:.3f}; mean={r.mean_ppm:.3f} ppm',fontsize=10,loc='left')
            ax.set_ylabel('CO₂ concentration (ppm)')
            ax.ticklabel_format(axis='y',useOffset=False)
            ax.grid(alpha=.2)
        axes[0].legend(frameon=False,ncol=2,loc='best',fontsize=9)
        axes[-1].set_xlim(e.bin_start,e.bin_end_exclusive)
        axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].set_xlabel('Database LOCALDATETIME (timezone unconfirmed); grey = empty 15 min cell')
        fig.suptitle(f'{e.example}: [{e.bin_start:%d.%b.%Y %H:%M}, {e.bin_end_exclusive:%d.%b.%Y %H:%M})',fontsize=14)
        save(fig,e.example,'I compare accepted CO₂ observations with each sensor’s own 2 h mean. Lines join only adjacent occupied 15 min cells; grey spans show missing cells. Selection: '+e.selection_rules)
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    return captions


def validate_intrabin_real_examples(cells, examples, frames):
    """Independent scalar arithmetic on selected real bins, no Fourier calls."""
    import math
    checks=[]
    for e in examples.itertuples():
        for sid in SUBSURFACE_IDS:
            g=frames[sid].loc[frames[sid].localdatetime.ge(e.bin_start)&frames[sid].localdatetime.lt(e.bin_end_exclusive)]
            y=g.datavalue.tolist(); t=g.localdatetime.tolist()
            r=cells.loc[cells.sensorid.eq(sid)&cells.bin_start.eq(e.bin_start)].iloc[0]
            assert len(y)==r.n
            if len(y)<2: continue
            mean=math.fsum(y)/len(y)
            changes=[abs(b-a) for a,b in zip(y,y[1:])]
            k=changes.index(max(changes)); seconds=(t[k+1]-t[k]).total_seconds()
            expected=dict(mean_ppm=mean,min_ppm=min(y),max_ppm=max(y),range_ppm=max(y)-min(y),
                max_abs_deviation_ppm=max(abs(v-mean) for v in y),last_minus_first_ppm=y[-1]-y[0],
                max_adjacent_abs_change_ppm=changes[k],max_change_dt_seconds=seconds,
                max_change_rate_ppm_h=changes[k]*3600/seconds)
            for key,value in expected.items():
                assert math.isclose(r[key],value,rel_tol=0,abs_tol=2e-12),(sid,e.bin_start,key)
            assert (r.max_change_from,r.max_change_to)==(t[k],t[k+1])
            checks.append(dict(sensorid=sid,bin_start=str(e.bin_start),n=len(y),passed=True))
    return checks


def generate_intrabin_2h(out=INTRABIN_OUT):
    """Bounded raw-concentration analysis from verified accepted Stage 0.2."""
    out=Path(out)
    frames,accepted,hashes,used,updates,point_checks=load_resolution_inputs()
    frames={sid:frames[sid] for sid in SUBSURFACE_IDS}
    if {sid:len(f) for sid,f in frames.items()} != INTRABIN_COUNTS:
        raise ValueError('Accepted counts differ; stop')
    metadata,time_note,meta_hashes=intrabin_metadata(accepted)
    protected={str(p.relative_to(ROOT)):digest(p) for folder in
        ('stage02_candidate_selection','stage02_final_admission','preliminary_spectra','preliminary_fourier_phases')
        for p in (OUT.parent/folder).rglob('*') if p.is_file() and not p.is_relative_to(out)}
    cells,pairs=intrabin_statistics(frames)
    grid_path=ROOT/'Project_description/Research_log/data/stage02_candidate_selection/calendar_2h_grid.parquet'
    grid=pd.read_parquet(grid_path)
    for sid in SUBSURFACE_IDS:
        g=cells.loc[cells.sensorid.eq(sid)]
        np.testing.assert_array_equal(g.mean_ppm,grid[f'C_{sid}'])
        np.testing.assert_array_equal(g.n,grid[f'n_{sid}'])
        np.testing.assert_array_equal(g.bin_start,grid.index)
        np.testing.assert_array_equal(g.bin_end_exclusive,grid.index+pd.Timedelta(hours=2))
    summary,counts,extrema=intrabin_summary(cells,pairs)
    examples=select_intrabin_examples(cells)
    independent=validate_intrabin_real_examples(cells,examples,frames)
    tests=subprocess.run([sys.executable,'-m','unittest',
        'Project_description.sensorDB.tests.test_preliminary_fourier_phases.IntrabinTests','-v'],
        cwd=ROOT,text=True,capture_output=True,check=True)
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('all',cells),('summary',summary),('counts',counts),('extrema',extrema),('examples',examples)]:
        data.to_csv(out/f'intrabin_2h_{name}.csv',index=False,float_format='%.17g')
    pairs.to_csv(out/'intrabin_2h_adjacent.csv.gz',index=False,float_format='%.17g',compression={'method':'gzip','mtime':0})
    selected=pd.concat([f.loc[f.localdatetime.dt.floor('2h').isin(examples.bin_start),
        ['localdatetime','datavalue','sensorid','plateau_review']] for f in frames.values()])
    selected.to_csv(out/'intrabin_2h_example_points.csv',index=False,float_format='%.17g')
    captions=intrabin_figures(cells,pairs,examples,frames,metadata,out)
    for rel,sha in {**hashes,**meta_hashes,**protected}.items():
        if digest(ROOT/rel)!=sha: raise ValueError(f'Protected file changed: {rel}')
    passport=dict(stage='0.5: observed intrabin CO2 concentration',baseline_commit='f22ea460a7db6db1dc38b72233d6601bb2624605',
        accepted_passport=str(CANDIDATE_PASSPORT.relative_to(ROOT)),accepted_passport_sha256=digest(CANDIDATE_PASSPORT),
        selector=accepted['selector'],metadata=metadata,start=str(GRID_START),end_exclusive=str(GRID_END),
        time_basis=time_note,time_conversion=None,bin_minutes=120,nominal_slot_minutes=15,
        full_bin_definition='All eight calendar 15min cells occupied; descriptive coverage, not an admission rule',
        adjacent_pairs='Consecutive accepted observations inside the same 2h bin, including gaps; no cross-bin pairs',
        quantiles='pandas linear empirical quantiles, own-sensor defined cells/pairs only; no confidence intervals',
        insufficient='n<2: all variation measures NaN; empty means NaN, never zero fill',
        point_checks=[c for c in point_checks if c['sensorid'] in SUBSURFACE_IDS],input_sha256={**used,**meta_hashes,
            str(grid_path.relative_to(ROOT)):digest(grid_path)},upstream_hashes_verified=len(hashes),
        reference_document_updates=updates,implementation_sha256=digest(Path(__file__)),captions=captions,
        versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__),
        instrument_source=dict(url='https://www.vaisala.com/sites/default/files/documents/GMM220_Datasheet-B210856EN-E.pdf',
            document='Vaisala B210856EN-E (2013), page 2',accessed='2026-11-01',
            accuracy='GMM222 at 25 C, 1013 hPa: +/-(1.5% full range + 2% reading)',
            analog_output_resolution_bits=12,response_63_percent_seconds=30,
            installed_range_and_logger_resolution='Not established by the checked archive metadata or workbook'),
        limitations=['Observed variability is not proof of physical significance or transport time',
            'A 2h mean can retain an excursion contribution but loses its internal shape',
            'Missing accepted points and unobserved sub-15min behavior remain unknown',
            'No Fourier calculation, new quality threshold, interpolation, air mask or physical model'])
    (out/'intrabin_2h_passport.json').write_text(json.dumps(passport,ensure_ascii=False,indent=2)+'\n')
    validation=dict(tests_passed=True,test_log=tests.stderr+tests.stdout,accepted_counts=INTRABIN_COUNTS,
        accepted_2h_means_counts_boundaries_exact=True,independent_scalar_checks=independent,
        independent_absolute_tolerance_ppm=2e-12,upstream_checksums_verified=len(hashes),
        upstream_inputs_unchanged=True,previous_results_unchanged=True,previous_result_sha256=protected,
        cells=len(cells),adjacent_pairs=len(pairs),unique_examples=len(examples),fourier_calculation_performed=False,
        artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='intrabin_2h_validation.json'})
    (out/'intrabin_2h_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    return cells,summary,examples,validation


def load_intrabin_2h(out=INTRABIN_OUT):
    """Published tables for notebook display, without regenerating results."""
    out=Path(out)
    return tuple(pd.read_csv(out/f'intrabin_2h_{name}.csv',float_precision='round_trip',parse_dates=dates)
        for name,dates in [('summary',[]),('counts',[]),('extrema',['bin_start','bin_end_exclusive','from_time','to_time']),
                           ('examples',['bin_start','bin_end_exclusive'])])


# Stage 0.5, 02.Nov.2026: matched observations, 14-day windows.
TEMPORAL_OUT = OUT.parent/'temporal_resolution_15min_vs_2h'


def concentration_residuals(frames, start=GRID_START, end=GRID_END):
    """Own-sensor means mapped only to actual accepted source timestamps."""
    grid=resolution_grid(frames,120,start,end)
    observations=[]
    for sid,frame in frames.items():
        keep=frame.technically_retained & ~frame.doubtful & frame.localdatetime.ge(start) & frame.localdatetime.lt(end)
        r=frame.loc[keep,['localdatetime','datavalue']].copy()
        r['sensorid']=sid
        r['bin_start']=r.localdatetime.dt.floor('2h')
        r['mean_2h_ppm']=r.bin_start.map(grid[f'mean_{sid}'])
        r['residual_ppm']=r.datavalue-r.mean_2h_ppm
        observations.append(r)
    points=pd.concat(observations,ignore_index=True)
    rows=[]
    for sid,g in points.groupby('sensorid',sort=False):
        e=g.residual_ppm.to_numpy()
        rows.append(dict(sensorid=sid,N=len(g),MAE_ppm=float(np.mean(np.abs(e))),
            RMSE_ppm=float(np.sqrt(np.mean(e*e))),p95_abs_ppm=float(np.quantile(np.abs(e),.95)),
            max_abs_ppm=float(np.max(np.abs(e))),mean_residual_ppm=float(e.mean())))
    bins=points.groupby(['sensorid','bin_start'],sort=True).agg(n=('residual_ppm','size'),
        sum_residual_ppm=('residual_ppm','sum'),max_abs_error_ppm=('residual_ppm',lambda x: x.abs().max()),
        RMSE_ppm=('residual_ppm',lambda x: np.sqrt(np.mean(x*x)))).reset_index()
    return points,pd.DataFrame(rows),bins,grid


def matched_resolution_grids(frames, start=GRID_START, end=GRID_END):
    """Fix the intersection of occupied 15min cells BEFORE either estimate.

    Require exactly one accepted record per sensor per occupied fine cell.
    Keep actual timestamps; Fourier coordinates remain calendar centers, as
    in the established method. Coarse means use only those same records.
    """
    fine_all=resolution_grid(frames,15,start,end)
    common=fine_all.index[fine_all.complete_subsurface]
    matched={}; cohort=pd.DataFrame(index=common)
    for sid in SUBSURFACE_IDS:
        f=frames[sid]
        keep=f.technically_retained & ~f.doubtful & f.localdatetime.ge(start) & f.localdatetime.lt(end)
        f=f.loc[keep].copy()
        f['fine_start']=f.localdatetime.dt.floor('15min')
        if f.fine_start.duplicated().any():
            raise ValueError('Matched comparison requires one real observation per occupied 15min cell')
        matched[sid]=f.loc[f.fine_start.isin(common)].sort_values('localdatetime')
        indexed=matched[sid].set_index('fine_start').reindex(common)
        cohort[f'time_{sid}']=indexed.localdatetime
        cohort[f'C_{sid}']=indexed.datavalue
    fine=resolution_grid(matched,15,start,end)
    coarse=resolution_grid(matched,120,start,end)
    for sid in SUBSURFACE_IDS:
        assert len(matched[sid])==len(common)==int(coarse[f'n_{sid}'].sum())
        pd.testing.assert_series_equal(fine.loc[common,f'mean_{sid}'],cohort[f'C_{sid}'],check_names=False)
    cohort.index.name='fine_start'
    return fine,coarse,cohort


def temporal_phase_summary(comparison):
    rows=[]; diagnostics=[]
    for pair,g in comparison.groupby('pair',sort=False):
        valid=g.loc[g.comparable]
        d=valid.delta_rad.abs()
        row=dict(pair=pair,total_windows=len(g),comparable_windows=len(valid),undefined_windows=len(g)-len(valid),
            MAE_rad=float(d.mean()),median_abs_rad=float(d.median()),p95_abs_rad=float(d.quantile(.95)),
            max_abs_rad=float(d.max()),MAE_over_pi=float(d.mean()/np.pi))
        if len(valid):
            r=valid.loc[d.idxmax()]
            row.update(max_start=r.start,max_end_exclusive=r.end_exclusive,max_center=r.center)
            diagnostics.append(r.to_dict())
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(diagnostics)


def temporal_figures(points,metrics,bins,comparison,metadata,out):
    """Exactly three print-friendly JPG illustrations; no PNG/SVG copies."""
    depths={m['sensorid']:-100*m['z'] for m in metadata}
    def location(sid):
        m=next(m for m in metadata if m['sensorid']==sid)
        return f"W_x1_y4, x={m['x']:g} m, y={m['y']:g} m, depth {depths[sid]:g} cm, GMM222 {sid}"
    focus=bins.loc[bins.sensorid.eq(996)].sort_values('bin_start')
    typical=focus.loc[(focus.RMSE_ppm-focus.RMSE_ppm.median()).abs().idxmin()]
    largest=focus.loc[focus.max_abs_error_ppm.idxmax()]
    examples=pd.DataFrame([dict(kind=label,bin_start=r.bin_start,bin_end_exclusive=r.bin_start+pd.Timedelta(hours=2),
        selector_sensorid=996,rule=rule) for label,r,rule in [
            ('Typical',typical,'5 cm: bin RMSE closest to its median; earliest exact tie'),
            ('Largest deviation',largest,'5 cm: maximum absolute deviation from own 2h mean; earliest exact tie')]])
    colors=['#17658a','#b05c16','#7954a0'];captions={}
    def save(fig,name,caption):
        fig.savefig(out/(name+'.jpg'),dpi=300,bbox_inches='tight',facecolor='white',pil_kwargs={'quality':95})
        plt.close(fig);captions[name]=caption
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for col,e in enumerate(examples.itertuples()):
        for row,(sid,color) in enumerate(zip(SUBSURFACE_IDS,colors)):
            ax=axes[row,col];g=points.loc[points.sensorid.eq(sid)&points.bin_start.eq(e.bin_start)]
            mean=g.mean_2h_ppm.iloc[0]
            ax.hlines(mean,e.bin_start,e.bin_end_exclusive,color='#333',ls='--',label='2 h mean')
            ax.vlines(g.localdatetime,np.minimum(g.datavalue,mean),np.maximum(g.datavalue,mean),color=color,alpha=.5,lw=1.3,label='Residual e(t)')
            ax.scatter(g.localdatetime,g.datavalue,color=color,s=23,zorder=3,label='Accepted observations')
            ax.set_title(location(sid).replace(', depth','\ndepth')+f' · n={len(g)}',fontsize=10)
            ax.set(ylabel='CO₂ concentration (ppm)',xlim=(e.bin_start,e.bin_end_exclusive))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,30]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.set_xlabel(f'{e.kind}: {e.bin_start:%d.%b.%Y} [{e.bin_start:%H:%M}, {e.bin_end_exclusive:%H:%M})')
            ax.grid(alpha=.2);ax.ticklabel_format(axis='y',useOffset=False)
            if row==0:ax.legend(fontsize=8,ncol=3,loc='best')
    fig.suptitle('Measured CO₂ and its own-sensor 2 h mean at the same source timestamps\nDatabase LOCALDATETIME; timezone unconfirmed · vertical segments show residuals',fontsize=14)
    save(fig,'concentration_examples','I compare real observations with their own-sensor two-hour mean. Vertical segments show residuals, not imputed observations. The two intervals are selected from the 5 cm sensor by fixed ranks, and all three depths are shown.')
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    x=np.arange(3);a=metrics.set_index('sensorid').loc[list(SUBSURFACE_IDS)]
    ax.bar(x-.19,a.MAE_ppm,.36,color='#17658a',label='MAE')
    ax.bar(x+.19,a.RMSE_ppm,.36,color='#b05c16',label='RMSE')
    for i,r in enumerate(a.itertuples()):
        ax.text(i-.19,r.MAE_ppm+.25,f'{r.MAE_ppm:.3f}',ha='center')
        ax.text(i+.19,r.RMSE_ppm+.25,f'{r.RMSE_ppm:.3f}',ha='center')
    ax.set_xticks(x,[f'Depth {depths[s]:g} cm\nGMM222 {s}' for s in SUBSURFACE_IDS])
    ax.set(ylabel='CO₂ concentration discrepancy (ppm)',ylim=(0,float(a.RMSE_ppm.max())*1.2),
        title='Effect of replacing each observed concentration by its own 2 h mean\nLEO West W_x1_y4 · x=1 m, y=4 m · 01.May.2025–30.Sep.2026')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,'concentration_errors','I report MAE and RMSE over actual accepted observations separately at each depth. These are aggregation discrepancies, not errors relative to an unknown true concentration.')
    fig,axes=plt.subplots(2,2,figsize=(14,9),sharex=True,layout='constrained')
    for col,pair in enumerate(PAIRS[1:]):
        g=comparison.loc[comparison.pair.eq(pair)]
        for suffix,color,label in [('left','#17658a','15 min'),('right','#b05c16','2 h')]:
            tx,y=broken_line(g.center,g[pair+'_rad_'+suffix]);axes[0,col].plot(tx,y,color=color,lw=1,label=label)
        tx,y=broken_line(g.center,g.delta_rad)
        axes[1,col].plot(tx,y,color='#303c45',lw=1)
        axes[1,col].axhline(0,color='#888',lw=.7)
        j=col;u,v=SUBSURFACE_IDS[j:j+2]
        title=f'W_x1_y4, x=1 m, y=4 m\nDepths {depths[u]:g} → {depths[v]:g} cm · GMM222 {u} → {v}'
        axes[0,col].set(title=title,ylabel='Relative Fourier phase of CO₂ (rad)',ylim=(-np.pi,np.pi))
        axes[0,col].set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi],labels=['−π','−π/2','0','π/2','π'])
        axes[0,col].legend(frameon=False,ncol=2)
        axes[1,col].set(title=title,ylabel='Circular phase difference: 15 min − 2 h (rad)',xlabel='14-day window center (LOCALDATETIME)')
        for ax in axes[:,col]:
            ax.grid(alpha=.2);ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b.%Y'))
    fig.suptitle('CO₂ Fourier phases: identical source observations in both representations\n1 cycle/day · 14-day windows · 1-day step · no atmospheric availability mask',fontsize=14)
    save(fig,'phase_comparison_14d','I compare two representations of the same accepted subsurface observations. The lower panels show the circular difference; gaps at branch crossings are display breaks. Aggregation changes values, calendar-center times and weights.')
    return examples,captions


def generate_temporal_resolution(out=TEMPORAL_OUT):
    """Only the newly authorized 14-day matched-support comparison."""
    out=Path(out)
    frames,accepted,hashes,used,updates,checks=load_resolution_inputs()
    frames={s:frames[s] for s in SUBSURFACE_IDS}
    assert {s:len(f) for s,f in frames.items()}==INTRABIN_COUNTS
    metadata,time_note,meta_hashes=intrabin_metadata(accepted)
    points,concentration,bins,own=concentration_residuals(frames)
    historical_path=INTRABIN_OUT/'intrabin_2h_all.csv'
    historical_sha=digest(historical_path)
    historical=pd.read_csv(historical_path,float_precision='round_trip',parse_dates=['bin_start'])
    for sid in SUBSURFACE_IDS:
        h=historical.loc[historical.sensorid.eq(sid)].set_index('bin_start')
        np.testing.assert_array_equal(own[f'mean_{sid}'],h.mean_ppm)
        np.testing.assert_array_equal(own[f'n_{sid}'],h.n)
    zero_bound=32*np.finfo(float).eps*points.datavalue.abs().max()*bins.n.max()
    assert bins.sum_residual_ppm.abs().max() <= zero_bound
    fine,coarse,cohort=matched_resolution_grids(frames)
    wf=resolution_windows(fine,15,window_days=14,step_days=1)
    wc=resolution_windows(coarse,120,window_days=14,step_days=1)
    for sid in SUBSURFACE_IDS:
        np.testing.assert_array_equal(wf[f'used_observations_{sid}'],wc[f'used_observations_{sid}'])
    comparison=compare_resolution_windows(wf,wc,'matched_observations_14d')
    summary,diagnostics=temporal_phase_summary(comparison)
    assert len(wf)==len(wc)==505
    assert (wf.end_exclusive-wf.start).eq(pd.Timedelta(days=14)).all()
    assert wf.start.diff().dropna().eq(pd.Timedelta(days=1)).all()
    # Independently evaluate selected real coefficients with Python scalar sums.
    import cmath,math
    independent=[]
    for r in diagnostics.itertuples():
        for minutes,grid,windows in [(15,fine,wf),(120,coarse,wc)]:
            g=grid.loc[(grid.index>=r.start)&(grid.index<r.end_exclusive)&grid.complete_subsurface]
            expected=windows.loc[windows.start.eq(r.start)].iloc[0]
            for sid in SUBSURFACE_IDS:
                values=g[f'mean_{sid}'].tolist();mean=math.fsum(values)/len(values)
                coefficient=sum((y-mean)*cmath.exp(-2j*math.pi*t) for y,t in zip(values,g.t_days))/len(values)
                stored=complex(expected[f'F_real_{sid}'],expected[f'F_imag_{sid}'])
                assert abs(coefficient-stored)<1e-10
            independent.append(dict(start=str(r.start),bin_minutes=minutes,all_three_coefficients_match=True))
    tests=subprocess.run([sys.executable,'-m','unittest',
        'Project_description.sensorDB.tests.test_preliminary_fourier_phases.TemporalResolutionTests','-v'],cwd=ROOT,text=True,capture_output=True,check=True)
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('concentration_summary',concentration),('concentration_bins',bins),
        ('phase_windows_15min',wf),('phase_windows_2h',wc),('phase_comparison',comparison),('phase_summary',summary),('phase_extrema_diagnostics',diagnostics)]:
        data.to_csv(out/f'{name}.csv',index=False,float_format='%.17g')
    for name,data,index in [('concentration_residuals',points,False),('matched_source_observations',cohort,True),('matched_grid_2h',coarse,True)]:
        data.to_csv(out/f'{name}.csv.gz',index=index,float_format='%.17g',compression={'method':'gzip','mtime':0})
    examples,captions=temporal_figures(points,concentration,bins,comparison,metadata,out)
    examples.to_csv(out/'concentration_examples.csv',index=False)
    for rel,sha in {**hashes,**meta_hashes,str(historical_path.relative_to(ROOT)):historical_sha}.items():
        assert digest(ROOT/rel)==sha,rel
    passport=dict(stage='0.5',baseline_commit='8c3b6ff79e679585ef8f34e77a3006909b2ca2d5',decision_date='2026-11-02',
        accepted_passport=str(CANDIDATE_PASSPORT.relative_to(ROOT)),accepted_passport_sha256=digest(CANDIDATE_PASSPORT),
        selector=accepted['selector'],metadata=metadata,time_basis=time_note,start=str(GRID_START),end_exclusive=str(GRID_END),
        window_days=14,step_days=1,f0_cycles_per_day=1,time_origin=str(START),
        common_observations_per_sensor=len(cohort),source_counts=INTRABIN_COUNTS,
        concentration='Own-sensor 2h means mapped to identical actual source timestamps; each accepted point weight 1/N',
        phase_support='Intersection of occupied 15min calendar cells across all three sensors, one observation per sensor per cell; exactly the same source records in 2h aggregates',
        phase_times='Calendar centers: fine start +7min30s; coarse start +1h. Actual source timestamps retained separately. No interpolation.',
        weights='Fine: 1/N per occupied fine cell. Coarse: 1/K per occupied 2h cell, hence 1/(K*n_b) per constituent observation. Each representation subtracts its own equally weighted window mean.',
        phase='wrap(Arg(F_upper*conj(F_lower))) in [-pi,pi); positive ideal sinusoidal phase indicates lower-channel lag',
        difference='wrap(Arg(exp(i*(phase_15min-phase_2h)))); no acceptance threshold',
        input_sha256={**used,**meta_hashes,str(historical_path.relative_to(ROOT)):historical_sha},
        upstream_hashes_verified=len(hashes),reference_document_updates=updates,
        implementation_sha256=digest(Path(__file__)),captions=captions,
        versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__))
    (out/'passport.json').write_text(json.dumps(passport,ensure_ascii=False,indent=2)+'\n')
    validation=dict(tests_passed=True,test_log=tests.stderr+tests.stdout,accepted_counts=INTRABIN_COUNTS,
        own_sensor_means_and_counts_match_intrabin_exactly=True,common_source_counts_match_in_every_window=True,
        max_abs_bin_residual_sum_ppm=float(bins.sum_residual_ppm.abs().max()),numeric_zero_sum_bound_ppm=float(zero_bound),
        independent_scalar_fourier_checks=independent,windows=505,window_days=14,step_days=1,
        fine_common_cells=len(cohort),coarse_common_cells=int(coarse.complete_subsurface.sum()),
        upstream_hashes_verified=len(hashes),upstream_inputs_unchanged=True,
        artifact_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='validation.json'})
    (out/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    return concentration,summary,diagnostics,validation


def load_temporal_resolution(out=TEMPORAL_OUT):
    """Short research notebook reads published results only."""
    out=Path(out)
    return tuple(pd.read_csv(out/f'{name}.csv',float_precision='round_trip')
        for name in ('concentration_summary','phase_summary','phase_extrema_diagnostics','concentration_examples'))


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
        py_phase = phases[[p+'_rad' for p in PAIRS]].to_numpy()
        np.testing.assert_allclose(js_phase,py_phase,atol=1e-9*np.pi/180,rtol=0,equal_nan=True)
        assert [r['n'] for r in actual] == coverage.complete_bins.tolist()
        assert [r['doubtful'] for r in actual] == coverage.doubtful_bins.tolist()
        assert [r['longestGapHours'] for r in actual] == coverage.longest_gap_hours.tolist()
        records.append(dict(window_days=width,step_days=step,windows=len(actual),
                            maximum_phase_difference_rad=float(np.nanmax(np.abs(js_phase-py_phase)))))
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
    valid = phases[[p+'_rad' for p in PAIRS]].notna().all(axis=1)
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
    parser.add_argument('--grid-resolution', action='store_true', help='Run only the accepted-input 15min/2h diagnostic')
    parser.add_argument('--intrabin-2h', action='store_true', help='Only accepted raw concentration within 2h bins; no Fourier calculation')
    parser.add_argument('--temporal-resolution-14d', action='store_true', help='Only matched-observation 15min/2h comparison with 14-day windows')
    args = parser.parse_args()
    if sum((args.grid_resolution,args.intrabin_2h,args.temporal_resolution_14d)) > 1:
        parser.error('Choose one diagnostic')
    if args.temporal_resolution_14d:
        _, _, _, report = generate_temporal_resolution(TEMPORAL_OUT if args.output == OUT else args.output)
    elif args.intrabin_2h:
        _, _, _, report = generate_intrabin_2h(INTRABIN_OUT if args.output == OUT else args.output)
    elif args.grid_resolution:
        _, _, _, _, report = generate_grid_resolution(RESOLUTION_OUT if args.output == OUT else args.output)
    else:
        _, _, _, report = generate(args.output,args.window_days,args.step_days)
    print(json.dumps({k:v for k,v in report.items() if k not in ('test_log','synthetic')},ensure_ascii=False,indent=2))

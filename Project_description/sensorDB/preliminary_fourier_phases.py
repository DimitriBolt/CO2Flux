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
    args = parser.parse_args()
    if args.grid_resolution:
        _, _, _, _, report = generate_grid_resolution(RESOLUTION_OUT if args.output == OUT else args.output)
    else:
        _, _, _, report = generate(args.output,args.window_days,args.step_days)
    print(json.dumps({k:v for k,v in report.items() if k not in ('test_log','synthetic')},ensure_ascii=False,indent=2))

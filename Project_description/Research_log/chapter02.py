"""Глава 2: утверждённый месячный допуск, прежний метод H/A и его отображение.

Исторические примеры сохранены; они не определяют текущий допуск архива.
Основной запуск: prepare_chapter02.py --final [--refresh].
"""

from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from Project_description.Research_log.chapter01 import _column_settings, _format_figure, column, load_october
from diurnal_analysis import observation_intervals, fit_diurnal, wrap_hours, PERIOD_HOURS, COLORS, PAIRS

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data/ch02"
EXAMPLE_DAY = pd.Timestamp("2017-11-16")
COMPARISON_DAYS = (EXAMPLE_DAY, pd.Timestamp("2017-11-17"))
DAILY_DATES = pd.date_range("2017-11-16", "2017-11-24")
FIGURES = ("raw", "counts", "stuck", "coverage", "example_trend", "example_diurnal", "example_lags", "two_days", "daily_series", "d3_review", "phase_sensitivity", "month_comparison")
# Два обнаруженных интервала, для которых принято исключение 25.09.2026.
# Это явное решение для данного снимка, а не общий порог поиска неисправностей.
STUCK_INTERVALS = (
    ("2017-11-12 06:10:04", "2017-11-14 23:55:08"),
    ("2017-11-26 06:25:05", "2017-11-26 08:25:03"),
)


def calculation_calendar(raw, sensor_ids, start, end_exclusive,
                         diagnostic_constants, approved_exclusions, *, constants_require_decision=True):
    """Календарь A/B/C без средних, подгонок, H(d) или A(d).

    Использует прежние trend_window_mask()/observation_intervals() и контекст
    example_trend()/november_selection(): [d-1 сутки, d+2 суток), обрезанный
    только границами выбранного блока. Шаг определяется прежним способом
    в каждом переданном контексте; месячные границы не разрывают данные.

    diagnostic_constants и approved_exclusions — таблицы level/start/end
    с включёнными концами. Первая только помечает уже найденные участки,
    вторая содержит исключительно явно согласованные исключения.
    Никакого автоматического правила исключения постоянных значений нет.
    Недопустимый диапазон 0<C<=3000 и согласованные исключения маскируются
    в рабочей копии для проверки окон; все исходные строки сохраняются.

    A: полные общие сутки, нет повторов в контексте и нерешённых постоянных
    участков в фактических окнах выбранного дня. B: заведомо не хватает
    покрытия/окон после установленных ограничений. C: покрытие достаточно,
    но нужны решения по повторам или постоянным участкам. B имеет приоритет
    перед C; обе группы причин сохраняются. A не гарантирует надёжность фазы.
    Месячный допуск исторического блока задан источником, здесь не переоценён.
    """
    start, end_exclusive = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    if (pd.isna(start) or pd.isna(end_exclusive) or start.tzinfo is not None
            or end_exclusive.tzinfo is not None or end_exclusive <= start):
        raise ValueError("Нужен непустой интервал местного времени базы без часового пояса.")
    if len(sensor_ids) != 3 or len(set(sensor_ids.values())) != 3:
        raise ValueError("Нужны три разных датчика с фактическими уровнями.")
    if column.TREND_WINDOW != pd.Timedelta(hours=24):
        raise ValueError("Для этого календаря согласовано окно 24 часа.")
    selected = raw.loc[raw.sensorid.isin(sensor_ids.values())].copy()
    selected["localdatetime"] = pd.to_datetime(selected.localdatetime)
    if selected.localdatetime.isna().any() or selected.localdatetime.dt.tz is not None:
        raise ValueError("Нужны определённые местные временные метки без часового пояса.")
    selected = selected.loc[selected.localdatetime.ge(start)
                            & selected.localdatetime.lt(end_exclusive)]
    channels = {}
    for level, sid in sensor_ids.items():
        rows = selected.loc[selected.sensorid.eq(sid)].sort_values("localdatetime", kind="stable").reset_index(drop=True)
        rows["known_exclusion"] = False
        rows["diagnostic_constant"] = rows.get("constant_run_hours", pd.Series(0., index=rows.index)).ge(1)
        for table, flag in ((approved_exclusions, "known_exclusion"),
                            (diagnostic_constants, "diagnostic_constant")):
            for interval in table.loc[table.level.eq(level)].itertuples():
                rows[flag] |= rows.localdatetime.between(pd.Timestamp(interval.start), pd.Timestamp(interval.end))
        channels[level] = rows
    result = []
    for day in pd.date_range(start.normalize(), end_exclusive.ceil("D"), inclusive="left"):
        next_day = day + pd.Timedelta(days=1)
        left = max(start, day-pd.Timedelta(days=1))
        right = min(end_exclusive, day+pd.Timedelta(days=2))
        entry = dict(date=day, context_start=left, context_end_exclusive=right)
        blocked, undecided = [], []
        for level, rows in channels.items():
            times = rows.localdatetime
            context = rows.iloc[times.searchsorted(left):times.searchsorted(right)].copy()
            times = context.localdatetime
            day_times = times.loc[times.ge(day) & times.lt(next_day)]
            entry[f"{level}_rows_day"] = len(day_times)
            if len(day_times):
                needed = times.ge(day_times.iloc[0]-pd.Timedelta(hours=12)) & times.lt(day_times.iloc[-1]+pd.Timedelta(hours=12))
            else:
                needed = pd.Series(False, index=context.index)
            finite = np.isfinite(context.datavalue.to_numpy())
            band = finite & context.datavalue.gt(0) & context.datavalue.le(3000)
            service = context.datavalue.le(column.ERROR_CODE_LIMIT)
            repeated = times.duplicated(keep=False)
            entry[f"{level}_duplicate_extra_rows_context"] = int(times.duplicated().sum())
            entry[f"{level}_duplicate_rows_required"] = int((needed & repeated).sum())
            entry[f"{level}_service_rows_required"] = int((needed & service).sum())
            entry[f"{level}_range_rows_required"] = int((needed & ~band & ~service).sum())
            entry[f"{level}_approved_exclusion_rows_required"] = int((needed & context.known_exclusion).sum())
            entry[f"{level}_constant_rows_required"] = int((needed & context.diagnostic_constant).sum())
            pending = needed & context.diagnostic_constant & ~context.known_exclusion & band & constants_require_decision
            entry[f"{level}_unresolved_constant_rows_required"] = int(pending.sum())
            valid = band & ~context.known_exclusion
            work = context.assign(datavalue=context.datavalue.where(valid))
            step = times.diff()
            typical = step.loc[step.gt(pd.Timedelta(0))].median()
            entry[f"{level}_typical_step_seconds"] = typical.total_seconds() if pd.notna(typical) else np.nan
            for label, mask in (("records", valid), ("window", column.trend_window_mask(work))):
                intervals = observation_intervals(times, mask, column.GAP_FACTOR)
                entry[f"{level}_{label}"] = any(a <= day and next_day <= b for a, b in intervals)
            doubtful = context.get("doubt_reason", pd.Series("", index=context.index)).ne("") & valid
            entry[f"{level}_doubtful_rows_context"] = int(doubtful.sum())
            entry[f"{level}_doubtful_rows_required"] = int((needed & doubtful).sum())
            # This is a sensitivity check of temporal eligibility, NOT an
            # automatic cleaning decision. The original work values are kept.
            # Reuse the exact same window/coverage rules, including the support
            # observation at/after the next midnight and timestamp jitter.
            unambiguous_window = entry[f"{level}_window"]
            if unambiguous_window and doubtful.any():
                alternate = work.assign(datavalue=work.datavalue.mask(doubtful))
                intervals = observation_intervals(times, column.trend_window_mask(alternate), column.GAP_FACTOR)
                unambiguous_window = any(a <= day and next_day <= b for a, b in intervals)
            entry[f"{level}_unambiguous_window"] = unambiguous_window
            if not entry[f"{level}_window"]:
                detail = ("нет измерений суток" if day_times.empty else
                          "неполное покрытие суток" if not entry[f"{level}_records"] else
                          "неполное окружение ±12 ч")
                flags = []
                if day-pd.Timedelta(hours=12) < start or next_day+pd.Timedelta(hours=12) > end_exclusive:
                    flags.append("граница выбранного блока")
                if entry[f"{level}_service_rows_required"]:
                    flags.append("служебные значения")
                if entry[f"{level}_range_rows_required"]:
                    flags.append("нарушение 0<C≤3000")
                if entry[f"{level}_approved_exclusion_rows_required"]:
                    flags.append("согласованное ноябрьское исключение")
                blocked.append(f"{level}: {detail}" + (" ("+", ".join(flags)+")" if flags else ""))
            if entry[f"{level}_duplicate_extra_rows_context"]:
                undecided.append(f"{level}: повторные метки в трёхсуточном контексте")
            if pending.any():
                undecided.append(f"{level}: диагностический постоянный участок в окнах расчёта")
        entry["common_records"] = all(entry[f"{level}_records"] for level in sensor_ids)
        entry["common_window"] = all(entry[f"{level}_window"] for level in sensor_ids)
        entry["category"] = "B" if blocked else "C" if undecided else "A"
        entry["blocking_reasons"] = "; ".join(blocked)
        entry["decision_reasons"] = "; ".join(undecided)
        entry["reason"] = "; ".join(blocked+undecided) or "Расчёт возможен по существующим правилам"
        result.append(entry)
    return pd.DataFrame(result)

HISTORICAL_STEM = "W_R4_C-4_2017-10-01_to_2020-09-01"


def calculate_trajectories(raw, sensor_ids, calendar, approved_exclusions):
    """Apply the Chapter 1 algorithm only to accepted A days and their saved context.

    No calendar revision, deduplication, new quality threshold or plateau detector.
    The global context fit returned by analyze_column is not used as a daily result.
    """
    calendar = calendar.copy()
    for key in ("date", "context_start", "context_end_exclusive"):
        calendar[key] = pd.to_datetime(calendar[key])
    if calendar.date.duplicated().any() or not calendar.category.isin(["A", "B", "C"]).all():
        raise ValueError("Invalid accepted calendar")
    work = raw.loc[raw.sensorid.isin(sensor_ids.values())].copy()
    work["localdatetime"] = pd.to_datetime(work.localdatetime)
    valid = np.isfinite(work.datavalue) & work.datavalue.gt(0) & work.datavalue.le(3000)
    for item in approved_exclusions.itertuples():
        valid &= ~(work.sensorid.eq(sensor_ids[item.level]) & work.localdatetime.between(
            pd.Timestamp(item.start), pd.Timestamp(item.end)))
    work["datavalue"] = work.datavalue.where(valid)
    work = work.sort_values("localdatetime", kind="stable").reset_index(drop=True)
    times = work.localdatetime
    daily, failures = [], {}
    for item in calendar.loc[calendar.category.eq("A")].itertuples():
        context = work.iloc[times.searchsorted(item.context_start):
                            times.searchsorted(item.context_end_exclusive)]
        try:
            result = column.analyze_column(context, sensor_ids, item.context_start,
                                           item.context_end_exclusive, daily_dates=[item.date])
            part = result.daily.loc[result.daily.date.eq(item.date)].copy()
            if len(part) != 3 or set(part.level) != set(sensor_ids):
                raise ValueError("Прежний алгоритм не подтверждает полные общие сутки")
            part["sensorid"] = part.level.map(sensor_ids)
            part["phase_status"] = np.where(part.peak_hour_local.isna(), "undefined", "not_assessed")
            part["phase_note"] = np.where(part.peak_hour_local.isna(),
                "Нулевая/численно неразличимая амплитуда по прежней защите fit_diurnal",
                "Надёжность отдельно не установлена; R² сохранён без порога")
            # Existing documented sensitivity result, not a new classifier.
            known = (part.date.eq(pd.Timestamp("2017-11-22")) & part.level.eq("D3")
                     & part.sensorid.eq(1026) & part.peak_hour_local.notna())
            part.loc[known, "phase_status"] = "known_sensitive"
            part.loc[known, "phase_note"] = "Ранее установлена чувствительность к двухчасовым участкам; details/02_data_checks.md"
            daily.append(part)
        except ValueError as error:
            failures[item.date] = str(error)
    diagnostics = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame()
    table = calendar.copy().set_index("date")
    table["calculation_status"] = np.where(table.category.eq("A"), "failed", "skipped")
    table["calculation_error"] = pd.Series(failures, dtype=str)
    if len(diagnostics):
        table.loc[diagnostics.date.unique(), "calculation_status"] = "computed"
        for level in sensor_ids:
            part = diagnostics.loc[diagnostics.level.eq(level)].set_index("date")
            for field, prefix in (("peak_hour_local", "H"), ("amplitude_ppm", "A"),
                                  ("r_squared", "R2"), ("phase_status", "phase_status")):
                table[f"{prefix}_{level}"] = part[field]
    return table.reset_index(), diagnostics


# Approved implementation, 30 September 2026. These definitions complete the
# numerical I.G. rules; they are not attributed to the report's unknown code.
ADMISSION_PARAMETERS = dict(service_limit=-9999, concentration_min_exclusive=0,
    concentration_max=3000, window_samples=7, mad_scale=1.4826, mad_multiplier=6,
    cadence_ratio=column.GAP_FACTOR, bins_hours=6, minimum_per_bin=2,
    minimum_days=10, monthly_concentration_min=100, monthly_concentration_max=2000,
    monthly_amplitude_min=8, period_hours=24)


def clean_admission_measurements(raw, sensor_ids, approved_exclusions):
    """Immutable input; unique timestamp work view with a reproducible decision log.

    Seven original consecutive timestamps, including invalid ones, form a window.
    All seven must be finite and its six spacings must have max/min <= the
    existing gap factor (1.5). Thus invalid values, gaps and cadence transitions
    cannot be silently jumped. Non-centred windows are never used at boundaries.
    Only the centre outside the MAD band with all six neighbours inside is
    unambiguous. Adjacent candidates are retained, never removed iteratively.
    """
    parts = []
    for level, sid in sensor_ids.items():
        source = raw.loc[raw.sensorid.eq(sid)].copy()
        source['localdatetime'] = pd.to_datetime(source.localdatetime)
        if source.localdatetime.isna().any():
            raise ValueError('Undefined source timestamps')
        grouped = source.groupby('localdatetime', sort=True).datavalue
        work = grouped.agg(datavalue='first', source_rows='size').reset_index()
        conflict = grouped.nunique(dropna=False).gt(1).to_numpy()
        work['sensorid'], work['variableid'], work['level'] = sid, 9, level
        work['original_value'] = work.datavalue
        work['exact_duplicate_extra'] = np.where(conflict, 0, work.source_rows-1)
        reason = np.full(len(work), '', dtype=object)
        y = work.datavalue.to_numpy(float, copy=True)
        reason[~np.isfinite(y)] = 'nonfinite'
        reason[np.isfinite(y) & ((y <= 0) | (y > 3000))] = 'outside_0_3000'
        reason[y <= -9999] = 'service_code'
        reason[conflict] = 'conflicting_timestamp'
        for row in approved_exclusions.loc[approved_exclusions.level.eq(level)].itertuples():
            mask = work.localdatetime.between(row.start, row.end).to_numpy()
            reason[mask & (reason == '')] = 'approved_individual_exclusion'
        y[reason != ''] = np.nan
        doubtful = np.full(len(work), '', dtype=object)
        eligible = np.zeros(len(work), dtype=bool)
        candidate = np.zeros(len(work), dtype=bool)
        isolated = np.zeros(len(work), dtype=bool)
        if len(work) >= 7:
            windows = np.lib.stride_tricks.sliding_window_view(y, 7)
            dt = np.diff(work.localdatetime.to_numpy()).astype('timedelta64[ns]').astype(float)/1e9
            spacings = np.lib.stride_tricks.sliding_window_view(dt, 6)
            valid = np.isfinite(windows).all(axis=1) & (spacings.min(axis=1) > 0)
            valid &= spacings.max(axis=1) <= column.GAP_FACTOR*spacings.min(axis=1)
            med = np.median(windows, axis=1)
            dev = np.abs(windows-med[:, None])
            mad = np.median(dev, axis=1)
            band = 6*1.4826*mad
            outside = dev > band[:, None]
            candidate[3:-3] = valid & (mad > 0) & outside[:, 3]
            isolated[3:-3] = candidate[3:-3] & (outside.sum(axis=1) == 1)
            # A constant seven-point window is not doubtful. A non-flat zero-MAD
            # centre is retained and flagged; multi-point excursions are retained.
            zero = valid & (mad == 0) & (dev[:, 3] > 0)
            doubtful[3:-3][zero] = 'zero_MAD_deviation'
            eligible[3:-3] = valid
        adjacent = np.roll(candidate, 1) | np.roll(candidate, -1)
        removable = isolated & ~adjacent
        doubtful[candidate & ~removable] = 'ambiguous_excursion'
        reason[removable] = 'isolated_7_sample_MAD_spike'
        y[removable] = np.nan
        work['datavalue'] = y
        work['exclusion_reason'], work['doubt_reason'] = reason, doubtful
        work['despike_window_eligible'] = eligible
        # Diagnostic only: no duration of a constant run changes admission.
        same = work.original_value.eq(work.original_value.shift()) & ~conflict
        step = work.localdatetime.diff().dt.total_seconds()
        same &= step.le(column.GAP_FACTOR*step[step.gt(0)].median())
        groups = (~same).cumsum()
        bounds = work.groupby(groups).localdatetime.agg(['min', 'max'])
        work['constant_run_hours'] = groups.map((bounds['max']-bounds['min']).dt.total_seconds()/3600)
        parts.append(work)
    return pd.concat(parts, ignore_index=True)


def monthly_day_fit(frame):
    """Raw-concentration fit for the monthly gate, NOT the residual H/A fit."""
    frame = frame.loc[np.isfinite(frame.datavalue)]
    counts = frame.localdatetime.dt.hour.floordiv(6).value_counts().reindex(range(4), fill_value=0)
    if (counts < 2).any() or frame.localdatetime.duplicated().any():
        return None
    try:
        fit = fit_diurnal(frame.localdatetime, frame.datavalue)
    except ValueError:
        return None
    return dict(concentration=float(frame.datavalue.median()), amplitude=fit['amplitude_ppm'],
                n_obs=len(frame), r_squared=fit['r_squared'])


def admission_day_bounds(frame, fit):
    """Enclose ALL subsets of unresolved samples, not just keep-all/drop-all.

    With <= 10 doubtful samples enumerate exactly (a computation shortcut only).
    Otherwise use a rigorous LS perturbation bound and concentration extrema;
    an inconclusive enclosure is reported, never used to certify admission.
    No replacement/interpolation of doubtful observations is considered.
    """
    frame = frame.loc[np.isfinite(frame.datavalue)].reset_index(drop=True)
    indices = np.flatnonzero(frame.doubt_reason.ne('').to_numpy())
    if not len(indices):
        return dict(c_lo=fit['concentration'], c_hi=fit['concentration'],
                    a_lo=fit['amplitude'], a_hi=fit['amplitude'], guaranteed=True, bounds='exact')
    if len(indices) <= 10:
        values, guaranteed = [], True
        for mask in range(1 << len(indices)):
            keep = np.ones(len(frame), bool)
            keep[indices[[bool(mask & (1 << j)) for j in range(len(indices))]]] = False
            part = monthly_day_fit(frame.loc[keep])
            guaranteed &= part is not None
            if part is not None:
                values.append(part)
        return dict(c_lo=min(x['concentration'] for x in values), c_hi=max(x['concentration'] for x in values),
                    a_lo=min(x['amplitude'] for x in values), a_hi=max(x['amplitude'] for x in values),
                    guaranteed=guaranteed, bounds='exact_subsets')
    guaranteed = monthly_day_fit(frame.loc[frame.doubt_reason.eq('')]) is not None
    t = (frame.localdatetime-frame.localdatetime.dt.normalize()).dt.total_seconds().to_numpy()/3600
    x = np.column_stack([np.ones(len(t)), np.cos(2*np.pi*t/24), np.sin(2*np.pi*t/24)])
    y = frame.datavalue.to_numpy()
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    # X_remaining' X_remaining >= X_certain' X_certain in the PSD ordering.
    certain = frame.doubt_reason.eq('').to_numpy()
    lower_eigenvalue = np.linalg.eigvalsh(x[certain].T @ x[certain]).min()
    radius = (np.linalg.norm(x[indices], axis=1)*np.abs(y-x @ beta)[indices]).sum()
    radius = radius/lower_eigenvalue if lower_eigenvalue > 0 else np.inf
    return dict(c_lo=float(y.min()), c_hi=float(y.max()),
                a_lo=max(0, fit['amplitude']-radius), a_hi=fit['amplitude']+radius,
                guaranteed=guaranteed, bounds='certified_enclosure')


def _monthly_median_bounds(days, prefix):
    """Extrema over optional valid days, allowing every count >= ten."""
    certain = days.loc[days.guaranteed]
    optional = days.loc[~days.guaranteed]
    lows, highs = [], []
    for n in range(max(0, 10-len(certain)), len(optional)+1):
        lows.append(np.median(np.r_[certain[prefix+'_lo'], np.sort(optional[prefix+'_lo'])[:n]]))
        highs.append(np.median(np.r_[certain[prefix+'_hi'], np.sort(optional[prefix+'_hi'])[::-1][:n]]))
    return (min(lows), max(highs)) if lows else (np.nan, np.nan)


def monthly_admission(work, sensor_ids, start, end_exclusive):
    """Every month, every sensor; gates and data ambiguities are distinct."""
    daily, monthly = [], []
    for level, sid in sensor_ids.items():
        rows = work.loc[work.sensorid.eq(sid)]
        for day, part in rows.groupby(rows.localdatetime.dt.normalize()):
            fit = monthly_day_fit(part)
            record = dict(date=day, month=day.strftime('%Y-%m'), level=level, sensorid=sid,
                          valid=fit is not None, doubtful_samples=int(part.doubt_reason.ne('').sum()))
            if fit is not None:
                record.update(fit)
                record.update(admission_day_bounds(part, fit))
            daily.append(record)
    daily = pd.DataFrame(daily)
    for month in pd.period_range(pd.Timestamp(start), pd.Timestamp(end_exclusive)-pd.Timedelta(nanoseconds=1), freq='M'):
        for level, sid in sensor_ids.items():
            d = daily.loc[daily.month.eq(str(month)) & daily.level.eq(level) & daily.valid].copy()
            n = len(d)
            record = dict(month=str(month), level=level, sensorid=sid, valid_days=n,
                          median_concentration=d.concentration.median() if n else np.nan,
                          median_raw_amplitude=d.amplitude.median() if n else np.nan)
            record['minimum_valid_days'] = int(d.guaranteed.sum()) if n else 0
            record['doubtful_valid_days'] = int(d.doubtful_samples.gt(0).sum()) if n else 0
            reasons = []
            if n < 10:
                status = 'fail'
                reasons.append('fewer_than_10_valid_days')
                clo=chi=alo=ahi=np.nan
            else:
                d['guaranteed'] = d.guaranteed.astype(bool)
                clo, chi = _monthly_median_bounds(d, 'c')
                alo, ahi = _monthly_median_bounds(d, 'a')
                if chi < 100 or clo > 2000: reasons.append('median_concentration_outside_100_2000')
                if ahi < 8: reasons.append('median_raw_amplitude_below_8')
                if reasons:
                    status = 'fail'
                elif record['minimum_valid_days'] >= 10 and clo >= 100 and chi <= 2000 and alo >= 8:
                    status = 'pass'
                    reasons.append('all_three_criteria_invariant_to_flagged_samples')
                else:
                    status = 'undetermined'
                    reasons.append('unresolved_excursions_can_affect_monthly_gate')
            record.update(status=status, reason=';'.join(reasons), concentration_lower=clo,
                          concentration_upper=chi, amplitude_lower=alo, amplitude_upper=ahi)
            monthly.append(record)
    return pd.DataFrame(monthly), daily


def final_admission_calendar(work, sensor_ids, monthly, start, end_exclusive, approved_exclusions):
    """Keep all dates/reasons; admit H/A only after monthly and temporal gates."""
    empty = pd.DataFrame(columns=['level', 'start', 'end'])
    calendar = calculation_calendar(work, sensor_ids, start, end_exclusive, empty,
                                    approved_exclusions, constants_require_decision=False)
    calendar['month'] = calendar.date.dt.strftime('%Y-%m')
    calendar['technical_category'] = calendar.category
    calendar['technical_reason'] = calendar.reason
    for i, row in calendar.iterrows():
        statuses = monthly.loc[monthly.month.eq(row.month)].set_index('level')
        status = 'fail' if statuses.status.eq('fail').any() else 'undetermined' if statuses.status.eq('undetermined').any() else 'pass'
        calendar.loc[i, 'admission_status'] = status
        notes = []
        if status != 'pass':
            notes = [f'{l}: {statuses.loc[l,"status"]}: {statuses.loc[l,"reason"]}'
                     for l in sensor_ids if statuses.loc[l,'status'] != 'pass']
        n_doubts = int(sum(row[f'{l}_doubtful_rows_required'] for l in sensor_ids))
        calendar.loc[i, 'unresolved_samples_required'] = n_doubts
        unresolved = row.technical_category == 'A' and not all(row[f'{l}_unambiguous_window'] for l in sensor_ids)
        calendar.loc[i, 'unresolved_context'] = unresolved
        if unresolved: notes.append('unresolved_excursion_in_required_12h_context')
        if status == 'fail' or row.technical_category == 'B': calendar.loc[i, 'category'] = 'B'
        elif status == 'undetermined' or unresolved: calendar.loc[i, 'category'] = 'C'
        if row.technical_category != 'A': notes.append(row.technical_reason)
        calendar.loc[i, 'reason'] = '; '.join(notes) or 'monthly_pass_and_complete_unambiguous_daily_context'
    return calendar


def load_historical_trajectories():
    """Read the fixed archive and accepted calendar, with original input hashes."""
    output = HERE / "output/ch02"
    provenance = json.loads((output / f"{HISTORICAL_STEM}_calendar_provenance.json").read_text())
    frames = []
    for filename, digest in provenance["input_hashes"].items():
        path = Path(filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Изменён исходный снимок: {path.name}")
        if path.suffix == ".parquet":
            frames.append(pd.read_parquet(path))
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[raw.variableid.eq(9)]
    raw["localdatetime"] = pd.to_datetime(raw.localdatetime)
    raw = raw.loc[raw.localdatetime.ge(provenance["start_inclusive"])
                  & raw.localdatetime.lt(provenance["end_exclusive"])]
    calendar = pd.read_csv(output / f"{HISTORICAL_STEM}_calendar.csv")
    exclusions = pd.DataFrame(provenance["approved_exclusions"])
    return calculate_trajectories(raw, provenance["sensor_ids"], calendar, exclusions)


def cyclic_segments(first, second, period=24.0):
    """Split shortest torus display links at cube faces; never unwrap the data.

    Links are visual guides, not inferred intermediate phase measurements.
    Exactly antipodal coordinates have no unique shortest branch: omit the link.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if not np.isfinite([first, second]).all():
        return []
    delta = (second-first+period/2) % period-period/2
    if np.any(np.isclose(np.abs(delta), period/2, rtol=0, atol=1e-12)):
        return []
    cuts = [0., 1.]
    for coordinate, shift in zip(first, delta):
        if shift:
            for face in (0., period):
                fraction = (face-coordinate)/shift
                if 0 < fraction < 1:
                    cuts.append(fraction)
    cuts = sorted(set(cuts))
    pieces = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = first + (left+right)/2*delta
        translation = np.floor(midpoint/period)*period
        pieces.append(np.array([first+left*delta-translation, first+right*delta-translation]))
    return pieces


def trajectory_segments(table, kind, levels=("D1", "D2", "D3")):
    """Only adjacent calendar dates with defined coordinates can be linked."""
    selected = table.loc[table.calculation_status.eq("computed")].sort_values("date")
    xyz = selected[[f"{kind}_{level}" for level in levels]].to_numpy(float)
    dates = pd.to_datetime(selected.date)
    segments, colors = [], []
    for i in range(1, len(selected)):
        if dates.iloc[i]-dates.iloc[i-1] != pd.Timedelta(days=1):
            continue
        if not np.isfinite(xyz[i-1:i+1]).all():
            continue
        pieces = cyclic_segments(xyz[i-1], xyz[i]) if kind == "H" else [xyz[i-1:i+1]]
        segments.extend(pieces)
        colors.extend([mdates.date2num(dates.iloc[i])]*len(pieces))
    return segments, colors


def trajectory_display_groups(table):
    """Display the saved monthly status and existing boundary caveat, without re-admission."""
    boundary = table.get("context_admission_note", pd.Series("", index=table.index)).fillna("").ne("")
    definitions = [
        (table.admission_status.eq("confirmed") & ~boundary, "Основание I.G.", "o", "circle"),
        (table.admission_status.eq("confirmed") & boundary, "Граничное окружение", "s", "square-open"),
        (table.admission_status.eq("unconfirmed"), "Предварительно, без допуска", "D", "diamond-open"),
    ]
    return [(table.loc[mask], f"{label} · {int(table.loc[mask].calculation_status.eq('computed').sum())} суток", marker, symbol)
            for mask, label, marker, symbol in definitions]


def plot_trajectory(table, kind, levels=("D1", "D2", "D3"), *, preliminary=False, final=False):
    """Date-colored 3D trajectory with continuous fit-quality display, no selection."""
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    if kind not in ("H", "A"):
        raise ValueError("kind must be H or A")
    if final:
        table = table.loc[table.admission_status.eq("pass")].copy()
    if preliminary:
        table = table.loc[table.admission_status.isin(["confirmed", "unconfirmed"])].copy()
    selected = table.loc[table.calculation_status.eq("computed")].sort_values("date")
    if selected.empty:
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.axis("off")
        ax.text(.5, .5, f"{kind}(d): нет допущенных суточных оценок.\nПричины сохранены в календаре.",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    xyz = selected[[f"{kind}_{level}" for level in levels]].to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    dates = mdates.date2num(pd.to_datetime(selected.date))
    quality = selected[[f"R2_{level}" for level in levels]].min(axis=1).to_numpy()
    size = 12 + 42*np.nan_to_num(np.clip(quality, 0, 1), nan=0)
    norm = Normalize(mdates.date2num(pd.to_datetime(table.date).min()),
                     mdates.date2num(pd.to_datetime(table.date).max()))
    if preliminary or final:
        norm = Normalize(dates.min(), dates.max())
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    groups = trajectory_display_groups(table) if preliminary else [(table, "", "o", "circle")]
    segments, color_dates = [], []
    for group, _, _, _ in groups:
        pieces, times = trajectory_segments(group, kind, levels)
        segments.extend(pieces)
        color_dates.extend(times)
    links = Line3DCollection(segments, cmap="viridis", norm=norm, linewidths=.65, alpha=.35)
    links.set_array(np.asarray(color_dates))
    ax.add_collection3d(links)
    legend = []
    for group, label, marker, _ in groups:
        mask = selected.index.isin(group.index) & finite
        if not mask.any():
            continue
        filled = marker == "o"
        ax.scatter(*xyz[mask].T, c=dates[mask] if filled else None, cmap="viridis" if filled else None,
                   norm=norm if filled else None, s=size[mask], alpha=.88, depthshade=False,
                   marker=marker, facecolors=None if filled else "none",
                   edgecolors="none" if filled else plt.cm.viridis(norm(dates[mask])), linewidths=.85)
        if preliminary:
            legend.append(Line2D([], [], marker=marker, linestyle="none", color="#52616b",
                                 markerfacecolor="#52616b" if filled else "none", markersize=5, label=label))
    sensitive = selected[[f"phase_status_{level}" for level in levels]].eq("known_sensitive").any(axis=1).to_numpy() & finite
    if kind == "H" and sensitive.any():
        ax.scatter(*xyz[sensitive].T, marker="o", facecolors="none", edgecolors="#555555", s=35, linewidths=.9,
                   depthshade=False, zorder=20)
    for axis, level in zip((ax.xaxis, ax.yaxis, ax.zaxis), levels):
        axis.set_label_text(f"{level}: {'максимум, ч' if kind == 'H' else 'амплитуда, ppm'}")
        axis.labelpad = 12
        if kind == "H":
            axis.set_ticks([0, 6, 12, 18, 24])
    if kind == "H":
        ax.set(xlim=(0, 24), ylim=(0, 24), zlim=(0, 24))
    else:
        for setter, values in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), xyz.T):
            setter(0, max(1., np.nanmax(values)*1.06))
    ax.view_init(elev=24, azim=-56)
    ax.set_box_aspect((1, 1, 1))
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=.62, pad=.11)
    ticks = pd.to_datetime(["2017-10-01", "2018-07-01", "2019-04-01", "2020-01-01", "2020-08-31"])
    if preliminary or final:
        ticks = pd.date_range(pd.to_datetime(selected.date).min(), pd.to_datetime(selected.date).max(), periods=6)
    colorbar.set_ticks(mdates.date2num(ticks), labels=[date.strftime("%m.%Y") for date in ticks])
    colorbar.set_label("Дата, местное время базы")
    if kind == "H" and sensitive.any():
        legend.append(Line2D([], [], marker="o", linestyle="none", color="#555555",
                                 markerfacecolor="none", markeredgewidth=.9, markersize=4,
                                 label="D3: чувствительная фаза 22.11.2017"))
    if legend:
        fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.045, .91), fontsize=9, frameon=False)
    period = "01.10.2017–31.08.2020"
    if preliminary or final:
        first, last = pd.to_datetime(selected.date).agg(["min", "max"])
        period = f"{first:%d.%m.%Y}–{last:%d.%m.%Y}"
    fig.suptitle(f"Траектория {kind}(d) · W_R4_C-4 · D1/D2/D3\n{period} · {finite.sum()} суточных точек", fontsize=15, y=.97)
    footer = ("H: 0 ≡ 24 ч; связи разрезаны на гранях куба. " if kind == "H" else "A: высота гармоники относительно её постоянного уровня. ")
    footer += "Пропуски не соединены.\nПлощадь точки: min R² трёх подгонок (не порог и не погрешность фазы).\n"
    footer += f"Неопределённых H: {selected[[f'H_{l}' for l in levels]].isna().any(axis=1).sum()}; надёжность остальных фаз отдельно не установлена."
    fig.text(.5, .035, footer, ha="center", fontsize=10)
    fig.subplots_adjust(left=.03, right=.86, bottom=.14, top=.89)
    return fig


def show_trajectory(table, kind, *, preliminary=False, final=False):
    """Store a static PNG directly in notebook output, using the same display function."""
    from IPython.display import Image, display
    fig = plot_trajectory(table, kind, preliminary=preliminary, final=final)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160)
        display(Image(data=buffer.getvalue()))
    finally:
        plt.close(fig)


def save_trajectories(table, diagnostics, directory=HERE / "output/ch02"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / f"{HISTORICAL_STEM}_trajectories.csv", index=False)
    diagnostics.to_csv(directory / f"{HISTORICAL_STEM}_daily_fits.csv", index=False)
    for kind in ("H", "A"):
        fig = plot_trajectory(table, kind)
        fig.savefig(directory / f"{HISTORICAL_STEM}_{kind}.png", dpi=180)
        fig.savefig(directory / f"{HISTORICAL_STEM}_{kind}.svg")
        plt.close(fig)



def interactive_trajectory(table, kind, levels=("D1", "D2", "D3"), *, preliminary=False, final=False):
    """Plotly view of saved vectors; reuse the static plot's torus links."""
    import plotly.graph_objects as go
    if kind not in ("H", "A"):
        raise ValueError("kind must be H or A")
    if final:
        table = table.loc[table.admission_status.eq("pass")].copy()
    if preliminary:
        # Explicit audit refusals never enter the exploratory figures, even if fitted.
        table = table.loc[table.admission_status.isin(["confirmed", "unconfirmed"])].copy()
    selected = table.loc[table.calculation_status.eq("computed")].sort_values("date")
    if selected.empty or not np.isfinite(selected[[f"{kind}_{l}" for l in levels]].to_numpy(float)).all(axis=1).any():
        fig = go.Figure()
        fig.update_layout(title=f"{kind}(d): нет определённых допущенных точек", annotations=[dict(
            text="Причины отсутствия оценок сохранены в календаре. Нулевой амплитуде не назначается фаза.",
            x=.5, y=.5, xref="paper", yref="paper", showarrow=False)])
        return fig
    xyz = selected[[f"{kind}_{level}" for level in levels]].to_numpy(float)
    finite = np.isfinite(xyz).all(axis=1)
    selected, xyz = selected.loc[finite], xyz[finite]
    dates = mdates.date2num(pd.to_datetime(selected.date))
    low, high = mdates.date2num(pd.to_datetime(table.date).agg(["min", "max"]))
    ticks = pd.to_datetime(["2017-10-01", "2018-07-01", "2019-04-01", "2020-01-01", "2020-08-31"])
    if preliminary or final:
        if selected.empty:
            raise ValueError("Нет определённых координат для предварительного графика")
        low, high = dates.min(), dates.max()
        ticks = pd.date_range(pd.to_datetime(selected.date).min(), pd.to_datetime(selected.date).max(), periods=6)
    status = {"not_assessed": "надёжность отдельно не установлена",
              "known_sensitive": "известная чувствительная фаза", "undefined": "фаза не определена"}
    notes = selected.apply(lambda row: "<br>".join(
        f"{level}: {status.get(row[f'phase_status_{level}'], 'нет оценки')}"
        for level in levels), axis=1)
    if preliminary:
        labels = {"confirmed": "Месяц: опубликованное подтверждение I.G. (методы различаются)",
                  "unconfirmed": "Месяц: научный допуск не установлен"}
        notes += "<br>" + selected.admission_status.map(labels)
        if "context_admission_note" in selected:
            notes += "<br>" + selected.context_admission_note.fillna("")
    custom = [[date.strftime("%d.%m.%Y"), *[float(row[f"R2_{l}"]) for l in levels], note]
              for date, (_, row), note in zip(pd.to_datetime(selected.date), selected.iterrows(), notes)]
    unit = "ч" if kind == "H" else "ppm"
    hover = ("Дата: %{customdata[0]}<br>"
             + f"{levels[0]}: %{{x:.5f}} {unit}<br>{levels[1]}: %{{y:.5f}} {unit}<br>{levels[2]}: %{{z:.5f}} {unit}<br>"
             + "R² D1/D2/D3: %{customdata[1]:.5f} / %{customdata[2]:.5f} / %{customdata[3]:.5f}<br>"
             + "%{customdata[4]}<extra></extra>")
    fig = go.Figure()
    groups = [(table, selected, "Суточные оценки", "circle")]
    if preliminary:
        groups = [(source, selected.loc[selected.index.isin(source.index)], label, symbol)
                  for source, label, _, symbol in trajectory_display_groups(table)]
    for source, _, label, _ in groups:
        segments, colors = trajectory_segments(source, kind, levels)
        coords, link_colors = [], []
        for segment, date in zip(segments, colors):
            coords.extend([segment[0].tolist(), segment[1].tolist(), [None]*3])
            link_colors.extend([date, date, date])
        if coords:
            points = list(zip(*coords))
            fig.add_trace(go.Scatter3d(x=points[0], y=points[1], z=points[2], mode="lines",
                line=dict(color=link_colors, colorscale="Viridis", cmin=low, cmax=high, width=2),
                legendgroup=label, opacity=.28, hoverinfo="skip", showlegend=False, connectgaps=False))
    quality = selected[[f"R2_{level}" for level in levels]].min(axis=1).to_numpy()
    colorbar_shown = False
    for _, points, label, symbol in groups:
        positions = np.flatnonzero(selected.index.isin(points.index))
        if not len(positions):
            continue
        fig.add_trace(go.Scatter3d(x=xyz[positions,0], y=xyz[positions,1], z=xyz[positions,2], mode="markers",
            customdata=[custom[i] for i in positions], hovertemplate=hover, name=label, legendgroup=label, showlegend=preliminary,
            marker=dict(color=dates[positions].tolist(), colorscale="Viridis", cmin=low, cmax=high,
                        symbol=symbol, showscale=not colorbar_shown,
                        size=np.sqrt(12+42*np.nan_to_num(np.clip(quality[positions],0,1),nan=0)).tolist(),
                        opacity=.9, colorbar=dict(title="Дата",tickvals=mdates.date2num(ticks).tolist(),
                        ticktext=[d.strftime("%m.%Y") for d in ticks]))))
        colorbar_shown = True
    sensitive = selected[[f"phase_status_{level}" for level in levels]].eq("known_sensitive").any(axis=1).to_numpy()
    if kind == "H" and sensitive.any():
        fig.add_trace(go.Scatter3d(x=xyz[sensitive,0], y=xyz[sensitive,1], z=xyz[sensitive,2],
            mode="markers", marker=dict(symbol="circle-open",size=4,color="#555555",line=dict(width=1)),
            customdata=[custom[i] for i in np.flatnonzero(sensitive)], hovertemplate=hover,
            name="D3 22.11.2017: чувствительная фаза"))
    axes = {}
    for axis, level in zip(("xaxis", "yaxis", "zaxis"), levels):
        axes[axis] = dict(title=f"{level}: {'максимум, ч' if kind=='H' else 'амплитуда, ppm'}")
        if kind == "H":
            axes[axis].update(range=[0,24],tickvals=[0,6,12,18,24],ticktext=["0 ≡ 24","6","12","18","24 ≡ 0"])
        else:
            axes[axis].update(rangemode="tozero")
    explanation = ("0 ≡ 24 ч: противоположные грани отождествлены; связи разрезаны на гранях. " if kind=="H" else "Амплитуда относительно постоянного уровня гармоники. ")
    explanation += "Пропуски не соединены.<br>Размер: min R² без порога; надёжность остальных фаз отдельно не установлена. Связи — ориентир порядка точек."
    title = f"{kind}(d) · W_R4_C-4 · 783 суток · 01.10.2017–31.08.2020"
    if preliminary or final:
        first, last = pd.to_datetime(selected.date).agg(["min", "max"])
        title = f"Траектория {kind}(d) · W_R4_C-4<br>{len(selected)} суток · {first:%d.%m.%Y}–{last:%d.%m.%Y}"
        missing = int((~finite).sum())
        explanation += f"<br>Неопределённые координаты: {missing} суток. Отказ аудита исключён. Месячный допуск не гарантирует надёжность фазы."
        explanation += "<br>Пустые ромбы — предварительные расчёты без месячного допуска; пустые квадраты — неподтверждённое окружение граничных суток. Методы отличаются от I.G."
    if final:
        explanation = ("0 ≡ 24 ч; противоположные грани отождествлены. " if kind == "H" else "Амплитуда в ppm. ")
        explanation += "Только совместный месячный pass и полный однозначный контекст. Пропуски не соединены.<br>Размер точки: min R² без порога; надёжность фаз отдельно не установлена."
        explanation += f"<br>Неопределённых H: {table.loc[table.calculation_status.eq('computed'), [f'H_{l}' for l in levels]].isna().any(axis=1).sum()} суток."
    fig.update_layout(title=title,
        template="plotly_white",scene=dict(**axes,aspectmode="cube",dragmode="orbit"),
        margin=dict(l=20,r=20,t=130 if preliminary else 100,b=120 if preliminary else 90),legend=dict(x=0,y=1.12 if preliminary else 1.06),
        annotations=[dict(text=explanation,x=.5,y=-.09,xref="paper",yref="paper",showarrow=False)],
        uirevision=f"chapter02-{kind}")
    if preliminary or final:
        fig.update_layout(title=dict(x=.03, y=.98, yanchor="top", font=dict(size=14)),
            legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11)),
            margin=dict(l=10, r=10, t=170, b=120), scene_camera=dict(eye=dict(x=1.7, y=1.7, z=1.7)))
    return fig


def save_preliminary_interactive(table, kind, path, *, final=False):
    """Self-contained browser view with wrapping notes outside the 3D canvas."""
    fig = interactive_trajectory(table, kind, preliminary=not final, final=final)
    notes = fig.layout.annotations[0].text
    fig.layout.annotations = ()
    fig.update_layout(margin=dict(b=10), height=740)
    html = fig.to_html(include_plotlyjs=True, full_html=True,
                      config={"scrollZoom": True, "responsive": True, "displaylogo": False})
    html = html.replace("<head>", '<head><meta name="viewport" content="width=device-width, initial-scale=1">'
                        '<style>body{margin:0;font:14px Arial,sans-serif;color:#2a3f5f}'
                        '.trajectory-note{padding:12px 24px 24px;line-height:1.5;max-width:1100px;margin:auto}</style>')
    html = html.replace("</body>", f'<div class="trajectory-note">{notes}</div></body>')
    Path(path).write_text(html)


def open_saved_trajectories(open_browser=True):
    """Export self-contained HTML from saved CSV; never fit or alter static files.

    PyCharm: run chapter02.py with the parameter --interactive.
    """
    import webbrowser
    directory = HERE / "output/ch02"
    table = pd.read_csv(directory / f"{HISTORICAL_STEM}_trajectories.csv", parse_dates=["date"])
    paths = []
    for kind in ("H", "A"):
        path = directory / f"{HISTORICAL_STEM}_{kind}_interactive.html"
        interactive_trajectory(table, kind).write_html(path, include_plotlyjs=True, full_html=True,
            config={"scrollZoom":True,"responsive":True,"displaylogo":False}, auto_open=False)
        paths.append(path)
        if open_browser:
            webbrowser.open_new_tab(path.resolve().as_uri())
    return paths


def audit_raw(raw, sensor_ids, service_limit=-9999, gap_factor=1.5, flat_hours=1.0):
    """Описательные проверки; ни одна запись не удаляется и не усредняется.

    Постоянный участок: точное равенство значений, без разрыва больше 1.5
    медианного положительного шага. Один час — порог вывода в диагностический
    список, а не критерий неисправности. Разброс на повторной метке сохраняется.
    """
    summaries, duplicates, constants, gaps, counts = [], [], [], [], []
    for level, sid in sensor_ids.items():
        rows = raw.loc[raw.sensorid.eq(sid)].sort_values("localdatetime", kind="stable")
        if rows.empty or rows.localdatetime.isna().any() or not np.isfinite(rows.datavalue).all():
            raise ValueError(f"{level}: пустой ряд, NaT или нечисловые значения.")
        steps = rows.localdatetime.diff().dt.total_seconds()
        typical = float(steps.loc[steps.gt(0)].median())
        repeated = rows.loc[rows.localdatetime.duplicated(keep=False)]
        n_conflicting = 0
        n_groups = 0
        for time, part in repeated.groupby("localdatetime", sort=True):
            unique_values = part.datavalue.nunique()
            n_groups += 1
            n_conflicting += int(unique_values > 1)
            duplicates.append(dict(level=level, sensorid=sid, localdatetime=time,
                                   rows=len(part), distinct_values=unique_values,
                                   minimum=part.datavalue.min(), maximum=part.datavalue.max(),
                                   spread_ppm=part.datavalue.max() - part.datavalue.min()))
        group = (rows.datavalue.ne(rows.datavalue.shift()) | steps.gt(gap_factor * typical)).cumsum()
        long_runs = 0
        for _, part in rows.groupby(group, sort=False):
            duration = (part.localdatetime.iloc[-1] - part.localdatetime.iloc[0]).total_seconds() / 3600
            if duration >= flat_hours and part.datavalue.iloc[0] > service_limit:
                long_runs += 1
                constants.append(dict(level=level, start=part.localdatetime.iloc[0],
                                      end=part.localdatetime.iloc[-1], rows=len(part),
                                      duration_hours=duration, value_ppm=part.datavalue.iloc[0]))
        prior_times = rows.localdatetime.shift()
        for idx in rows.index[steps.gt(gap_factor * typical)]:
            gaps.append(dict(level=level, previous=prior_times.loc[idx],
                             following=rows.loc[idx, "localdatetime"], duration_seconds=steps.loc[idx]))
        for day, part in rows.groupby(rows.localdatetime.dt.normalize()):
            counts.append(dict(level=level, date=day, rows=len(part),
                               unique_timestamps=part.localdatetime.nunique()))
        summaries.append(dict(level=level, sensorid=sid, rows=len(rows),
                              service_codes=int(rows.datavalue.le(service_limit).sum()),
                              duplicate_times=n_groups, conflicting_times=n_conflicting,
                              exact_extra_rows=int(rows.duplicated(["localdatetime", "datavalue"]).sum()),
                              repeated_time_extra_rows=int(rows.localdatetime.duplicated().sum()),
                              short_steps_le_10s=int(steps.between(0, 10).sum()),
                              typical_step_seconds=typical,
                              gaps=int(steps.gt(gap_factor * typical).sum()), long_constant_runs=long_runs))
    return {
        "summary": pd.DataFrame(summaries),
        "duplicates": pd.DataFrame(duplicates, columns=["level", "sensorid", "localdatetime", "rows",
                                                       "distinct_values", "minimum", "maximum", "spread_ppm"]),
        "constant_runs": pd.DataFrame(constants, columns=["level", "start", "end", "rows", "duration_hours", "value_ppm"]),
        "gaps": pd.DataFrame(gaps, columns=["level", "previous", "following", "duration_seconds"]),
        "daily_counts": pd.DataFrame(counts),
    }


@dataclass
class NovemberReport:
    manifest: dict
    raw: pd.DataFrame
    sensor_ids: dict
    audit: dict
    data_dir: Path

    def verify(self):
        dates = {"summary": [], "duplicates": ["localdatetime"], "constant_runs": ["start", "end"],
                 "gaps": ["previous", "following"], "daily_counts": ["date"]}
        for name, frame in self.audit.items():
            expected = pd.read_csv(self.data_dir / f"reference_{name}.csv", parse_dates=dates[name])
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected,
                                          check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8)
        return len(self.audit)

    def table(self, name="summary"):
        if name == "november_selection":
            return self.november_selection()
        if name in ("month_comparison_summary", "month_comparison_lags", "month_comparison_medians"):
            return self.month_comparison()[("month_comparison_summary", "month_comparison_lags", "month_comparison_medians").index(name)]
        if name in ("phase_sensitivity_summary", "phase_sensitivity_values"):
            return self.phase_sensitivity()[("phase_sensitivity_summary", "phase_sensitivity_values").index(name)]
        if name in ("d3_review_summary", "d3_review_values"):
            return self.d3_review()[("d3_review_summary", "d3_review_values").index(name)]
        if name in ("daily_series_summary", "daily_series_values", "daily_series_lags"):
            return self.daily_series()[("daily_series_summary", "daily_series_values", "daily_series_lags").index(name)]
        if name in ("two_day_summary", "two_day_values", "two_day_lags"):
            return self.two_days()[("two_day_summary", "two_day_values", "two_day_lags").index(name)]
        if name == "example_lags":
            return self.example_lags()
        if name == "example_diurnal":
            return self.example_diurnal()[0]
        if name == "example_diurnal_values":
            return self.example_diurnal()[1]
        if name == "example_trend":
            return self.example_trend()
        if name == "coverage":
            return self.day_coverage()
        if name == "exclusions":
            work = self.working_rows()
            return pd.DataFrame([
                {"level": level, "original_rows": int(work.sensorid.eq(sid).sum()),
                 "excluded_rows": int(work.loc[work.sensorid.eq(sid), "excluded_stuck"].sum()),
                 "remaining_rows": int((work.sensorid.eq(sid) & ~work.excluded_stuck).sum())}
                for level, sid in self.sensor_ids.items()
            ])
        return self.audit[name].copy()

    def working_rows(self):
        """Копия всех строк с признаком исключения и отдельным рабочим значением."""
        work = self.raw.copy(deep=True)
        excluded = pd.Series(False, index=work.index)
        for start, end in STUCK_INTERVALS:
            excluded |= work.localdatetime.between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")
        work["excluded_stuck"] = excluded
        work["datavalue_for_analysis"] = work.datavalue.mask(excluded)
        return work

    def example_trend(self, date=EXAMPLE_DAY):
        """Среднее и отклонения выбранных суток; контекст — соседние сутки."""
        date = pd.Timestamp(date).normalize()
        work = self.working_rows()
        context = work.loc[work.localdatetime.ge(date - pd.Timedelta(days=1))
                           & work.localdatetime.lt(date + pd.Timedelta(days=2))]
        frames = []
        with _column_settings(self.manifest):
            for level, sid in self.sensor_ids.items():
                rows = context.loc[context.sensorid.eq(sid)].copy()
                if rows.localdatetime.duplicated().any():
                    raise ValueError(f"{level}: в контексте примера есть повторные метки; нужно отдельное решение.")
                rows["datavalue"] = rows.datavalue_for_analysis
                derived = column.daily_trend(rows)
                day = derived.loc[derived.localdatetime.ge(date)
                                  & derived.localdatetime.lt(date + pd.Timedelta(days=1))].copy()
                if day.empty or not np.isfinite(day[["datavalue", "trend_24h", "residual"]].to_numpy()).all():
                    raise ValueError(f"{level}: для примера не хватает исходных значений или полных окон.")
                day["level"] = level
                frames.append(day[["level", "localdatetime", "sensorid", "datavalue", "trend_24h", "residual"]])
        return pd.concat(frames, ignore_index=True)

    def save_example_trend(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "november_2017_example_trend.csv"
        self.example_trend().to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        return path

    def example_diurnal(self, date=EXAMPLE_DAY):
        """МНК одной гармоники за выбранные сутки; по умолчанию — 16 ноября."""
        date = pd.Timestamp(date).normalize()
        data = self.example_trend(date)
        parameters, predictions = [], []
        for level in self.sensor_ids:
            part = data.loc[data.level.eq(level)].copy()
            fit = fit_diurnal(part.localdatetime, part.residual)
            parameters.append({"date": date, "level": level, **fit})
            hours = (part.localdatetime - part.localdatetime.dt.normalize()).dt.total_seconds() / 3600.
            angle = 2 * np.pi * hours / PERIOD_HOURS
            part["fitted_residual"] = (fit["offset_ppm"] + fit["cosine_ppm"] * np.cos(angle)
                                       + fit["sine_ppm"] * np.sin(angle))
            part["fit_error"] = part.residual - part.fitted_residual
            predictions.append(part)
        return pd.DataFrame(parameters), pd.concat(predictions, ignore_index=True)

    def save_example_diurnal(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        parameters, values = self.example_diurnal()
        paths = []
        for name, frame in (("summary", parameters), ("values", values)):
            path = directory / f"november_2017_example_diurnal_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            paths.append(path)
        return paths

    def _plot_example_diurnal(self):
        parameters, data = self.example_diurnal()
        parameters = parameters.set_index("level")
        fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.5), sharex=True, sharey=True)
        model_hours = np.linspace(0, PERIOD_HOURS, 721)
        end = EXAMPLE_DAY + pd.Timedelta(days=1)
        model_times = EXAMPLE_DAY + pd.to_timedelta(model_hours, unit="h")
        bound = float(data.residual.abs().max())
        for ax, level in zip(axes, self.sensor_ids):
            part, fit = data.loc[data.level.eq(level)], parameters.loc[level]
            model = (fit.offset_ppm + fit.cosine_ppm * np.cos(2 * np.pi * model_hours / PERIOD_HOURS)
                     + fit.sine_ppm * np.sin(2 * np.pi * model_hours / PERIOD_HOURS))
            bound = max(bound, float(np.abs(model).max()))
            ax.plot(part.localdatetime, part.residual, color="#8D959D", linewidth=.9,
                    label="Отклонения от среднего")
            ax.plot(model_times, model, color=COLORS[level], linewidth=2.2,
                    label="Модель с периодом 24 часа")
            if np.isfinite(fit.peak_hour_local):
                peak = EXAMPLE_DAY + pd.Timedelta(hours=fit.peak_hour_local)
                ax.axvline(peak, color=COLORS[level], linestyle="--", linewidth=1.0, alpha=.8)
                ax.plot(peak, fit.offset_ppm + fit.amplitude_ppm, "o", color=COLORS[level], markersize=4)
            label = (f"{level}   |   A = {fit.amplitude_ppm:.2f} ppm   |   "
                     f"Максимум: {fit.peak_time_local}   |   R² = {fit.r_squared:.2f}")
            ax.set_title(label.replace('.', ','), loc="left", fontsize=12, pad=9)
            ax.axhline(0, color="#555555", linewidth=.7, alpha=.5)
            ax.set_ylabel("Отклонение, ppm", fontsize=11)
            ax.grid(True, alpha=.2)
            ax.tick_params(labelsize=10)
        axes[0].set_ylim(-1.1 * max(bound, 1), 1.1 * max(bound, 1))
        axes[-1].set_xlim(EXAMPLE_DAY, end)
        axes[-1].set_xticks(pd.date_range(EXAMPLE_DAY, end, freq="4h"),
                            [f"{hour:02d}:00" for hour in range(0, 25, 4)])
        axes[-1].set_xlabel("16 ноября 2017, местное время базы", fontsize=11)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94), ncol=2, frameon=False, fontsize=10)
        fig.suptitle("W_R4_C-4 · одна 24-часовая гармоника за 16 ноября", fontsize=14, y=.995)
        fig.text(.5, .018, "Пунктир отмечает максимум модели. По 288 отсчётов на датчик.\n"
                 "Период 24 часа задан заранее; время до минуты — округление, не оценка точности.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, .88))
        return fig

    def example_lags(self, date=EXAMPLE_DAY):
        """Сдвиги максимумов выбранных суток на прежней ветви [-12, 12) часов."""
        date = pd.Timestamp(date).normalize()
        parameters = self.example_diurnal(date)[0].set_index("level")
        rows = []
        for first, second in PAIRS:
            a, b = parameters.loc[first], parameters.loc[second]
            difference = float(b.peak_hour_local - a.peak_hour_local)
            rows.append({
                "date": date, "pair": f"{first} → {second}",
                "first_level": first, "second_level": second,
                "first_peak_hour_local": a.peak_hour_local,
                "second_peak_hour_local": b.peak_hour_local,
                "first_peak_time_local": a.peak_time_local,
                "second_peak_time_local": b.peak_time_local,
                "raw_difference_hours": difference,
                "lag_hours": float(wrap_hours(difference)),
                "first_r_squared": a.r_squared, "second_r_squared": b.r_squared,
            })
        return pd.DataFrame(rows)

    def save_example_lags(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "november_2017_example_lags.csv"
        self.example_lags().to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def _plot_example_lags(self):
        data = self.example_lags()
        fig, ax = plt.subplots(figsize=(10.5, 3.8))
        for y, row in enumerate(data.itertuples()):
            start = row.first_peak_hour_local
            # Эквивалентный максимум второго сигнала на выбранной ветви суток.
            end = start + row.lag_hours
            if not np.isfinite([start, end]).all():
                ax.text(.5, y, "Сдвиг не определён", transform=ax.get_yaxis_transform(), ha="center")
                continue
            ax.annotate("", xy=(end, y), xytext=(start, y),
                        arrowprops={"arrowstyle": "->", "color": "#536575", "lw": 1.8,
                                    "shrinkA": 6, "shrinkB": 6})
            for hour, level, label in ((start, row.first_level, row.first_peak_time_local),
                                       (end, row.second_level, row.second_peak_time_local)):
                ax.plot(hour, y, "o", color=COLORS[level], markersize=7, zorder=3)
                ax.text(hour, y-.2, f"{level} · {label}", ha="center", fontsize=11, color=COLORS[level])
            ax.text((start+end)/2, y+.18, f"{row.lag_hours:+.2f} ч".replace('.', ','),
                    ha="center", va="top", fontsize=12, fontweight="bold")
        ax.set_yticks(range(len(data)), data.pair, fontsize=11)
        ax.set_ylim(len(data)-.35, -.55)
        ax.set_xlim(12, 24)
        ax.set_xticks(range(12, 25, 2), [f"{hour:02d}:00" for hour in range(12, 25, 2)])
        ax.set_xlabel("16 ноября 2017, местное время базы", fontsize=11)
        ax.grid(axis="x", alpha=.2)
        ax.tick_params(axis="y", length=0, pad=12)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title("W_R4_C-4 · сдвиги максимумов 24-часовой модели", fontsize=14, pad=20)
        fig.text(.5, .025, "Каждая строка сравнивает два модельных максимума. Знак + означает более поздний второй максимум.\n"
                 "Разности вычислены до округления времени; погрешность сдвигов пока не оценена.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .15, 1, 1))
        return fig

    def two_days(self):
        """Прежний расчёт отдельно для 16 и 17 ноября, без объединённой подгонки."""
        return self._analyze_days(COMPARISON_DAYS)

    def daily_series(self):
        """Девять отдельных суточных подгонок за 16–24 ноября, без фильтра по R²."""
        return self._analyze_days(DAILY_DATES)

    def november_selection(self):
        """Все сутки ноября: прежнее покрытие и запрет повторов в 3-суточном контексте.

        Консервативное правило example_trend сохранено, включая контекст вне
        фактически используемого окна ±12 ч. Нет нового отбора по R²/амплитуде.
        """
        coverage = self.day_coverage()
        rows = []
        for day in coverage.itertuples():
            context = self.raw.loc[self.raw.localdatetime.ge(day.date-pd.Timedelta(days=1))
                                   & self.raw.localdatetime.lt(day.date+pd.Timedelta(days=2))]
            repeats = int(context.duplicated(["sensorid", "localdatetime"]).sum())
            included = bool(day.common_window and repeats == 0)
            reason = ("included" if included else "incomplete_window" if not day.common_window
                      else "duplicates_in_context")
            rows.append(dict(date=day.date, full_window=day.common_window,
                             duplicate_extra_rows_context=repeats, included=included, reason=reason))
        return pd.DataFrame(rows)

    def month_comparison(self):
        """Сопоставимые суточные оценки, не две общие месячные гармоники."""
        selection = self.november_selection()
        november, _, nov_lags = self._analyze_days(selection.loc[selection.included, "date"])
        # Историческое расхождение хешей кода описано в details; результаты
        # обязательно сверяются с эталонами, другие предупреждения не скрываются.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="После исходной версии главы изменены модули:", category=UserWarning)
            october = load_october()
        october.verify()
        summary = pd.concat([october.result.daily.assign(month="October"),
                             november.assign(month="November")], ignore_index=True)
        lags = pd.concat([october.result.daily_lags.assign(month="October"),
                         nov_lags[["date", "pair", "lag_hours"]].assign(month="November")], ignore_index=True)
        rows = []
        for month in ("October", "November"):
            for level in self.sensor_ids:
                part = summary.loc[summary.month.eq(month) & summary.level.eq(level)]
                for metric in ("amplitude_ppm", "r_squared"):
                    rows.append(dict(month=month, series=level, metric=metric, n_days=len(part),
                                     median=float(part[metric].median())))
            for first, second in PAIRS:
                pair = f"{first} → {second}"
                part = lags.loc[lags.month.eq(month) & lags.pair.eq(pair)]
                rows.append(dict(month=month, series=pair, metric="lag_hours", n_days=len(part),
                                 median=float(part.lag_hours.median())))
        return summary, lags, pd.DataFrame(rows)

    def save_month_comparison(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        frames = dict(zip(("summary", "lags", "medians"), self.month_comparison()))
        frames["selection"] = self.november_selection()
        paths = []
        for name, frame in frames.items():
            path = directory / f"october_november_2017_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d")
            paths.append(path)
        return paths

    def _plot_month_comparison(self):
        summary, lags, _ = self.month_comparison()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
        for level in self.sensor_ids:
            part = summary.loc[summary.level.eq(level)]
            axes[0, 0].scatter(part.date, part.amplitude_ppm, s=23, color=COLORS[level], label=level)
            axes[1, 1].scatter(part.date, part.r_squared, s=23, color=COLORS[level], label=level)
        for ax, pair in ((axes[0, 1], "D1 → D2"), (axes[1, 0], "D2 → D3")):
            part = lags.loc[lags.pair.eq(pair)]
            ax.scatter(part.date, part.lag_hours, color="#536575", s=23)
        for ax, title, ylabel in (
            (axes[0, 0], "Амплитуды", "A, ppm"),
            (axes[0, 1], "Сдвиг D1 → D2", "Часы"),
            (axes[1, 0], "Сдвиг D2 → D3", "Часы"),
            (axes[1, 1], "Качество суточной подгонки", "R²"),
        ):
            ax.axvspan(pd.Timestamp("2017-11-01"), pd.Timestamp("2017-12-01"), color="#EEF2F5", zorder=0)
            ax.axvline(pd.Timestamp("2017-11-01"), color="#ADB5BD", linewidth=.8)
            ax.set(title=title, ylabel=ylabel, xlim=(pd.Timestamp("2017-10-01"), pd.Timestamp("2017-12-01")))
            ax.grid(alpha=.18)
            ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 10, 20]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        axes[0, 0].set_ylim(bottom=0)
        axes[1, 1].set_ylim(0, 1)
        axes[0, 0].legend(ncol=3, fontsize=10)
        axes[1, 1].legend(ncol=3, fontsize=10)
        # Сохраняю слабый день; кольцо — отметка выполненной диагностики, не фильтр.
        flagged = summary.loc[summary.level.eq("D3") & summary.date.eq(pd.Timestamp("2017-11-22"))]
        axes[1, 1].scatter(flagged.date, flagged.r_squared, s=95, facecolors="none", edgecolors="#C23B40", linewidths=1.3)
        for ax in axes[-1]:
            ax.set_xlabel("Дата 2017 года")
        fig.suptitle("W_R4_C-4 · октябрь и ноябрь: одинаковые суточные оценки", fontsize=14)
        fig.text(.5, .02, "Октябрь: 22 дня. Ноябрь: 10 дней (16–24, 28). Каждая точка — сутки; пропуски не соединены.\n"
                 "Серый фон — ноябрь. Красное кольцо — D3 22 ноября: чувствительная фаза; день сохранён.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .1, 1, .95))
        return fig

    def _analyze_days(self, dates):
        summaries, values, lags = [], [], []
        for date in dates:
            summary, fitted = self.example_diurnal(date)
            summaries.append(summary)
            values.append(fitted.assign(date=date))
            lags.append(self.example_lags(date))
        return tuple(pd.concat(frames, ignore_index=True) for frames in (summaries, values, lags))

    def save_daily_series(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in zip(("summary", "values", "lags"), self.daily_series()):
            path = directory / f"november_2017_daily_16_24_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            paths.append(path)
        return paths

    def d3_review(self):
        """D3 за 21–23 ноября: прежняя модель и неописанная часть, без новой фильтрации."""
        summaries, values = [], []
        for date in pd.date_range("2017-11-21", "2017-11-23"):
            summary, fitted = self.example_diurnal(date)
            part = fitted.loc[fitted.level.eq("D3")].copy()
            row = summary.loc[summary.level.eq("D3")].iloc[0].to_dict()
            row["unexplained_rms_ppm"] = float(np.sqrt(np.mean(part.fit_error.to_numpy()**2)))
            row["residual_std_ppm"] = float(part.residual.std(ddof=0))
            summaries.append(row)
            values.append(part.assign(date=date))
        return pd.DataFrame(summaries), pd.concat(values, ignore_index=True)

    def save_d3_review(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in zip(("summary", "values"), self.d3_review()):
            path = directory / f"november_2017_d3_21_23_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            paths.append(path)
        return paths

    def phase_sensitivity(self):
        """D3 22 ноября: 12 исключений [2k, 2k+2) при фиксированном тренде.

        Исходные данные не меняются. R² относится к оставшимся точкам каждой
        подгонки; его нельзя считать оценкой качества на исключённом блоке.
        Диапазон максимумов — чувствительность, не доверительный интервал.
        """
        date = pd.Timestamp("2017-11-22")
        parameters, all_values = self.example_diurnal(date)
        baseline = parameters.loc[parameters.level.eq("D3")].iloc[0].to_dict()
        data = all_values.loc[all_values.level.eq("D3")].copy()
        hours = (data.localdatetime - date).dt.total_seconds() / 3600
        rows = [dict(**baseline, omitted_block="none", omitted_start_hour=np.nan,
                     omitted_end_hour=np.nan, omitted_count=0, peak_shift_hours=0.0)]
        values = []
        for start in range(0, 24, 2):
            omitted = hours.ge(start) & hours.lt(start + 2)
            kept = data.loc[~omitted]
            fit = fit_diurnal(kept.localdatetime, kept.residual)
            label = f"{start:02d}–{start+2:02d}"
            rows.append(dict(date=date, level="D3", **fit, omitted_block=label,
                             omitted_start_hour=start, omitted_end_hour=start+2,
                             omitted_count=int(omitted.sum()),
                             peak_shift_hours=float(wrap_hours(fit["peak_hour_local"]
                                                               - baseline["peak_hour_local"]))))
            part = data.copy()
            part["omitted_block"] = label
            part["used_in_fit"] = ~omitted
            part["refitted_residual"] = (fit["offset_ppm"]
                + fit["cosine_ppm"] * np.cos(2*np.pi*hours/PERIOD_HOURS)
                + fit["sine_ppm"] * np.sin(2*np.pi*hours/PERIOD_HOURS))
            values.append(part)
        return pd.DataFrame(rows), pd.concat(values, ignore_index=True)

    def save_phase_sensitivity(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in zip(("summary", "values"), self.phase_sensitivity()):
            path = directory / f"november_2017_d3_phase_sensitivity_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            paths.append(path)
        return paths

    def _plot_phase_sensitivity(self):
        summary, _ = self.phase_sensitivity()
        base, trials = summary.iloc[0], summary.iloc[1:]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.3), sharey=True)
        y = np.arange(len(trials))
        # Развёртка относительно базовой фазы не создаёт скачка при переходе 00:00.
        peaks = base.peak_hour_local + trials.peak_shift_hours.to_numpy()
        for ax, values, reference, title, xlabel in (
            (axes[0], peaks, base.peak_hour_local, "Время максимума модели", "Местное время суток"),
            (axes[1], trials.amplitude_ppm, base.amplitude_ppm, "Амплитуда модели", "A, ppm"),
        ):
            ax.hlines(y, reference, values, color="#B6BEC6", linewidth=1.5)
            ax.scatter(values, y, color=COLORS["D3"], s=38, zorder=3)
            ax.axvline(reference, color="#536575", linestyle="--", label="Все 288 точек")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.grid(axis="x", alpha=.25)
            ax.legend(loc="lower right", fontsize=10)
        axes[0].set_yticks(y, trials.omitted_block)
        axes[0].set_ylabel("Исключённый из подгонки интервал, часы", fontsize=11)
        axes[0].invert_yaxis()
        from matplotlib.ticker import FuncFormatter
        from diurnal_analysis import clock_text
        axes[0].xaxis.set_major_formatter(FuncFormatter(lambda h, pos: clock_text(h)))
        fig.suptitle("W_R4_C-4 · D3 · 22 ноября: чувствительность к двухчасовым фрагментам", fontsize=14)
        fig.text(.5, .025, "Каждая точка — отдельная подгонка по 264 показаниям; среднее за 24 часа фиксировано.\n"
                 "Пунктир — исходная подгонка. Диапазон результатов не является доверительным интервалом.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .11, 1, .94))
        return fig

    def _plot_d3_review(self):
        summary, values = self.d3_review()
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.4), sharex=True, sharey=True)
        model_hours = np.linspace(0, PERIOD_HOURS, 721)
        bound = float(values[["residual", "fit_error"]].abs().max().max())
        for (left, right), fit in zip(axes, summary.itertuples()):
            part = values.loc[values.date.eq(fit.date)]
            hours = (part.localdatetime-fit.date).dt.total_seconds()/3600
            model = (fit.offset_ppm + fit.cosine_ppm*np.cos(2*np.pi*model_hours/PERIOD_HOURS)
                     + fit.sine_ppm*np.sin(2*np.pi*model_hours/PERIOD_HOURS))
            bound = max(bound, float(np.abs(model).max()))
            left.plot(hours, part.residual, color="#8D959D", linewidth=.9, label="Отклонения от среднего")
            left.plot(model_hours, model, color=COLORS["D3"], linewidth=2, label="24-часовая модель")
            if np.isfinite(fit.peak_hour_local):
                left.axvline(fit.peak_hour_local, color=COLORS["D3"], linestyle="--", linewidth=.9)
            label = (f"{fit.date.day} ноября   A={fit.amplitude_ppm:.2f} ppm   Макс.: {fit.peak_time_local}   R²={fit.r_squared:.2f}")
            left.set_title(label.replace('.', ','), fontsize=10.5, loc="left", pad=9)
            right.plot(hours, part.fit_error, color="#536575", linewidth=.9)
            right.set_title(f"{fit.date.day} ноября · после вычитания модели", fontsize=11, loc="left", pad=9)
            left.set_ylabel("r, ppm", fontsize=11)
            right.set_ylabel("e, ppm", fontsize=11)
            for ax in (left, right):
                ax.axhline(0, color="#555555", linewidth=.7, alpha=.6)
                ax.grid(True, alpha=.2)
                ax.tick_params(labelsize=10)
        axes[0, 0].set_ylim(-1.1*max(bound, 1), 1.1*max(bound, 1))
        axes[0, 0].set_xlim(0, 24)
        axes[0, 0].set_xticks(range(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
        for ax in axes[-1]:
            ax.set_xlabel("Время суток, местное время базы", fontsize=11)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False, fontsize=10)
        fig.suptitle("W_R4_C-4 · D3: 22 ноября и соседние дни", fontsize=14, y=.995)
        fig.text(.5, .018, "Слева: r = показание − среднее за 24 часа. Справа: e = r − модель.\n"
                 "Шкалы одинаковы. Неописанная часть сохранена; её частоты и физическая причина здесь не определяются.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .09, 1, .90))
        return fig

    def _plot_daily_series(self):
        summary, _, lags = self.daily_series()
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True)
        for level in self.sensor_ids:
            part = summary.loc[summary.level.eq(level)]
            for ax, key in ((axes[0, 0], "amplitude_ppm"), (axes[0, 1], "peak_hour_local"),
                            (axes[1, 1], "r_squared")):
                ax.plot(part.date, part[key], "o-", color=COLORS[level], label=level,
                        markersize=4, linewidth=1.3)
        for (first, second), color in zip(PAIRS, COLORS.values()):
            pair = f"{first} → {second}"
            part = lags.loc[lags.pair.eq(pair)]
            axes[1, 0].plot(part.date, part.lag_hours, "o-", color=color, label=pair,
                            markersize=4, linewidth=1.3)
        for ax, title, ylabel in (
            (axes[0, 0], "Амплитуда 24-часовой модели", "A, ppm"),
            (axes[0, 1], "Время максимума модели", "Местное время суток"),
            (axes[1, 0], "Сдвиги между максимумами", "Сдвиг, часы"),
            (axes[1, 1], "Доля описанного разброса", "R²"),
        ):
            ax.set_title(title, fontsize=12, pad=12)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=.2)
            ax.tick_params(labelsize=10)
            ax.legend(loc="best", fontsize=9, framealpha=.85)
            ax.set_xticks(DAILY_DATES)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
            ax.set_xlim(DAILY_DATES[0]-pd.Timedelta(hours=9), DAILY_DATES[-1]+pd.Timedelta(hours=9))
        axes[0, 0].set_ylim(bottom=0)
        axes[0, 1].set_ylim(12, 24)
        axes[0, 1].set_yticks(range(12, 25, 2), [f"{hour:02d}:00" for hour in range(12, 25, 2)])
        axes[1, 0].set_ylim(bottom=0)
        axes[1, 1].set_ylim(0, 1)
        for ax in axes[-1]:
            ax.set_xlabel("День ноября 2017", fontsize=11)
        fig.suptitle("W_R4_C-4 · суточные оценки за 16–24 ноября", fontsize=14, y=.99)
        fig.text(.5, .025, "Точки — отдельные сутки; линии помогают проследить последовательность.\n"
                 "Все 9 дней сохранены. Сдвиги — оценки одной 24-часовой компоненты, без интервалов неопределённости.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .09, 1, .95))
        return fig

    def save_two_days(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for name, frame in zip(("summary", "values", "lags"), self.two_days()):
            path = directory / f"november_2017_two_days_16_17_{name}.csv"
            frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
            paths.append(path)
        return paths

    def _plot_two_days(self):
        summary, values, _ = self.two_days()
        parameters = summary.set_index(["date", "level"])
        fig, axes = plt.subplots(3, 2, figsize=(12, 7.4), sharex=True, sharey=True)
        model_hours = np.linspace(0, PERIOD_HOURS, 721)
        bound = float(values.residual.abs().max())
        for col, date in enumerate(COMPARISON_DAYS):
            for row, level in enumerate(self.sensor_ids):
                ax = axes[row, col]
                data = values.loc[values.date.eq(date) & values.level.eq(level)]
                fit = parameters.loc[(date, level)]
                hours = (data.localdatetime - date).dt.total_seconds() / 3600.
                model = (fit.offset_ppm + fit.cosine_ppm * np.cos(2*np.pi*model_hours/PERIOD_HOURS)
                         + fit.sine_ppm * np.sin(2*np.pi*model_hours/PERIOD_HOURS))
                bound = max(bound, float(np.abs(model).max()))
                ax.plot(hours, data.residual, color="#8D959D", linewidth=.7, label="Отклонения от среднего")
                ax.plot(model_hours, model, color=COLORS[level], linewidth=2, label="Модель с периодом 24 часа")
                if np.isfinite(fit.peak_hour_local):
                    ax.axvline(fit.peak_hour_local, color=COLORS[level], linestyle="--", linewidth=.9, alpha=.8)
                    ax.plot(fit.peak_hour_local, fit.offset_ppm+fit.amplitude_ppm, "o", color=COLORS[level], markersize=4)
                label = (f"{level}   A={fit.amplitude_ppm:.2f} ppm   Макс.: {fit.peak_time_local}   R²={fit.r_squared:.2f}")
                ax.set_title(label.replace('.', ','), loc="left", fontsize=10.5, pad=9)
                ax.axhline(0, color="#555555", linewidth=.7, alpha=.5)
                ax.grid(True, alpha=.2)
                ax.tick_params(labelsize=10)
                if col == 0:
                    ax.set_ylabel("Отклонение, ppm", fontsize=11)
        axes[0, 0].set_ylim(-1.1*max(bound, 1), 1.1*max(bound, 1))
        axes[0, 0].set_xlim(0, 24)
        axes[0, 0].set_xticks(range(0, 25, 4), [f"{h:02d}:00" for h in range(0, 25, 4)])
        for ax in axes[-1]:
            ax.set_xlabel("Время суток, местное время базы", fontsize=11)
        fig.suptitle("W_R4_C-4 · одинаковая модель для двух соседних суток", fontsize=14, y=.995)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False, fontsize=10)
        fig.tight_layout(rect=(0, .06, 1, .85))
        for col, date in enumerate(COMPARISON_DAYS):
            pos = axes[0, col].get_position()
            fig.text((pos.x0+pos.x1)/2, .855, f"{date.day} ноября 2017", ha="center", fontsize=13, fontweight="bold")
        fig.text(.5, .018, "Пунктир — максимум модели. Одинаковые шкалы; по 288 показаний на датчик за сутки.\n"
                 "Две отдельные подгонки с периодом 24 часа; интервалы неопределённости не оценены.", ha="center", fontsize=10)
        return fig

    def _plot_example_trend(self):
        data = self.example_trend()
        fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.1), sharex=True, sharey="col")
        for (left, right), level, color in zip(axes, self.sensor_ids, ["#2563A6", "#18856A", "#B4661B"]):
            part = data.loc[data.level.eq(level)]
            left.plot(part.localdatetime, part.datavalue, color=color, linewidth=1.0)
            left.plot(part.localdatetime, part.trend_24h, color="#202020", linewidth=2.0)
            right.plot(part.localdatetime, part.residual, color=color, linewidth=1.0)
            right.axhline(0, color="#555555", linewidth=1, linestyle="--")
            left.set_ylabel(f"{level}\nCO₂, ppm", fontsize=11)
            right.set_ylabel(f"{level}\nОтклонение, ppm", fontsize=11)
            for ax in (left, right):
                ax.grid(True, alpha=.22)
                ax.tick_params(labelsize=10)
        axes[0, 0].set_title("Показания и среднее за 24 часа", fontsize=13, pad=12)
        axes[0, 1].set_title("Показания − среднее", fontsize=13, pad=12)
        bound = float(data.residual.abs().max())
        axes[0, 1].set_ylim(-max(1., 1.08 * bound), max(1., 1.08 * bound))
        end = EXAMPLE_DAY + pd.Timedelta(days=1)
        axes[-1, 0].set_xlim(EXAMPLE_DAY, end)
        axes[-1, 0].set_xticks(pd.date_range(EXAMPLE_DAY, end, freq="4h"),
                              [f"{h:02d}:00" for h in range(0, 25, 4)])
        for ax in axes[-1]:
            ax.set_xlabel("16 ноября 2017, местное время базы", fontsize=11)
        fig.suptitle("W_R4_C-4 · среднее и отклонения за 16 ноября", fontsize=15, y=.99)
        fig.text(.5, .025, "Слева: цвет — показания, чёрная линия — среднее в окне ±12 часов.\n"
                 "Справа: отклонения от этого среднего. Интерполяции нет.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .08, 1, .95))
        return fig

    def day_coverage(self):
        """Покрытие суток у трёх уровней: без вычисления среднего, фазы или весов."""
        work = self.working_rows()
        days = pd.date_range(self.manifest["start_inclusive"], self.manifest["end_exclusive"],
                             freq="D", inclusive="left")
        coverage = pd.DataFrame({"date": days})
        with _column_settings(self.manifest):
            for level, sid in self.sensor_ids.items():
                rows = work.loc[work.sensorid.eq(sid)].sort_values("localdatetime", kind="stable")
                rows = rows.assign(datavalue=rows.datavalue_for_analysis)
                masks = {"records": np.isfinite(rows.datavalue.to_numpy()),
                         "window": column.trend_window_mask(rows).to_numpy()}
                for kind, valid in masks.items():
                    intervals = observation_intervals(rows.localdatetime, valid, self.manifest["gap_factor"])
                    coverage[f"{level}_{kind}"] = [
                        any(start <= day and day + pd.Timedelta(days=1) <= end for start, end in intervals)
                        for day in days
                    ]
        for kind in ("records", "window"):
            coverage[f"common_{kind}"] = coverage[[f"{level}_{kind}" for level in self.sensor_ids]].all(axis=1)
        return coverage

    def save_coverage(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "november_2017_day_coverage.csv"
        self.day_coverage().to_csv(path, index=False, date_format="%Y-%m-%d")
        return path

    def _plot_coverage(self):
        coverage = self.day_coverage()
        fig, ax = plt.subplots(figsize=(12, 4.1))
        present, missing = "#237B68", "#E5E7EB"
        rows = [("common_records", "Записи после\nисключения залипаний"),
                ("common_window", "Данные для окна\n±12 часов")]
        for y, (key, _) in enumerate(rows):
            for x, item in coverage.iterrows():
                full = bool(item[key])
                ax.barh(y, .94, left=x-.47, height=.78, color=present if full else missing, edgecolor="white")
                ax.text(x, y, str(item.date.day), ha="center", va="center",
                        color="white" if full else "#475569", fontsize=10)
            ax.text(len(coverage)-.1, y, f"{coverage[key].sum()} / {len(coverage)}", va="center", fontsize=11)
        ax.set_yticks([0, 1], [item[1] for item in rows], fontsize=11)
        ax.set_xlim(-.6, len(coverage)+2.4)
        ax.set_ylim(1.6, -.65)
        ax.set_xticks([])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("W_R4_C-4 · покрытие суток ноября 2017\nОбщее для D1, D2 и D3", fontsize=14, pad=18)
        ax.legend(handles=[Patch(facecolor=present, label="Полное покрытие суток"),
                           Patch(facecolor=missing, label="Покрытие неполное")],
                  loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, frameon=False, fontsize=11)
        fig.text(.5, .035, "Числа в клетках — дни ноября. Проверено наличие данных по времени;\n"
                 "решение по повторам и расчёт фаз ещё предстоят.", ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .15, 1, 1))
        return fig

    def save_working_data(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "november_2017_working.csv"
        self.working_rows().to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
        self.table("exclusions").to_csv(directory / "november_2017_exclusions.csv", index=False)
        return path

    def _plot_stuck_exclusion(self):
        work = self.working_rows()
        fig, axes = plt.subplots(3, 2, figsize=(12, 8.5), sharex=True, sharey=True)
        counts = self.table("exclusions").set_index("level")
        colors = ["#2563A6", "#18856A", "#B4661B"]
        with _column_settings(self.manifest):
            for (left, right), (level, sid), color in zip(axes, self.sensor_ids.items(), colors):
                rows = work.loc[work.sensorid.eq(sid)]
                original = column.line_with_gaps(rows)
                masked = column.line_with_gaps(rows.assign(datavalue=rows.datavalue_for_analysis))
                left.plot(original.index, original.to_numpy(), color=color, linewidth=.7)
                right.plot(masked.index, masked.to_numpy(), color=color, linewidth=.7)
                for start, end in STUCK_INTERVALS:
                    start, end = pd.Timestamp(start), pd.Timestamp(end)
                    selected = rows.loc[rows.localdatetime.between(start, end)]
                    left.plot(selected.localdatetime, selected.datavalue, color="#C63B3B", linewidth=1.8)
                    for ax in (left, right):
                        ax.axvspan(start, end, color="#C63B3B", alpha=.14, linewidth=0)
                for ax in (left, right):
                    ax.set_ylabel(f"{level}\nCO₂, ppm", fontsize=11)
                    ax.grid(True, alpha=.2)
                    ax.tick_params(labelsize=10)
                row = counts.loc[level]
                right.text(.98, .94, f"Исключено: {row.excluded_rows:,}; осталось: {row.remaining_rows:,}".replace(',', ' '),
                           transform=right.transAxes, ha="right", va="top", fontsize=9,
                           bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        axes[0, 0].set_title("Исходные записи", fontsize=13, pad=25)
        axes[0, 1].set_title("После исключения залипаний", fontsize=13, pad=25)
        for ax in axes[0]:
            for date, label in (("2017-11-13 12:00", "12–14.11"), ("2017-11-26 07:25", "26.11")):
                ax.text(pd.Timestamp(date), 1.02, label, color="#A62727", ha="center",
                        transform=ax.get_xaxis_transform(), fontsize=9)
        axes[-1, 0].set_xlim(pd.Timestamp(self.manifest["start_inclusive"]), pd.Timestamp(self.manifest["end_exclusive"]))
        axes[-1, 0].set_xticks(pd.to_datetime([f"2017-11-{day:02d}" for day in [1, 6, 11, 16, 21, 26, 30]]))
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        for ax in axes[-1]:
            ax.set_xlabel("Ноябрь 2017, местное время базы", fontsize=11)
        fig.suptitle("W_R4_C-4 · исключение двух периодов залипания", fontsize=15, y=.99)
        fig.text(.5, .02, "Красным выделены исключённые интервалы. Справа на их месте — пропуски.\n"
                 "Исходные записи сохранены; оставшиеся данные ещё требуют решения по повторным меткам.",
                 ha="center", fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, .95))
        return fig

    def figure(self, kind="raw"):
        if kind == "month_comparison":
            return self._plot_month_comparison()
        if kind == "phase_sensitivity":
            return self._plot_phase_sensitivity()
        if kind == "d3_review":
            return self._plot_d3_review()
        if kind == "daily_series":
            return self._plot_daily_series()
        if kind == "two_days":
            return self._plot_two_days()
        if kind == "example_lags":
            return self._plot_example_lags()
        if kind == "example_diurnal":
            return self._plot_example_diurnal()
        if kind == "example_trend":
            return self._plot_example_trend()
        if kind == "raw":
            with _column_settings(self.manifest), redirect_stdout(io.StringIO()):
                return _format_figure(column.plot_raw_timeseries(self.raw, self.sensor_ids), "raw")
        if kind == "stuck":
            return self._plot_stuck_exclusion()
        if kind == "coverage":
            return self._plot_coverage()
        if kind != "counts":
            raise ValueError(f"Допустимые рисунки: {', '.join(FIGURES)}")
        counts = self.audit["daily_counts"].pivot(index="date", columns="level", values="rows")
        if not counts.eq(counts["D1"], axis=0).all().all():
            raise ValueError("Суточные количества различаются: нужен рисунок для каждого датчика.")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(counts.index + pd.Timedelta(hours=12), counts.D1, width=.75, color="#2563A6")
        ax.axhline(288, color="#555555", linestyle="--", linewidth=1.2, label="288 = 24 × 60 / 5")
        ax.set(xlim=(pd.Timestamp(self.manifest["start_inclusive"]), pd.Timestamp(self.manifest["end_exclusive"])),
               ylim=(0, 590), xlabel="Ноябрь 2017, местное время базы", ylabel="Записей за сутки",
               title="Число записей за день совпадает у D1, D2 и D3")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
        ax.grid(axis="y", alpha=.22)
        ax.legend(loc="upper right")
        fig.text(.5, .02, "Все исходные записи. Пунктир — ориентир для ровного шага 5 минут, не число достоверных измерений.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .07, 1, 1))
        return fig

    def show(self, kind="raw"):
        from IPython.display import Image, display
        fig = self.figure(kind)
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=160)
            display(Image(data=buffer.getvalue()))
        finally:
            plt.close(fig)

    def save_figures(self, directory=HERE / "output/ch02"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for kind in FIGURES:
            fig = self.figure(kind)
            path = directory / f"november_2017_{kind}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path)
        return paths


def load_november(data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "provenance.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Изменился файл снимка: {name}")
    raw = pd.read_csv(data_dir / "measurements.csv", parse_dates=["localdatetime"])
    ids = {level: item["sensor_id"] for level, item in manifest["sensors"].items()}
    if set(raw.sensorid) != set(ids.values()):
        raise ValueError("Изменился состав датчиков.")
    if not (raw.localdatetime.ge(pd.Timestamp(manifest["start_inclusive"]))
            & raw.localdatetime.lt(pd.Timestamp(manifest["end_exclusive"]))).all():
        raise ValueError("Временные метки выходят за границы ноября.")
    audit = audit_raw(raw, ids, manifest["service_value_limit"], manifest["gap_factor"],
                      manifest["diagnostics"]["constant_run_minimum_hours"])
    report = NovemberReport(manifest, raw, ids, audit, data_dir)
    report.verify()
    return report


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        for path in open_saved_trajectories():
            print(path)
        raise SystemExit(0)
    report = load_november()
    print(report.table().to_string(index=False))
    print(report.table("constant_runs").to_string(index=False))
    print("Проверено диагностических таблиц:", report.verify())
    print(report.table("exclusions").to_string(index=False))
    print(report.table("coverage").to_string(index=False))
    for kind in FIGURES:
        report.figure(kind)
    plt.show()

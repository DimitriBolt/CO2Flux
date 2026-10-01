"""Суточная гармоника остатков: общий период наблюдений, фазы и дневной разброс.

Модель r(t) = c + a*cos(2*pi*h/24) + b*sin(2*pi*h/24), где h — местное
время суток. Обычный МНК использует исходные временные метки без интерполяции.
Амплитуда = hypot(a, b); максимум = atan2(b, a)*24/(2*pi) по модулю 24 ч.
Разность фаз выражается в часах на ветви [-12, 12); это не время переноса газа.
R² и междневный разброс — описательные показатели, не тест значимости.
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PERIOD_HOURS = 24.0
COLORS = {"D1": "#2563A6", "D2": "#18856A", "D3": "#B4661B"}
PAIRS = [("D1", "D2"), ("D2", "D3"), ("D1", "D3")]


def wrap_hours(hours):
    """Кратчайший знаковый сдвиг по модулю суток; + означает более поздний пик."""
    return (np.asarray(hours) + 12.0) % 24.0 - 12.0


def clock_text(hours: float) -> str:
    if not np.isfinite(hours):
        return "не определён"
    minutes = int(round(hours * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def fit_diurnal(timestamps, values) -> dict:
    times = pd.DatetimeIndex(timestamps)
    y = np.asarray(values, dtype=float)
    if len(times) != len(y):
        raise ValueError("Количество временных меток и значений различается.")
    valid = ~times.isna() & np.isfinite(y)
    times, y = times[valid], y[valid]
    if len(y) < 3:
        raise ValueError("Для суточной гармоники недостаточно измерений.")
    hours = (times - times.normalize()).total_seconds().to_numpy() / 3600.0
    angle = 2 * np.pi * hours / PERIOD_HOURS
    design = np.column_stack([np.ones(len(y)), np.cos(angle), np.sin(angle)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != 3:
        raise ValueError("Времена наблюдений не позволяют определить суточную гармонику.")
    offset, cosine, sine = coefficients
    amplitude = float(np.hypot(cosine, sine))
    # Только численная защита для постоянного/нулевого сигнала, не критерий значимости.
    numerical_floor = 1e-10 * max(1.0, float(np.max(np.abs(y))))
    peak = float(np.arctan2(sine, cosine) * 24 / (2 * np.pi) % 24)
    if amplitude <= numerical_floor:
        peak = float("nan")
    sst = float(np.sum((y - y.mean()) ** 2))
    sse = float(np.sum((y - design @ coefficients) ** 2))
    r2 = 1 - sse / sst if sst > len(y) * numerical_floor**2 else float("nan")
    return {
        "n_observations": len(y),
        "offset_ppm": float(offset),
        "cosine_ppm": float(cosine),
        "sine_ppm": float(sine),
        "amplitude_ppm": amplitude,
        "peak_hour_local": peak,
        "peak_time_local": clock_text(peak),
        "r_squared": r2,
    }


def observation_intervals(timestamps, valid, gap_factor=1.5) -> list:
    """Границы покрытия по времени; концентрации и повторы не обрабатываются."""
    times = pd.Series(pd.DatetimeIndex(timestamps))
    if not times.is_monotonic_increasing or times.isna().any():
        raise ValueError("Для покрытия нужны упорядоченные временные метки без NaT.")
    steps = times.diff()
    positive = steps[steps > pd.Timedelta(0)]
    if positive.empty:
        return []
    valid_times = times[np.asarray(valid, dtype=bool)]
    if valid_times.empty:
        return []
    groups = (valid_times.diff() > gap_factor * positive.median()).cumsum()
    return [(part.iloc[0], part.iloc[-1]) for _, part in valid_times.groupby(groups)]


def _supported_intervals(frame: pd.DataFrame, gap_factor: float) -> list:
    """Непрерывные участки с рассчитанными остатками; шаг — из исходных дат."""
    times = frame["localdatetime"]
    if times.duplicated().any():
        raise ValueError("Повторные временные метки требуют отдельного решения перед анализом.")
    return observation_intervals(times, np.isfinite(frame["residual"].to_numpy()), gap_factor)


@dataclass
class DiurnalResult:
    summary: pd.DataFrame
    lags: pd.DataFrame
    daily: pd.DataFrame
    daily_lags: pd.DataFrame
    day_coverage: pd.DataFrame
    common_intervals: list
    sensor_ids: dict[str, int] = field(default_factory=dict)
    start: pd.Timestamp | None = None
    end_exclusive: pd.Timestamp | None = None

    def _daily_vectors(self, value: str) -> pd.DataFrame:
        """Строка — пригодные сутки, столбцы — три фактических уровня по порядку."""
        vectors = self.daily.pivot(index="date", columns="level", values=value)
        vectors = vectors.reindex(columns=self.summary.index)
        vectors.index = pd.DatetimeIndex(vectors.index, name="date")
        return vectors

    @property
    def H(self) -> pd.DataFrame:
        """H(d): местные часы максимумов в [0, 24); неопределённая фаза — NaN.

        Это оценки модели, а не свидетельство надёжности фаз. R² и остальные
        параметры остаются в daily; нового отбора по качеству здесь нет.
        """
        return self._daily_vectors("peak_hour_local")

    @property
    def A(self) -> pd.DataFrame:
        """A(d): амплитуды выделенной 24-часовой гармоники в ppm, не c + A."""
        return self._daily_vectors("amplitude_ppm")


def analyze_diurnal(derived_by_level: dict[str, pd.DataFrame], gap_factor=1.5, daily_dates=None) -> DiurnalResult:
    """Пересечь интервалы трёх уровней и оценить гармоники прежним методом.

    Ключи и их порядок задают фактические уровни j=1,2,3; например, D1/D2/D4.
    Названия уровней не заменяются стандартными D1/D2/D3.
    daily_dates ограничивает только суточные подгонки; общий контекст и метод
    подгонки не меняются. None означает все полные сутки, как прежде.
    """
    requested = None if daily_dates is None else set(pd.DatetimeIndex(daily_dates).normalize())
    levels = tuple(derived_by_level)
    if len(levels) != 3 or any(not isinstance(level, str) or not level.strip() for level in levels):
        raise ValueError("Нужны рассчитанные остатки трёх различных именованных уровней.")
    frames = {
        level: frame.sort_values("localdatetime").copy()
        for level, frame in derived_by_level.items()
    }
    common = None
    for frame in frames.values():
        intervals = _supported_intervals(frame, gap_factor)
        if common is None:
            common = intervals
        else:
            common = [
                (max(a, c), min(b, d))
                for a, b in common for c, d in intervals
                if max(a, c) < min(b, d)
            ]
    if not common:
        raise ValueError("Нет общих допустимых временных участков трёх датчиков.")
    common = sorted(common)
    selected = {}
    summary_rows = []
    for level, frame in frames.items():
        mask = np.zeros(len(frame), dtype=bool)
        for start, end in common:
            mask |= frame["localdatetime"].between(start, end).to_numpy()
        selected[level] = frame.loc[mask & np.isfinite(frame["residual"].to_numpy())]
        fit = fit_diurnal(selected[level]["localdatetime"], selected[level]["residual"])
        summary_rows.append({"level": level, **fit})
    summary = pd.DataFrame(summary_rows).set_index("level")

    first = min(frame["localdatetime"].min() for frame in frames.values()).normalize()
    last = max(frame["localdatetime"].max() for frame in frames.values()).normalize()
    daily_rows, coverage_rows = [], []
    for day in pd.date_range(first, last, freq="D"):
        if requested is not None and day not in requested:
            continue
        next_day = day + pd.Timedelta(days=1)
        full = any(start <= day and next_day <= end for start, end in common)
        coverage_rows.append({"date": day, "included": full,
                              "reason": "полные общие сутки" if full else "неполное окно или пропуск"})
        if not full:
            continue
        for level, frame in selected.items():
            part = frame.loc[(frame["localdatetime"] >= day) & (frame["localdatetime"] < next_day)]
            daily_rows.append({"date": day, "level": level,
                               **fit_diurnal(part["localdatetime"], part["residual"])})
    daily = pd.DataFrame(daily_rows, columns=["date", "level", *summary.columns])
    for level in summary.index:
        peaks = daily.loc[daily["level"] == level, "peak_hour_local"].dropna().to_numpy(dtype=float)
        offsets = wrap_hours(peaks - summary.loc[level, "peak_hour_local"])
        offsets = offsets[np.isfinite(offsets)]
        summary.loc[level, "n_complete_days"] = sum(row["included"] for row in coverage_rows)
        summary.loc[level, "n_defined_daily_peaks"] = len(peaks)
        summary.loc[level, "daily_peak_iqr_hours"] = (
            float(np.quantile(offsets, .75) - np.quantile(offsets, .25)) if len(offsets) else np.nan
        )

    daily_lag_rows, lag_rows = [], []
    pairs = [(levels[0], levels[1]), (levels[1], levels[2]), (levels[0], levels[2])]
    for first_level, second_level in pairs:
        pair = f"{first_level} → {second_level}"
        monthly_lag = float(wrap_hours(summary.loc[second_level, "peak_hour_local"]
                                      - summary.loc[first_level, "peak_hour_local"]))
        pair_daily = []
        for day, part in daily.groupby("date"):
            peaks = part.set_index("level")["peak_hour_local"]
            lag = float(wrap_hours(peaks[second_level] - peaks[first_level]))
            daily_lag_rows.append({"date": day, "pair": pair, "lag_hours": lag})
            if np.isfinite(lag):
                pair_daily.append(lag)
        # Квантили на ветви вокруг месячной оценки, чтобы переход через ±12 ч
        # не создавал искусственного большого разброса. Это не доверительный интервал.
        unwrapped = monthly_lag + wrap_hours(np.asarray(pair_daily) - monthly_lag)
        q25, q75 = np.quantile(unwrapped, [.25, .75]) if len(unwrapped) else (np.nan, np.nan)
        lag_rows.append({"pair": pair, "lag_hours": monthly_lag,
                         "daily_q25_hours": q25, "daily_q75_hours": q75,
                         "n_complete_days": len(pair_daily)})
    return DiurnalResult(summary, pd.DataFrame(lag_rows), daily,
                         pd.DataFrame(daily_lag_rows, columns=["date", "pair", "lag_hours"]),
                         pd.DataFrame(coverage_rows), common)


def _number(value, digits=2) -> str:
    return f"{value:.{digits}f}" if np.isfinite(value) else "—"


def _table(axis, rows, labels, title):
    axis.axis("off")
    axis.set_title(title, loc="left", fontsize=11, pad=7)
    table = axis.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D2D8DF")
        if row == 0:
            cell.set_facecolor("#EAF0F6")
            cell.set_text_props(weight="bold")


def plot_diurnal(result: DiurnalResult, title: str):
    figure = plt.figure(figsize=(13, 10))
    grid = figure.add_gridspec(3, 1, height_ratios=[3.6, 1.1, 1.1], hspace=.48)
    axis = figure.add_subplot(grid[0])
    hours = np.linspace(0, 24, 481)
    for level, row in result.summary.iterrows():
        wave = row.cosine_ppm * np.cos(2 * np.pi * hours / 24) + row.sine_ppm * np.sin(2 * np.pi * hours / 24)
        axis.plot(hours, wave, color=COLORS[level], linewidth=2, label=level)
        if np.isfinite(row.peak_hour_local):
            axis.scatter(row.peak_hour_local, row.amplitude_ppm, color=COLORS[level], s=45, zorder=4)
            axis.annotate(f"{level}: {row.peak_time_local}", (row.peak_hour_local, row.amplitude_ppm),
                          xytext=(0, 12), textcoords="offset points", ha="center", color=COLORS[level])
    axis.axhline(0, color="#666666", linewidth=.8)
    axis.set_xlim(0, 24)
    axis.set_xticks(np.arange(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    axis.margins(y=.25)
    axis.set_xlabel("Местное время суток")
    axis.set_ylabel("Суточная составляющая, ppm")
    axis.grid(True, alpha=.22)
    axis.legend(loc="lower left", ncol=3)
    axis.set_title("Средняя суточная составляющая за выбранный месяц", fontsize=13)
    rows = [[level, _number(row.amplitude_ppm), row.peak_time_local, _number(row.r_squared),
             _number(row.daily_peak_iqr_hours)] for level, row in result.summary.iterrows()]
    _table(figure.add_subplot(grid[1]), rows,
           ["Датчик", "Амплитуда, ppm", "Максимум", "R²", "Дневной IQR пика, ч"],
           "Амплитуды и время максимумов (амплитуда — от среднего до пика)")
    rows = [[row.pair, _number(row.lag_hours),
             f"{_number(row.daily_q25_hours)} … {_number(row.daily_q75_hours)}"]
            for row in result.lags.itertuples()]
    _table(figure.add_subplot(grid[2]), rows,
           ["Пара", "Месячный сдвиг, ч", "Дневные сдвиги: Q25 … Q75, ч"],
           "Задержки суточной составляющей: плюс означает более поздний максимум второго датчика")
    days = int(result.day_coverage["included"].sum())
    figure.suptitle(title, fontsize=14, y=.98)
    figure.subplots_adjust(top=.90, bottom=.12, left=.10, right=.96)
    figure.text(.5, .028,
                f"Период фиксирован: 24 ч. Дневная проверка: {days} полных суток. IQR — ширина центральных 50% дневных оценок.\n"
                "Сдвиги определены по модулю 24 ч, месячная ветвь [−12, +12) ч. Дневные квантили — не доверительный интервал.\n"
                "R² показывает долю вариации остатков, описанную моделью. Причину задержки этот расчёт не устанавливает.",
                ha="center", fontsize=9)
    return figure


def plot_daily_stability(result: DiurnalResult, title: str):
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    display_peaks = []
    for level, row in result.summary.iterrows():
        part = result.daily.loc[result.daily["level"] == level]
        # Например, 00:05 рядом с месячным пиком 22:00 показываем как 24:05,
        # чтобы переход через полночь не выглядел сдвигом почти на сутки.
        peaks = row.peak_hour_local + wrap_hours(part["peak_hour_local"].to_numpy(dtype=float) - row.peak_hour_local)
        display_peaks.extend(peaks[np.isfinite(peaks)])
        axes[0].scatter(part["date"], peaks, color=COLORS[level], label=level, s=32)
        if np.isfinite(row.peak_hour_local):
            axes[0].axhline(row.peak_hour_local, color=COLORS[level], linestyle="--", alpha=.7)
    for pair, color in zip(result.lags["pair"], COLORS.values()):
        part = result.daily_lags.loc[result.daily_lags["pair"] == pair]
        axes[1].scatter(part["date"], part["lag_hours"], color=color, label=pair, s=32)
        monthly = result.lags.loc[result.lags["pair"] == pair, "lag_hours"].iloc[0]
        if np.isfinite(monthly):
            axes[1].axhline(monthly, color=color, linestyle="--", alpha=.7)
    for day in result.day_coverage.loc[~result.day_coverage["included"], "date"]:
        for axis in axes:
            axis.axvspan(day, day + pd.Timedelta(days=1), color="#B0B8C0", alpha=.14, linewidth=0)
    low = float(min(display_peaks)) - 1.2 if display_peaks else 0.0
    high = float(max(display_peaks)) + 1.2 if display_peaks else 24.0
    ticks = np.arange(np.floor(low / 3) * 3, np.ceil(high / 3) * 3 + 1, 3)
    axes[0].set_yticks(ticks, [f"{int(h):02d}:00" for h in ticks])
    axes[0].set_ylim(low, high)
    axes[0].set_ylabel("Время максимума")
    axes[1].set_ylabel("Задержка суточной\nсоставляющей, ч")
    axes[1].axhline(0, color="#555555", linewidth=.7)
    axes[1].set_ylim(-12, 12)
    for axis in axes:
        axis.grid(True, alpha=.22)
        axis.legend(loc="lower left", ncol=3)
    axes[0].legend(loc="upper right", ncol=3)
    first, last = result.day_coverage["date"].iloc[[0, -1]]
    axes[-1].set_xlim(first, last + pd.Timedelta(days=1))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[-1].set_xlabel("Дата")
    figure.suptitle(f"{title}\nПроверка по отдельным полным суткам", fontsize=14)
    figure.text(.5, .025,
                "Точки — оценки по отдельным суткам; пунктир — оценка по всему месяцу.\n"
                "Время пика показано около месячной оценки: 24:00 соответствует 00:00 следующих суток.\n"
                "Серым отмечены исключённые неполные сутки. Дневные задержки показаны на ветви [−12, +12) ч.",
                ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .12, 1, .92))
    return figure


def save_diurnal(result: DiurnalResult, output_dir: Path, stem: str, title: str) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {"diurnal_summary": result.summary.reset_index(), "diurnal_lags": result.lags,
              "diurnal_daily": result.daily, "diurnal_daily_lags": result.daily_lags,
              "diurnal_day_coverage": result.day_coverage,
              "diurnal_common_intervals": pd.DataFrame(result.common_intervals, columns=["start", "end"])}
    for suffix, table in tables.items():
        table.to_csv(output_dir / f"{stem}_{suffix}.csv", index=False, float_format="%.8f")
    figures = [plot_diurnal(result, title), plot_daily_stability(result, title)]
    for figure, suffix in zip(figures, ["diurnal24h", "diurnal_daily_stability"]):
        path = output_dir / f"{stem}_{suffix}.png"
        figure.savefig(path, dpi=160)
        print(f"Суточный анализ: {path}")
    print("\nСуточные составляющие:")
    print(result.summary[["amplitude_ppm", "peak_time_local", "r_squared", "daily_peak_iqr_hours"]].round(3).to_string())
    print("\nЗадержки суточной составляющей, ч (плюс = второй датчик позже):")
    print(result.lags.round(3).to_string(index=False))
    return figures

"""Временные ряды CO2, суточное среднее и фазовые отношения одной колонки.

Запускать этот файл в PyCharm или командой python3 plot_column_timeseries.py.
Параметры для изменения находятся ниже. Используется подключение из sensorDB.py.
Программа сохраняет в output/ исходные ряды, суточное среднее и отклонения,
суточные гармоники, таблицы их задержек и проверку по отдельным суткам.
Интерполяция не выполняется. Суточные задержки не переводятся в коэффициент диффузии.
"""

from collections.abc import Mapping
from datetime import datetime
from numbers import Integral
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from diurnal_analysis import DiurnalResult, analyze_diurnal, save_diurnal


# Параметры первого примера из раздела 7 отчёта Габитова.
COLUMN_NAME = "W_R4_C-4"
SENSOR_CODES = {
    "D1": "LEO-W_4_-4_1_GMM222",
    "D2": "LEO-W_4_-4_2_GMM222",
    "D3": "LEO-W_4_-4_3_GMM222",
}
START = datetime(2017, 10, 1)
END_EXCLUSIVE = datetime(2017, 11, 1)  # 1 ноября уже не входит в выбранный месяц.
SHOW_PLOT = True  # В PyCharm открыть окно с графиками после сохранения PNG.
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TREND_WINDOW = pd.Timedelta(hours=24)  # Центрированное окно: 12 ч до и 12 ч после.
GAP_FACTOR = 1.5  # Разрыв больше 1.5 обычного шага считаем пропуском.

# CO2 в базальте: variableid=9, как в CO2basalt.sql и BasaltCO2Series.
VARIABLE_ID = 9
ERROR_CODE_LIMIT = -9999  # -10000/-9999 — служебные значения, не концентрация.

SENSOR_QUERY = """
SELECT sensorid, sensorcode
FROM leo_west.sensors
WHERE sensorcode IN (:code_d1, :code_d2, :code_d3)
"""

DATA_QUERY = """
SELECT localdatetime, sensorid, datavalue
FROM leo_west.datavalues
WHERE localdatetime >= :start_datetime
  AND localdatetime < :end_datetime
  AND variableid = :variable_id
  AND sensorid IN (:sensor_d1, :sensor_d2, :sensor_d3)
ORDER BY localdatetime, sensorid
"""


def fetch_data() -> tuple[pd.DataFrame, dict[str, int]]:
    """Два SELECT: найти ID по кодам, затем получить один месяц для всех трёх ID."""
    # Отчёт по сохранённой выгрузке работает без Oracle и его настроек.
    from sensorDB import SensorDB

    if END_EXCLUSIVE <= START:
        raise ValueError("Конец периода должен быть позже начала.")
    if list(SENSOR_CODES) != ["D1", "D2", "D3"]:
        raise ValueError("В SENSOR_CODES должны быть три уровня: D1, D2, D3.")

    with SensorDB() as database:
        catalog = database.fetch_dataframe(
            SENSOR_QUERY,
            {f"code_{level.lower()}": code for level, code in SENSOR_CODES.items()},
        )
        sensor_ids = {}
        for level, code in SENSOR_CODES.items():
            matches = catalog.loc[catalog["sensorcode"].str.strip() == code, "sensorid"]
            if len(matches) != 1:
                raise ValueError(f"Для {code} ожидался один датчик; найдено: {len(matches)}.")
            sensor_ids[level] = int(matches.iloc[0])
        if len(set(sensor_ids.values())) != 3:
            raise ValueError("Три уровня должны соответствовать трём разным датчикам.")

        print("Датчики:", sensor_ids, flush=True)
        print(f"Получаю данные за {START:%Y-%m-%d} — {END_EXCLUSIVE:%Y-%m-%d} (не включая конец)...", flush=True)
        raw = database.fetch_dataframe(
            DATA_QUERY,
            {
                "start_datetime": START,
                "end_datetime": END_EXCLUSIVE,
                "variable_id": VARIABLE_ID,
                **{f"sensor_{level.lower()}": sid for level, sid in sensor_ids.items()},
            },
        )

    raw["localdatetime"] = pd.to_datetime(raw["localdatetime"])
    raw["datavalue"] = pd.to_numeric(raw["datavalue"], errors="raise")
    for level, sid in sensor_ids.items():
        if not raw["sensorid"].eq(sid).any():
            raise ValueError(f"За выбранный период нет измерений {level} (sensorid={sid}).")
    return raw, sensor_ids


def line_with_gaps(rows: pd.DataFrame) -> pd.Series:
    """Исходные точки для рисунка; NaN разрывает линию на ошибках и пропусках.

    Разрыв времени больше 1.5 медианного шага отмечается дополнительной пустой
    точкой между двумя измерениями. Их исходные даты и значения не изменяются.
    Это только оформление графика; в сохранённом CSV остаются все исходные строки.
    """
    rows = rows.sort_values("localdatetime")
    values = rows["datavalue"].mask(rows["datavalue"] <= ERROR_CODE_LIMIT)
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(rows["localdatetime"]))
    times = pd.Series(series.index)
    steps = times.diff()
    positive_steps = steps[steps > pd.Timedelta(0)]
    if positive_steps.empty:
        return series
    typical_step = positive_steps.median()
    gaps = steps > GAP_FACTOR * typical_step
    gap_times = times[gaps] - steps[gaps] / 2
    missing = pd.Series(float("nan"), index=pd.DatetimeIndex(gap_times))
    return pd.concat([series, missing]).sort_index()


def _trend_segments(rows: pd.DataFrame) -> list[pd.DataFrame]:
    """Общие непрерывные участки для проверки окна и вычисления среднего."""
    result = rows.sort_values("localdatetime")
    if result.empty:
        return []
    values = result["datavalue"].mask(result["datavalue"] <= ERROR_CODE_LIMIT)
    steps = result["localdatetime"].diff()
    positive_steps = steps[steps > pd.Timedelta(0)]
    if positive_steps.empty:
        return []
    segment_starts = (
        steps.gt(GAP_FACTOR * positive_steps.median())
        | values.isna()
        | values.shift().isna()
    )
    return [part for _, part in result.loc[values.notna()].groupby(segment_starts.cumsum(), sort=False)]


def _full_trend_window(part: pd.DataFrame):
    times = pd.DatetimeIndex(part["localdatetime"])
    half_window = TREND_WINDOW / 2
    return (times >= times[0] + half_window) & (times <= times[-1] - half_window)


def trend_window_mask(rows: pd.DataFrame) -> pd.Series:
    """Наличие полного окна ±12 ч; среднее и остатки не вычисляются.

    Повторные метки сохраняются. Этот признак проверяет только покрытие по времени,
    а не пригодность повторных значений для последующей подгонки модели.
    """
    supported = pd.Series(False, index=rows.index)
    for part in _trend_segments(rows):
        supported.loc[part.index] = _full_trend_window(part)
    return supported


def daily_trend(rows: pd.DataFrame) -> pd.DataFrame:
    """Среднее за 24 ч и отклонения, только для полных окон без пропусков.

    Среднее берётся по исходным отсчётам в [t - 12 ч, t + 12 ч).
    Временные метки сохраняются. Сначала ряд делится на непрерывные участки:
    ни разрыв времени, ни служебное значение не могут попасть внутрь окна.
    """
    result = rows.sort_values("localdatetime").copy()
    result["trend_24h"] = float("nan")
    result["residual"] = float("nan")
    if result.empty:
        return result

    values = result["datavalue"].mask(result["datavalue"] <= ERROR_CODE_LIMIT)
    for part in _trend_segments(result):
        times = pd.DatetimeIndex(part["localdatetime"])
        series = pd.Series(part["datavalue"].to_numpy(), index=times)
        mean = series.rolling(TREND_WINDOW, center=True, closed="left").mean()
        full_window = _full_trend_window(part)
        result.loc[part.index, "trend_24h"] = mean.where(full_window).to_numpy()

    result["residual"] = values - result["trend_24h"]
    return result


def analyze_column(
    raw: pd.DataFrame,
    sensor_ids: Mapping[str, int],
    start,
    end_exclusive,
    daily_dates=None,
) -> DiurnalResult:
    """Прежний расчёт главы 1 для выбранной тройки и интервала [start, end).

    raw содержит localdatetime, sensorid, datavalue. sensor_ids задаёт три
    фактических уровня и их ID в порядке координат H(d), A(d), например
    {"D1": 994, "D2": 1010, "D3": 1026}. Названия уровней произвольны.
    Время — местное время базы без часового пояса и без преобразования часов.

    Сначала выбираются записи интервала, затем вызываются daily_trend() и
    analyze_diurnal(). Данные вне интервала не достраивают краевые окна;
    исходная таблица не изменяется. Сохраняются прежние правила среднего,
    разрывов, служебных значений, полных суток и отказа при повторных метках.
    При отсутствии измерений/общих окон расчёт явно сообщает об этом.

    daily_dates ограничивает суточные подгонки явно выбранными датами;
    окружающие измерения остаются доступными для прежнего скользящего среднего.
    При None сохраняется прежний расчёт всех полных суток.

    Функция не подключается к базе, не сохраняет файлы и не строит рисунки.
    Она не присваивает данным допуск I.G.: научный допуск входных данных и
    согласованные маски исключений должны быть установлены до её применения.
    Результат содержит прежние шесть таблиц, H (ч), A (ppm), ID и границы.
    """
    if not isinstance(sensor_ids, Mapping) or len(sensor_ids) != 3:
        raise ValueError("Нужны три фактических уровня и три различных sensorid.")
    if any(not isinstance(level, str) or not level.strip() for level in sensor_ids):
        raise ValueError("Каждый фактический уровень должен иметь непустое название.")
    ids = list(sensor_ids.values())
    if any(isinstance(sid, bool) or not isinstance(sid, Integral) for sid in ids) or len(set(ids)) != 3:
        raise ValueError("Нужны три различных целочисленных sensorid.")
    start, end_exclusive = pd.Timestamp(start), pd.Timestamp(end_exclusive)
    if pd.isna(start) or pd.isna(end_exclusive):
        raise ValueError("Начало и конец интервала должны быть определены.")
    if start.tzinfo is not None or end_exclusive.tzinfo is not None:
        raise ValueError("Нужно местное время базы без часового пояса.")
    if end_exclusive <= start:
        raise ValueError("Конец периода должен быть позже начала.")
    required = {"localdatetime", "sensorid", "datavalue"}
    if not required.issubset(raw.columns):
        raise ValueError("Нужны столбцы localdatetime, sensorid, datavalue.")
    selected = raw.loc[raw.sensorid.isin(ids)].copy()
    selected["localdatetime"] = pd.to_datetime(selected.localdatetime, errors="raise")
    if selected.localdatetime.isna().any():
        raise ValueError("В выбранных каналах есть неопределённые временные метки.")
    if selected.localdatetime.dt.tz is not None:
        raise ValueError("Нужно местное время базы без часового пояса.")
    selected = selected.loc[selected.localdatetime.ge(start)
                            & selected.localdatetime.lt(end_exclusive)].reset_index(drop=True)
    selected["datavalue"] = pd.to_numeric(selected.datavalue, errors="raise")
    for level, sid in sensor_ids.items():
        rows = selected.loc[selected.sensorid.eq(sid)]
        if rows.empty:
            raise ValueError(f"За выбранный период нет измерений {level} (sensorid={sid}).")
        if rows.localdatetime.duplicated().any():
            raise ValueError("Повторные временные метки требуют отдельного решения перед анализом.")
    derived = {level: daily_trend(selected.loc[selected.sensorid.eq(sid)])
               for level, sid in sensor_ids.items()}
    result = analyze_diurnal(derived, gap_factor=GAP_FACTOR, daily_dates=daily_dates)
    # Календарь относится к запрошенному интервалу, включая дни без записей
    # у его границ. Добавляются только причины отсутствия, не суточные оценки.
    days = pd.date_range(start.normalize(), end_exclusive.ceil("D"), freq="D", inclusive="left")
    if daily_dates is not None:
        days = days.intersection(pd.DatetimeIndex(daily_dates).normalize())
    coverage = result.day_coverage.set_index("date").reindex(days)
    coverage["included"] = coverage["included"].eq(True)
    coverage["reason"] = coverage["reason"].fillna("неполное окно или пропуск")
    result.day_coverage = coverage.rename_axis("date").reset_index()
    result.sensor_ids = dict(sensor_ids)
    result.start, result.end_exclusive = start, end_exclusive
    return result


def plot_daily_trend(raw: pd.DataFrame, sensor_ids: dict[str, int]):
    """Дополнительный рисунок: три датчика × (ряд со средним / отклонение)."""
    figure, axes = plt.subplots(3, 2, figsize=(17, 10), sharex=True, sharey="col")
    colors = ["#2563A6", "#18856A", "#B4661B"]
    residual_max = 0.0
    for (left, right), (level, sid), color in zip(axes, sensor_ids.items(), colors):
        rows = raw.loc[raw["sensorid"] == sid]
        derived = daily_trend(rows)
        original = line_with_gaps(rows)
        left.plot(original.index, original.to_numpy(), color=color, linewidth=0.75, label="Исходный ряд")
        left.plot(derived["localdatetime"], derived["trend_24h"], color="#202020", linewidth=2, label="Среднее за 24 ч")
        right.plot(derived["localdatetime"], derived["residual"], color=color, linewidth=0.75)
        right.axhline(0, color="#555555", linewidth=0.9, linestyle="--")
        left.set_ylabel(f"{level}\nCO₂, ppm")
        right.set_ylabel(f"{level}\nОтклонение, ppm")
        for axis in (left, right):
            axis.grid(True, alpha=0.22)
        valid_count = int(derived["residual"].notna().sum())
        print(f"{level}: полное суточное окно для {valid_count:,} из {len(rows):,} измерений.")
        if valid_count:
            residual_max = max(residual_max, float(derived["residual"].abs().max()))
        else:
            right.text(0.5, 0.5, "Нет полного суточного окна без пропусков",
                       transform=right.transAxes, ha="center", va="center", fontsize=10)

    axes[0, 0].set_title("Исходный ряд и суточное среднее", fontsize=13, pad=12)
    axes[0, 1].set_title("Отклонение = исходный ряд − суточное среднее", fontsize=13, pad=12)
    axes[0, 0].legend(loc="lower right", fontsize=9)
    if residual_max:
        axes[0, 1].set_ylim(-1.08 * residual_max, 1.08 * residual_max)
    axes[0, 0].set_xlim(START, END_EXCLUSIVE)
    axes[0, 0].xaxis.set_major_locator(mdates.DayLocator(interval=5))
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    for axis in axes[-1]:
        axis.set_xlabel("Местное время из базы (LOCALDATETIME)")
    last_day = END_EXCLUSIVE - pd.Timedelta(microseconds=1)
    figure.suptitle(
        f"LEO West · {COLUMN_NAME} · {START:%d.%m.%Y} — {last_day:%d.%m.%Y}",
        fontsize=15,
    )
    figure.text(
        0.5, 0.025,
        "Среднее: окно ±12 часов. При неполном окне на краях и около пропусков среднее и отклонение не рассчитываются.\n"
        "Интерполяция не применяется. Отклонения включают колебания разных периодов и шум.",
        ha="center", fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.08, 1, 0.96))
    return figure


def plot_raw_timeseries(raw: pd.DataFrame, sensor_ids: dict[str, int]):
    """Исходные три ряда; общий рисунок для запуска программы и notebook."""
    figure, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, sharey=True)
    colors = ["#2563A6", "#18856A", "#B4661B"]
    for axis, (level, sid), color in zip(axes, sensor_ids.items(), colors):
        rows = raw.loc[raw["sensorid"] == sid]
        series = line_with_gaps(rows)
        errors = int((rows["datavalue"] <= ERROR_CODE_LIMIT).sum())
        axis.plot(series.index, series.to_numpy(), linewidth=0.8, color=color)
        axis.set_title(f"{level}  |  {SENSOR_CODES[level]}  |  sensorid={sid}", loc="left", fontsize=11)
        axis.set_ylabel("CO₂, ppm")
        axis.grid(True, alpha=0.22)
        axis.text(
            0.99, 0.94, f"Измерений: {len(rows):,}; кодов ошибки: {errors}",
            transform=axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )
        print(f"{level}: {len(rows):,} строк; кодов ошибки: {errors}", flush=True)

    axes[-1].set_xlim(START, END_EXCLUSIVE)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    axes[-1].set_xlabel("Местное время из базы (LOCALDATETIME)")
    last_day = END_EXCLUSIVE - pd.Timedelta(microseconds=1)
    figure.suptitle(
        f"LEO West · {COLUMN_NAME}\n{START:%d.%m.%Y} — {last_day:%d.%m.%Y}",
        fontsize=15,
    )
    figure.text(
        0.5, 0.018,
        "Исходные измерения без усреднения и интерполяции. Пропуски — разрывы линий.\n"
        "Служебные значения ≤ −9999 скрыты только на рисунке; в CSV они сохранены.",
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.94))
    return figure


def main() -> None:
    raw, sensor_ids = fetch_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{COLUMN_NAME}_{START:%Y-%m-%d}_to_{END_EXCLUSIVE:%Y-%m-%d}_exclusive"
    csv_path = OUTPUT_DIR / f"{stem}_raw.csv"
    png_path = OUTPUT_DIR / f"{stem}.png"
    trend_png_path = OUTPUT_DIR / f"{stem}_trend24h.png"

    # Сохраняем даже служебные значения: исходные записи не теряются.
    raw.to_csv(csv_path, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")
    figure = plot_raw_timeseries(raw, sensor_ids)
    figure.savefig(png_path, dpi=160)
    trend_figure = plot_daily_trend(raw, sensor_ids)
    trend_figure.savefig(trend_png_path, dpi=160)
    print(f"Исходные данные: {csv_path}")
    print(f"Три графика: {png_path}")
    print(f"Суточное среднее и отклонения (3 × 2): {trend_png_path}")
    diurnal = analyze_column(raw, sensor_ids, START, END_EXCLUSIVE)
    diurnal_figures = save_diurnal(
        diurnal, OUTPUT_DIR, stem,
        f"LEO West · {COLUMN_NAME} · {START:%d.%m.%Y} — {(END_EXCLUSIVE - pd.Timedelta(microseconds=1)):%d.%m.%Y}",
    )
    if SHOW_PLOT and plt.get_backend().lower() != "agg":
        plt.show()
    plt.close(figure)
    plt.close(trend_figure)
    for diurnal_figure in diurnal_figures:
        plt.close(diurnal_figure)


if __name__ == "__main__":
    main()

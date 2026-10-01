"""Октябрьский пример: один расчёт для notebook и обычного Python.

load_october() читает только сохранённый снимок. Алгоритмы подготовки и
подгонки остаются в sensorDB; этот модуль связывает их с данными главы,
проверяет результаты и готовит рисунки и справочные таблицы.
Запуск файла в PyCharm повторяет расчёт и открывает четыре рисунка.
"""

from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import io
import json
import sys
import textwrap
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SENSOR_MODULES = PROJECT / "Project_description/sensorDB"
# Прежние программы sensorDB используют импорты соседних модулей по имени.
if str(SENSOR_MODULES) not in sys.path:
    sys.path.insert(0, str(SENSOR_MODULES))
import plot_column_timeseries as column
from diurnal_analysis import (
    DiurnalResult, analyze_diurnal, plot_daily_stability, plot_diurnal, wrap_hours,
)

DATA_DIR = Path(__file__).resolve().parent / "data/ch01"
FIGURES = ("raw", "trend", "diurnal", "daily")


@contextmanager
def _column_settings(manifest):
    """Параметры снимка действуют только во время вызова общих функций."""
    settings = {
        "COLUMN_NAME": manifest["column"],
        "START": datetime.fromisoformat(manifest["start_inclusive"]),
        "END_EXCLUSIVE": datetime.fromisoformat(manifest["end_exclusive"]),
        "SENSOR_CODES": {k: v["sensor_code"] for k, v in manifest["sensors"].items()},
        "TREND_WINDOW": pd.Timedelta(hours=manifest["trend_window_hours"]),
        "GAP_FACTOR": manifest["gap_factor"],
        "ERROR_CODE_LIMIT": manifest["service_value_limit"],
    }
    previous = {key: getattr(column, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(column, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(column, key, value)


def runtime_versions():
    """Необязательные сведения о версиях не должны прерывать вычисления."""
    runtime = {"Python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "matplotlib", "nbformat", "ipykernel"):
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            runtime[name] = "метаданные версии недоступны"
    return runtime


def _format_figure(fig, kind):
    """Единое оформление для печати, notebook и PyCharm; данные не меняются."""
    has_tables = any(len(ax.tables) for ax in fig.axes)
    height = 9 if has_tables else (7 if kind == "daily" else 8)
    fig.set_size_inches(10.8 if kind == "trend" else 10, height)
    for ax in fig.axes:
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_fontsize(13)
        ax.yaxis.label.set_fontsize(13)
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(13)
        if ax.get_legend() is not None:
            for text in ax.get_legend().get_texts():
                text.set_fontsize(11)
        for text in ax.texts:
            text.set_fontsize(11)
        for table in ax.tables:
            for (row, _), cell in table.get_celld().items():
                cell.get_text().set_fontsize(10)
                if row == 0:
                    cell.get_text().set_text(textwrap.fill(cell.get_text().get_text(), 20))
                    cell.set_height(cell.get_height() * 1.5)
    # Учебные пояснения вынесены в справочник; на рисунках остаются условия расчёта.
    footers = {
        "raw": "Исходные измерения. Пропуски показаны разрывами; интерполяции нет.",
        "trend": "Окно ±12 часов; только полные непрерывные окна. Интерполяции нет.",
        "diurnal": "Период 24 ч; месячная подгонка. Дневные квартили: 22 полных суток, не доверительный интервал.",
        "daily": "Точки: сутки; пунктир: месяц; серые полосы: исключённые дни.\n"
                 "Фазы развёрнуты около месячной оценки; 24:01 соответствует 00:01 по модулю суток.",
    }
    for text in fig.texts:
        if text is fig._suptitle:
            text.set_fontsize(15)
        else:
            text.set_text(footers[kind])
            text.set_fontsize(10)
            text.set_position((0.5, 0.015))
    if has_tables:
        fig.axes[1].set_title("Амплитуды, время максимумов и дневной разброс", loc="left", fontsize=11)
        fig.axes[2].set_title("Фазовые сдвиги, ч; положительный знак: более поздний пик", loc="left", fontsize=11)
        fig.subplots_adjust(top=.87, bottom=.11, left=.10, right=.97, hspace=.95)
    else:
        fig.tight_layout(rect=(0, .08, 1, .94))
    return fig


@dataclass
class OctoberReport:
    data_dir: Path
    manifest: dict
    raw: pd.DataFrame
    sensor_ids: dict
    derived: dict
    result: DiurnalResult

    def verify(self):
        """Сверить все шесть таблиц с эталонами; вернуть число успешных сверок."""
        tables = {
            "diurnal_summary": (self.result.summary.reset_index(), []),
            "diurnal_lags": (self.result.lags, []),
            "diurnal_daily": (self.result.daily, ["date"]),
            "diurnal_daily_lags": (self.result.daily_lags, ["date"]),
            "diurnal_day_coverage": (self.result.day_coverage, ["date"]),
            "diurnal_common_intervals": (
                pd.DataFrame(self.result.common_intervals, columns=["start", "end"]), ["start", "end"],
            ),
        }
        for name, (computed, dates) in tables.items():
            expected = pd.read_csv(self.data_dir / f"reference_{name}.csv", parse_dates=dates)
            pd.testing.assert_frame_equal(computed.reset_index(drop=True), expected,
                                          check_dtype=False, check_exact=False, atol=1e-7, rtol=1e-7)
        return len(tables)

    def figure(self, kind):
        """Вернуть matplotlib Figure: raw, trend, diurnal или daily."""
        if kind not in FIGURES:
            raise ValueError(f"Неизвестный рисунок {kind!r}; допустимы {FIGURES}")
        title = "LEO West · W_R4_C-4 · 01.10.2017 - 31.10.2017"
        with _column_settings(self.manifest), redirect_stdout(io.StringIO()):
            factories = {
                "raw": lambda: column.plot_raw_timeseries(self.raw, self.sensor_ids),
                "trend": lambda: column.plot_daily_trend(self.raw, self.sensor_ids),
                "diurnal": lambda: plot_diurnal(self.result, title),
                "daily": lambda: plot_daily_stability(self.result, title),
            }
            return _format_figure(factories[kind](), kind)

    def show(self, kind):
        """Показать рисунок в notebook и сохранить PNG в выводе ячейки."""
        from IPython.display import Image, display
        fig = self.figure(kind)
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=160)
            display(Image(data=buffer.getvalue()))
        finally:
            plt.close(fig)

    def table(self, name):
        """Справочные таблицы по запросу; основные результаты в self.result."""
        if name == "runtime":
            return pd.DataFrame(runtime_versions().items(), columns=["Компонент", "Версия"])
        if name == "counts":
            rows = []
            for level, sid in self.sensor_ids.items():
                part = self.raw.loc[self.raw.sensorid.eq(sid)].sort_values("localdatetime")
                steps = part.localdatetime.diff()
                typical = steps[steps.gt(pd.Timedelta(0))].median()
                rows.append([level, len(part), int(part.datavalue.le(self.manifest["service_value_limit"]).sum()),
                             round(typical.total_seconds() / 60, 2)])
            return pd.DataFrame(rows, columns=["Датчик", "Исходных записей", "Кодов ошибки", "Шаг, мин"])
        if name == "intervals":
            return pd.DataFrame(self.result.common_intervals, columns=["Начало общего интервала", "Конец общего интервала"])
        if name == "peak_quartiles":
            def clock(hours):
                minutes = int(round(hours * 60))
                return f"{minutes // 60:02d}:{minutes % 60:02d}"
            rows = []
            for level, row in self.result.summary.iterrows():
                peaks = self.result.daily.loc[self.result.daily.level.eq(level), "peak_hour_local"].dropna().to_numpy()
                aligned = row.peak_hour_local + wrap_hours(peaks - row.peak_hour_local)
                q25, q75 = np.quantile(aligned, [.25, .75])
                rows.append([level, clock(q25), clock(q75), f"{q75-q25:.2f}", f"{(q75-q25)*60:.0f}"])
            return pd.DataFrame(rows, columns=["Датчик", "Q25 пика", "Q75 пика", "IQR, ч", "IQR, мин"])
        if name == "excluded_days":
            dates = self.result.day_coverage.loc[~self.result.day_coverage.included, "date"]
            return pd.DataFrame({"Исключённые дни": dates.dt.strftime("%d.%m.%Y").to_numpy()})
        if name in ("example_peaks", "example_lags"):
            dates = pd.to_datetime(["2017-10-02", "2017-10-03", "2017-10-04", "2017-10-07", "2017-10-30"])
            if name == "example_peaks":
                frame = self.result.daily.pivot(index="date", columns="level", values="peak_time_local")
            else:
                frame = self.result.daily_lags.pivot(index="date", columns="pair", values="lag_hours").round(2)
            frame = frame.loc[dates].rename_axis("День").reset_index()
            frame["День"] = frame["День"].dt.strftime("%d.%m")
            return frame
        raise ValueError(f"Неизвестная справочная таблица: {name}")


def load_october(data_dir=DATA_DIR):
    """Прочитать снимок, выполнить прежний расчёт и проверить эталонные таблицы."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "provenance.json").read_text())
    for name, expected_hash in manifest["files_sha256"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Изменился файл снимка: {name}")
    changed = [name for name, sha in manifest["implementation_sha256"].items()
               if hashlib.sha256((PROJECT / "Project_description/sensorDB" / name).read_bytes()).hexdigest() != sha]
    if changed:
        warnings.warn("После исходной версии главы изменены модули: " + ", ".join(changed), stacklevel=2)
    sensor_ids = {level: info["sensor_id"] for level, info in manifest["sensors"].items()}
    raw = pd.read_csv(data_dir / "measurements.csv", parse_dates=["localdatetime"])
    expected = manifest["expected"]
    assert set(raw.sensorid) == set(sensor_ids.values())
    assert raw.groupby("sensorid").size().eq(expected["raw_rows_per_sensor"]).all()
    assert raw.localdatetime.ge(pd.Timestamp(manifest["start_inclusive"])).all()
    assert raw.localdatetime.lt(pd.Timestamp(manifest["end_exclusive"])).all()
    with _column_settings(manifest):
        derived = {level: column.daily_trend(raw.loc[raw.sensorid.eq(sid)]) for level, sid in sensor_ids.items()}
    result = analyze_diurnal(derived, gap_factor=manifest["gap_factor"])
    assert len(result.common_intervals) == expected["common_intervals"]
    assert result.summary.n_observations.eq(expected["fitted_observations_per_sensor"]).all()
    assert result.day_coverage.included.sum() == expected["complete_days"]
    assert len(result.daily) == len(sensor_ids) * expected["complete_days"]
    report = OctoberReport(data_dir, manifest, raw, sensor_ids, derived, result)
    report.verify()
    return report


if __name__ == "__main__":
    report = load_october()
    print(report.result.summary[["amplitude_ppm", "peak_time_local", "r_squared"]].to_string())
    print(report.result.lags.to_string(index=False))
    print(f"Эталонных таблиц сверено: {report.verify()}")
    for name in FIGURES:
        report.figure(name)
    plt.show()

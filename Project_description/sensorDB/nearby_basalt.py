"""Поиск соседних вертикалей LEO West по месячному допуску с заданным порогом амплитуды."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
from itertools import combinations, groupby
import json
import multiprocessing
from math import dist, isfinite, isnan
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import sys

import pandas as pd

import export_west_x1_y4 as exporter

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "Research_log"
sys.path.insert(0, str(RESEARCH))
from chapter02 import ADMISSION_PARAMETERS, monthly_admission
from stage02_candidate_selection import classify_months

TABLE = "LEO_WEST.DATAVALUES"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sensor_months(job):
    version, sid, source_hash, method = job
    cache = version.parent / "_basalt_months" / version.name
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{sid}-{source_hash}-{method}.csv"
    if not path.exists():
        source = version / f"diagnostics/sensors/{sid}_working.parquet"
        if digest(source) != source_hash:
            raise ValueError(f"Контрольная сумма исходных решений изменилась: {sid}")
        work = pd.read_parquet(source, columns=["localdatetime", "datavalue", "sensorid", "doubt_reason"])
        months, _ = monthly_admission(work, {str(sid): sid}, work.localdatetime.min(),
                                      work.localdatetime.max() + pd.Timedelta(microseconds=1))
        with NamedTemporaryFile(dir=cache, delete=False) as temp:
            months.to_csv(temp.name, index=False)
        Path(temp.name).replace(path)
    print(f"Месячные решения готовы: {sid}", flush=True)
    return sid, pd.read_csv(path, float_precision="round_trip")


def best_group(verticals, passes, occupancy, count, max_distance, min_months=1, rank_by="distance"):
    best = None
    for group in combinations(sorted(verticals), count):
        diameter = max((dist(a, b) for a, b in combinations(group, 2)), default=0.)
        if diameter > max_distance:
            continue
        ids = [c["sensorid"] for xy in group for c in verticals[xy]]
        months = set.intersection(*(passes[sid] for sid in ids))
        if not months:
            continue
        common = occupancy[ids].all(axis=1)
        observed = common[common].index.to_period("M")
        periods = sorted(set(observed) & {pd.Period(m, freq="M") for m in months})
        for _, items in groupby(enumerate(periods), key=lambda pair: pair[1].ordinal - pair[0]):
            run = [month for _, month in items]
            if len(run) < min_months:
                continue
            cells = int(observed.isin(run).sum())
            # При равенстве двух основных критериев сравниваем общее покрытие.
            priority = (-len(run), diameter) if rank_by == "duration" else (diameter, -len(run))
            rank = (*priority, -cells, group, run[0])
            if best is None or rank < best[0]:
                best = rank, group, run, cells, diameter
    return best


def load_occupancy(diagnostics, checks, ids, bin_minutes):
    grids = {}
    for sid in ids:
        suffix = "2h_counts" if bin_minutes == 120 else "working"
        name = f"sensors/{sid}_{suffix}.parquet"
        path = diagnostics / name
        if digest(path) != checks[name]:
            raise ValueError(f"Контрольная сумма данных для сетки изменилась: {sid}")
        if bin_minutes == 120:
            grids[sid] = pd.read_parquet(path, columns=["retained"]).retained.gt(0)
        else:
            # Ячейка занята, когда в ней есть измерение после технической очистки.
            work = pd.read_parquet(path, columns=["localdatetime", "technically_retained"])
            times = work.loc[work.technically_retained, "localdatetime"].dropna()
            bins = times.dt.floor(f"{bin_minutes}min").drop_duplicates()
            grids[sid] = pd.Series(True, index=pd.DatetimeIndex(bins), dtype="boolean")
    return pd.DataFrame(grids).fillna(False).astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument(
        "--amplitude-min", type=float, default=ADMISSION_PARAMETERS["monthly_amplitude_min"],
        help="Минимум месячной медианы суточной амплитуды, ppm (по умолчанию 8)",
    )
    parser.add_argument(
        "--rank-by", choices=("distance", "duration"), default="distance",
        help="Первый критерий отбора: диаметр группы или длительность общего периода",
    )
    parser.add_argument("--max-distance", type=float, default=float("inf"), help="Диаметр группы, м")
    parser.add_argument(
        "--bin-minutes", type=int, choices=(15, 120), default=15,
        help="Шаг сетки для проверки общих измерений, минуты (по умолчанию 15)",
    )
    parser.add_argument("--min-months", type=int, default=1, help="Минимум последовательных месяцев")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archive", type=Path, default=exporter.ARCHIVE)
    parser.add_argument("--output", type=Path, help="Папка для экспорта полных исходных рядов")
    args = parser.parse_args()
    if not isfinite(args.amplitude_min) or args.amplitude_min <= 0:
        parser.error("--amplitude-min должен быть конечным положительным числом")
    if args.min_months < 1 or args.workers < 1 or isnan(args.max_distance) or args.max_distance < 0:
        parser.error("Проверьте число месяцев, процессов и максимальное расстояние")
    exporter.ARCHIVE = args.archive
    version, manifest = exporter.latest_complete()
    verticals = {}
    for channel in manifest["channels"]:
        if channel["table"] == TABLE and channel["variableid"] == 9:
            verticals.setdefault((channel["x"], channel["y"]), []).append(channel)
    if len(verticals) != 16 or any(sorted(abs(c["z"]) for c in channels) not in
                                    ([.05, .20, .35], [.05, .20, .50]) for channels in verticals.values()):
        raise ValueError("Проверьте состав глубин и координаты в манифесте")
    if not 1 <= args.count <= len(verticals):
        parser.error("--count вне диапазона")
    diagnostics = version / "diagnostics"
    checks = json.loads((diagnostics / "summary.json").read_text())["files_sha256"]
    signature = (digest(RESEARCH / "chapter02.py") + digest(HERE / "diurnal_analysis.py")
                 + inspect.getsource(sensor_months))
    method = hashlib.sha256(signature.encode()).hexdigest()[:16]
    ids = sorted(c["sensorid"] for channels in verticals.values() for c in channels)
    jobs = [(version, sid, checks[f"sensors/{sid}_working.parquet"], method) for sid in ids]
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        metrics = dict(pool.map(sensor_months, jobs))
    # Границы месячных показателей не зависят от порога. Меняем только решение.
    passes = {}
    for sid, months in metrics.items():
        decisions = classify_months(months, args.amplitude_min)
        passes[sid] = set(decisions.loc[decisions.status.eq("pass"), "month"])
    occupancy = load_occupancy(diagnostics, checks, ids, args.bin_minutes)
    best = best_group(
        verticals, passes, occupancy, args.count, args.max_distance, args.min_months, args.rank_by,
    )
    if best is None:
        print("Общего допустимого месяца не найдено.")
        return
    _, group, run, cells, diameter = best
    start = run[0].start_time
    end = (run[-1] + 1).start_time
    expected_cells = int((end - start) / pd.Timedelta(minutes=args.bin_minutes))
    print(f"{version.name} · порог {args.amplitude_min:g} ppm · диаметр {diameter:.3f} м")
    print(f"Общий период: [{start.date()}, {end.date()}) · {len(run)} мес. · "
          f"{cells}/{expected_cells} общих ячеек по {args.bin_minutes} мин")
    selected = {}
    for xy in group:
        for c in sorted(verticals[xy], key=lambda c: -c["z"]):
            name = c["sensorcode"].replace("GMM222", "basalt") + ".parquet"
            selected[c["sensorid"], 9] = name
            print(xy, c["sensorid"], c["z"], name)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.output) as temp:
            stage = Path(temp)
            names = exporter.export(version, manifest, stage, {TABLE: selected})
            for name in names:
                (stage / name).replace(args.output / name)


if __name__ == "__main__":
    main()

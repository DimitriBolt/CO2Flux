"""Поиск соседних вертикалей LEO West по строгому месячному допуску I.G."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import inspect
from itertools import combinations, groupby
import json
import multiprocessing
from math import dist
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
import sys

import pandas as pd

import export_west_x1_y4 as exporter

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent / "Research_log"
sys.path.insert(0, str(RESEARCH))
from chapter02 import monthly_admission

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
    return sid, set(pd.read_csv(path).query("status == 'pass'").month)


def best_group(verticals, passes, occupancy, count, max_distance, min_months=1):
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
            # Сначала самый компактный набор, затем самый длинный период,
            # при равенстве — больше общих двухчасовых ячеек.
            rank = (diameter, -len(run), -cells, group, run[0])
            if best is None or rank < best[0]:
                best = rank, group, run, cells
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=float("inf"), help="Диаметр группы, м")
    parser.add_argument("--min-months", type=int, default=1, help="Минимум последовательных месяцев")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--archive", type=Path, default=exporter.ARCHIVE)
    parser.add_argument("--output", type=Path, help="Папка для экспорта полных исходных рядов")
    args = parser.parse_args()
    if args.min_months < 1 or args.workers < 1 or args.max_distance < 0:
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
        passes = dict(pool.map(sensor_months, jobs))
    grids = {}
    for sid in ids:
        name = f"sensors/{sid}_2h_counts.parquet"
        if digest(diagnostics / name) != checks[name]:
            raise ValueError(f"Контрольная сумма сетки изменилась: {sid}")
        grids[sid] = pd.read_parquet(diagnostics / name, columns=["retained"]).retained.gt(0)
    occupancy = pd.DataFrame(grids).fillna(False)
    best = best_group(verticals, passes, occupancy, args.count, args.max_distance, args.min_months)
    if best is None:
        print("Общего допустимого месяца не найдено.")
        return
    rank, group, run, cells = best
    print(version.name, group, "диаметр, м", rank[0], "период", run[0], run[-1], "общих ячеек", cells)
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

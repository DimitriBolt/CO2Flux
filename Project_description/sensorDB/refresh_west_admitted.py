"""Полное обновление рабочих рядов CO₂ LEO West из Oracle."""

import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from co2_full_diagnostics import STUCK_INTERVALS, clean_admission_measurements
from co2_refresh import Oracle, inspect_metadata, workbook_channels
from stage02_candidate_selection import classify_months
from chapter02 import monthly_admission

HERE = Path(__file__).resolve().parent
TARGET = HERE / "Row_data/LEO_West"
THRESHOLD = 2.0
SERVICE = {1302: "AirInflow", 1303: "AirOutflow", 1305: "Vent"}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def channels():
    entries = workbook_channels()
    for sid, label in SERVICE.items():
        entries.append(dict(sensorid=sid, variableid=56, table="LEO_WEST.DATAVALUESLICOR",
                            sensorcode=f"LEO-W_{label}_LI-7000", kind="LI-COR",
                            x=None, y=None, z=None, units="umol/mol"))
    for c in entries:
        basalt = c["kind"] == "GMM222"
        stem = c["sensorcode"].removesuffix("_GMM222" if basalt else "_LI-7000")
        c["file"] = stem + ("_basalt.parquet" if basalt else "_air.parquet")
    if len({c["file"] for c in entries}) != 56 or len({c["sensorid"] for c in entries}) != 56:
        raise ValueError("Неверный состав каналов LEO West")
    return entries


def read_table(db, entries, scratch):
    """Один проход по таблице. Временные части живут только до конца обновления."""
    table = entries[0]["table"]
    ids = [c["sensorid"] for c in entries]
    where = (f" FROM {table} WHERE variableid=:var AND sensorid IN ("
             + ",".join(map(str, ids)) + ")")
    params = {"var": entries[0]["variableid"]}
    bounds = {int(sid): (int(n), first, last) for sid, n, first, last in db.select(
        "SELECT sensorid,count(*),min(localdatetime),max(localdatetime)" + where
        + " GROUP BY sensorid", params)}
    writers = {}
    try:
        for rows in db.stream("SELECT sensorid,localdatetime,datavalue" + where, params):
            frame = pd.DataFrame(rows, columns=["sid", "localdatetime", "datavalue"]).astype(
                {"sid": "int32", "localdatetime": "datetime64[us]", "datavalue": "float64"})
            for sid, group in frame.groupby("sid"):
                data = pa.Table.from_pandas(group[["localdatetime", "datavalue"]], preserve_index=False)
                if sid not in writers:
                    writers[sid] = pq.ParquetWriter(scratch / f"{sid}.parquet", data.schema)
                writers[sid].write_table(data)
    finally:
        for writer in writers.values():
            writer.close()
    for c in entries:
        sid = c["sensorid"]
        path = scratch / f"{sid}.parquet"
        raw = pd.read_parquet(path) if path.exists() else pd.DataFrame({
            "localdatetime": pd.Series(dtype="datetime64[us]"), "datavalue": pd.Series(dtype="float64")})
        count, first, last = bounds.get(sid, (0, None, None))
        if (len(raw) != count or raw.localdatetime.isna().any()
                or (count and (raw.localdatetime.min(), raw.localdatetime.max()) != (first, last))):
            raise ValueError(f"Неполная история {sid}")
        yield c, raw


def monthly_input_hash(frame):
    # Все входы суточной аппроксимации и расчёта её границ, включая сомнительные точки.
    h = hashlib.sha256()
    h.update(frame.localdatetime.to_numpy(dtype="<M8[us]").tobytes())
    h.update(frame.datavalue.to_numpy(dtype="<f8").tobytes())
    h.update(frame.doubt_reason.ne("").to_numpy(dtype="uint8").tobytes())
    return h.hexdigest()


def method_hash():
    files = [Path(__file__), HERE.parent / "Research_log/chapter02.py",
             HERE / "diurnal_analysis.py", HERE / "plot_column_timeseries.py",
             HERE / "stage02_candidate_selection.py"]
    return hashlib.sha256("".join(sha(f) for f in files).encode()).hexdigest()


def build_channel(raw, c, staging, previous):
    sid = c["sensorid"]
    level = str(sid)
    raw = raw.assign(sensorid=sid, variableid=c["variableid"])
    intervals = STUCK_INTERVALS if sid in (994, 1010, 1026) else []
    exclusions = pd.DataFrame([(level, a, b) for a, b in intervals], columns=["level", "start", "end"])
    work = clean_admission_measurements(raw, {level: sid}, exclusions, channel_metadata={level: c})
    periods = work.localdatetime.dt.to_period("M")
    months = []
    if len(work):
        groups = work.groupby(periods).indices
        for month in pd.period_range(periods.min(), periods.max(), freq="M"):
            group = work.iloc[groups.get(month, [])]
            fingerprint = monthly_input_hash(group)
            old = previous.get((sid, str(month)))
            if old is not None and old["input_sha256"] == fingerprint:
                record = dict(old)
            elif group.empty:
                record = dict(month=str(month), level=level, sensorid=sid, valid_days=0,
                              minimum_valid_days=0, doubtful_valid_days=0, status="fail",
                              reason="fewer_than_10_valid_days")
            else:
                metrics, _ = monthly_admission(group, {level: sid}, month.start_time, (month + 1).start_time)
                record = classify_months(metrics, THRESHOLD).iloc[0].to_dict()
            record.update(input_sha256=fingerprint, file=c["file"])
            months.append(record)
    passed = [r["month"] for r in months if r["status"] == "pass"]
    kept = work.loc[work.exclusion_reason.eq("") & periods.isin(pd.PeriodIndex(passed, freq="M"))]
    df = kept.set_index("localdatetime")[["datavalue"]].astype("float64")
    df.index = df.index.as_unit("us", round_ok=False)
    if (not df.index.is_unique or not df.index.is_monotonic_increasing
            or not df.datavalue.between(0, 3000, inclusive="right").all()):
        raise ValueError(f"Неверный ряд канала {sid}")
    counts = df.groupby(df.index.to_period("M")).size()
    for r in months:
        r["exported_rows"] = int(counts.get(pd.Period(r["month"], freq="M"), 0))
        if bool(r["exported_rows"]) != (r["status"] == "pass"):
            raise ValueError(f"Месячный допуск {sid} {r['month']}")
    dest = staging / c["file"]
    df.to_parquet(dest, compression="zstd")
    pd.testing.assert_frame_equal(df, pd.read_parquet(dest), check_exact=True)
    report = {key: c[key] for key in ("sensorid", "sensorcode", "variableid", "table", "units", "x", "y", "z", "file")}
    report.update(source_rows=len(raw), source_first=str(raw.localdatetime.min()) if len(raw) else None,
                  source_last=str(raw.localdatetime.max()) if len(raw) else None,
                  exported_rows=len(df), pass_months=len(passed), file_sha256=sha(dest))
    return report, months


def finish(staging, snapshot, reports, monthly):
    if {r["sensorid"] for r in reports} != {c["sensorid"] for c in channels()} or len(reports) != 56:
        raise ValueError("Выборка не охватывает 56 каналов")
    path = staging / "months.csv"
    pd.DataFrame(monthly).sort_values(["sensorid", "month"]).to_csv(path, index=False)
    info = dict(oracle_snapshot=snapshot, monthly_amplitude_min=THRESHOLD, method_sha256=method_hash(),
                months_sha256=sha(path), channels=sorted(reports, key=lambda r: r["sensorid"]))
    (staging / "selection.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")


def exchange(staging):
    # Атомарная смена всей папки на Linux после проверки всех 56 рядов.
    # Не удаляем неизвестные пользовательские файлы.
    unknown = [p.name for p in TARGET.iterdir() if not (p.is_file() and
               (p.match("LEO-W_*_basalt.parquet") or p.match("LEO-W_*_air.parquet")
                or p.name in {"months.csv", "selection.json"}))]
    if unknown:
        raise ValueError(f"В папке есть посторонние файлы: {unknown}")
    rename = ctypes.CDLL(None, use_errno=True).renameat2
    if rename(-100, os.fsencode(staging), -100, os.fsencode(TARGET), 2):
        raise OSError(ctypes.get_errno(), "Не удалось заменить LEO_West")


def main():
    entries = channels()
    lock = os.open(TARGET.parent, os.O_RDONLY)
    db = Oracle()
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db.connection().call_timeout = 600000
        with db.connection().cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
        snapshot = str(db.select("SELECT sysdate FROM dual")[0][0])
        inspect_metadata(db, [c for c in entries if c["sensorid"] not in SERVICE])
        services = db.select("SELECT sensorid,sensorcode,datatablename FROM leo_west.sensors "
                             "WHERE sensorid IN (1302,1303,1305)")
        expected = {(i, f"LEO-W_{n}_LI-7000", "DATAVALUESLICOR") for i, n in SERVICE.items()}
        if {(int(i), n, t.upper()) for i, n, t in services} != expected:
            raise ValueError("В Oracle изменились обозначения служебных каналов")
        previous = {}
        if (TARGET / "selection.json").exists():
            info = json.loads((TARGET / "selection.json").read_text())
            path = TARGET / "months.csv"
            if (info.get("method_sha256") == method_hash() and info.get("monthly_amplitude_min") == THRESHOLD
                    and info["months_sha256"] == sha(path)):
                for row in pd.read_csv(path, float_precision="round_trip").to_dict("records"):
                    previous[(row["sensorid"], row["month"])] = row
        with TemporaryDirectory(prefix=".leo-west-refresh-", dir=TARGET.parent) as tmp:
            staging = Path(tmp) / "ready"
            staging.mkdir()
            reports, monthly = [], []
            for table in sorted({c["table"] for c in entries}):
                group = [c for c in entries if c["table"] == table]
                for c, raw in read_table(db, group, Path(tmp)):
                    report, months = build_channel(raw, c, staging, previous)
                    reports.append(report)
                    monthly.extend(months)
                    print(c["file"], report["exported_rows"], flush=True)
            finish(staging, snapshot, reports, monthly)
            exchange(staging)
    finally:
        db.close()
        os.close(lock)


if __name__ == "__main__":
    main()

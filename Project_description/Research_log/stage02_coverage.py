"""Stage 0.2 only: immutable inputs, descriptive clocks, bounded air SELECTs.

Run from the repository root with python3 -B. No cleaning, interpolation,
carry-forward, frequency selection or spectral admission occurs here.
Oracle is imported/connected only for the explicit inspect/extract actions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = HERE / "data/stage0"
BASE = HERE / "data/ch02/archive/raw.parquet"
OLD_AIR = ROOT / "Project_description/sensorDB/Row_data/LEO_West/LEO-W_4_0_1_air.parquet"
EXPECTED = {
    BASE: "89e69150734e8ae18ce00ebb554e6cd0dfc605172d54af2febf71c1c086ab40a",
    OLD_AIR: "2e9b081b3f0012c43a4324ffbefbe2b8a0e8b44a37e1837985e33da50cad8c83",
}
START = pd.Timestamp("2013-09-11 12:30:00")
END = pd.Timestamp("2026-10-01 00:00:00")
TABLE = "leo_west.datavalueslicor"
KEY = {"sensor_id": 1275, "variable_id": 56}
FILTER = "sensorid=:sensor_id AND variableid=:variable_id AND localdatetime>=:start_time AND localdatetime<:end_time"
STATS_SQL = f"""SELECT COUNT(*), MIN(localdatetime), MAX(localdatetime),
    COUNT(*)-COUNT(datavalue), SUM(CASE WHEN datavalue<=-9999 THEN 1 ELSE 0 END),
    COUNT(*)-COUNT(DISTINCT localdatetime)
    FROM {TABLE} WHERE {FILTER}"""
STATS_NAMES = ("rows", "first", "last", "null_values", "service_le_minus9999", "duplicate_time_extras")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n")


def check_inputs():
    for path, expected in EXPECTED.items():
        if sha(path) != expected:
            raise ValueError(f"Input checksum mismatch: {path}")


def month_windows(start=START, end=END):
    current = pd.Timestamp(start)
    while current < end:
        following = min(current.normalize().replace(day=1) + pd.offsets.MonthBegin(1), end)
        yield current, following
        current = following


def read_channels(air_path=OLD_AIR):
    check_inputs()
    basalt = pd.read_parquet(BASE)
    if set(basalt.sensorid.unique()) != {994, 1010, 1026} or set(basalt.variableid.unique()) != {9}:
        raise ValueError("Unexpected basalt channels")
    channels = {name: basalt.loc[basalt.sensorid.eq(sid), ["localdatetime", "datavalue"]].copy()
                for name, sid in (("C_5", 994), ("C_20", 1010), ("C_35", 1026))}
    air = pd.read_parquet(air_path)
    if "localdatetime" not in air:
        air = air.reset_index()
    if Path(air_path) != OLD_AIR:
        if not air.sensorid.eq(1275).all() or not air.variableid.eq(56).all():
            raise ValueError("Unexpected air channel")
    channels["C_air"] = air[["localdatetime", "datavalue"]].copy()
    return channels


def window_metrics(frame, start, end):
    """Half-open interval. Boundary gaps are separate from inter-observation gaps."""
    part = frame.loc[frame.localdatetime.ge(start) & frame.localdatetime.lt(end)]
    t = part.localdatetime.dropna().sort_values()
    delta = t.diff().dt.total_seconds()
    return dict(rows=len(part), nonnull_values=int(part.datavalue.notna().sum()),
                first=None if t.empty else t.iloc[0], last=None if t.empty else t.iloc[-1],
                max_internal_gap_s=None if len(t) < 2 else float(delta.max()),
                leading_gap_s=None if t.empty else (t.iloc[0] - start).total_seconds(),
                trailing_gap_s=None if t.empty else (end - t.iloc[-1]).total_seconds())


def compare_observations(old, new):
    """Compare at the old export's float64 precision; raw Decimal data stay intact."""
    def counts(frame):
        return Counter((pd.Timestamp(t).isoformat(), None if pd.isna(v) else float(v))
                       for t, v in frame[["localdatetime", "datavalue"]].itertuples(index=False, name=None))
    a, b = counts(old), counts(new)
    matched = a & b
    missing, extra = a - b, b - a
    return {"old_rows": sum(a.values()), "new_rows": sum(b.values()),
            "matched_old_rows": sum(matched.values()), "missing_old_rows": sum(missing.values()),
            "additional_new_rows": sum(extra.values()),
            "missing_old_examples": list(missing.items())[:20], "additional_examples": list(extra.items())[:20]}


def common_runs(calendar):
    active = calendar.common_nonnull_day.to_numpy()
    runs, i = [], 0
    while i < len(calendar):
        j = i + 1
        while j < len(calendar) and active[j] == active[i]:
            j += 1
        runs.append(dict(start=calendar.index[i], end_exclusive=calendar.index[j-1] + pd.Timedelta(days=1),
                         days=j-i, all_four_have_nonnull_each_day=bool(active[i])))
        i = j
    return pd.DataFrame(runs)


def run_metrics(channels, runs):
    return pd.DataFrame([
        dict(channel=name, start=row.start, end_exclusive=row.end_exclusive,
             calendar_days=row.days, **window_metrics(frame, row.start, row.end_exclusive))
        for row in runs.itertuples() if row.all_four_have_nonnull_each_day
        for name, frame in channels.items()
    ])


def clock_proximity(channels):
    air = pd.DatetimeIndex(channels["C_air"].localdatetime.dropna()).asi8
    proximity = pd.DataFrame({"air_time": pd.to_datetime(air)})
    summaries = {}
    for name in ("C_5", "C_20", "C_35"):
        ref = np.sort(pd.DatetimeIndex(channels[name].localdatetime.dropna()).asi8)
        j = np.searchsorted(ref, air)
        nearest = np.minimum(abs(air-ref[np.clip(j, 0, len(ref)-1)]),
                             abs(air-ref[np.clip(j-1, 0, len(ref)-1)]))/1e9
        proximity["nearest_" + name + "_seconds"] = nearest
        summaries[name] = {"min": float(nearest.min()), "median": float(np.median(nearest)),
                           "p95": float(np.quantile(nearest, .95)), "max": float(nearest.max())}
    return proximity, summaries


def diagnostics(channels, output, source_hashes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    dates = pd.date_range(START.normalize(), END, inclusive="left", freq="D")
    calendar = pd.DataFrame(index=dates)
    summaries, monthly, distributions, gaps, windows = [], [], [], [], []
    for name, frame in channels.items():
        valid_time = frame.localdatetime.notna()
        t = frame.loc[valid_time, "localdatetime"].sort_values()
        delta = t.diff().dt.total_seconds()
        summaries.append(dict(channel=name, rows=len(frame), first=t.min(), last=t.max(),
                              observed_days=t.dt.normalize().nunique(), null_times=int((~valid_time).sum()),
                              null_values=int(frame.datavalue.isna().sum()),
                              service_le_minus9999=int(frame.datavalue.le(-9999).sum()),
                              duplicate_time_extras=int(t.duplicated().sum()), timezone=str(t.dt.tz),
                              max_internal_gap_s=delta.max()))
        # Calendar buckets are derived labels only; original times are never changed.
        daily = frame.groupby(frame.localdatetime.dt.normalize()).agg(rows=("datavalue", "size"),
                                                                       nonnull=("datavalue", "count"))
        calendar[name + "_rows"] = daily.rows.reindex(dates, fill_value=0)
        calendar[name + "_nonnull"] = daily.nonnull.reindex(dates, fill_value=0)
        for start, end in month_windows():
            monthly.append(dict(channel=name, start=start, end_exclusive=end,
                                **window_metrics(frame, start, end)))
        distributions.extend(dict(channel=name, seconds=float(seconds), count=int(n))
                             for seconds, n in delta[delta.gt(0)].value_counts().items())
        times = pd.DatetimeIndex(t).asi8
        diffs = np.diff(times) / 1e9
        for i in np.argsort(diffs)[-20:][::-1]:
            gaps.append(dict(channel=name, start=pd.Timestamp(times[i]), end=pd.Timestamp(times[i+1]), seconds=diffs[i]))
        # Vector bounds and NumPy slices avoid repeated full-history scans.
        ordered = frame.loc[valid_time].sort_values("localdatetime")
        nonnull = np.r_[0, np.cumsum(ordered.datavalue.notna().to_numpy())]
        for days in (1, 7, 30):
            for anchor in dates:
                end = anchor + pd.Timedelta(days=days)
                if end > END:
                    continue
                start = max(anchor, START)
                lo, hi = np.searchsorted(times, [start.value, end.value])
                n = hi - lo
                windows.append(dict(channel=name, days=days, start=start, end_exclusive=end,
                                    rows=n, nonnull_values=nonnull[hi]-nonnull[lo],
                                    max_internal_gap_s=float(diffs[lo:hi-1].max()) if n > 1 else None,
                                    leading_gap_s=(times[lo]-start.value)/1e9 if n else None,
                                    trailing_gap_s=(end.value-times[hi-1])/1e9 if n else None))
    calendar["common_row_day"] = calendar[[n+"_rows" for n in channels]].gt(0).all(axis=1)
    calendar["common_nonnull_day"] = calendar[[n+"_nonnull" for n in channels]].gt(0).all(axis=1)
    calendar.to_csv(output / "daily_calendar.csv", index_label="date")
    runs = common_runs(calendar)
    runs.to_csv(output / "calendar_runs.csv", index=False)
    run_metrics(channels, runs).to_csv(output / "common_run_metrics.csv", index=False)
    for stem, rows in (("channels", summaries), ("monthly", monthly),
                       ("positive_intervals", distributions), ("largest_gaps", gaps), ("windows", windows)):
        pd.DataFrame(rows).to_csv(output / (stem + ".csv"), index=False)
    # Clock proximity, not an admission threshold or proof of physical simultaneity.
    proximity, proximity_summary = clock_proximity(channels)
    proximity.to_csv(output / "air_nearest_basalt.csv", index=False)
    result = {"created_at_utc": utc_now(), "source_hashes": source_hashes,
              "timezone_status": "unconfirmed; all calendar/clock comparisons provisional",
              "window_definition": "1/7/30 calendar days, daily starts, [start,end); first start clipped to source bound",
              "count_definition": "rows count actual source rows including duplicates/service/NULL values; nonnull counts stored values, not valid values",
              "zero_counts": "empty calendar buckets only; no zero-filled concentration series",
              "max_gap_definition": "between observations within window; leading/trailing gaps reported separately; empty/singleton internal gap undefined",
              "common_days": int(calendar.common_nonnull_day.sum()),
              "common_ranges": runs.loc[runs.all_four_have_nonnull_each_day].to_dict("records"),
              "bounds_intersection": [max(s["first"] for s in summaries), min(s["last"] for s in summaries)],
              "nearest_basalt_seconds": proximity_summary,
              "spectral_admission": "not assessed", "files_sha256": {p.name: sha(p) for p in output.glob("*.csv")}}
    write_json(output / "summary.json", result)
    return result


def connect():
    # Reuse the existing thick-mode connector; never print connection parameters.
    sys.path.insert(0, str(ROOT / "scripts"))
    from update_co2_sheet import connect as existing_connect
    import oracledb
    # Preserve Oracle NUMBER/FLOAT decimal values rather than coercing to float64.
    oracledb.defaults.fetch_decimals = True
    conn = existing_connect()
    conn.call_timeout = 30000
    return conn


def bounded_select(cur, sql, params=None):
    start = time.monotonic()
    cur.execute(sql, params or {})
    rows = cur.fetchall()
    elapsed = time.monotonic() - start
    if elapsed > 30:
        raise TimeoutError("SELECT exceeded 30-second budget; no automatic retry")
    return rows


def metadata(cur):
    columns = bounded_select(cur, """SELECT column_name, data_type, data_length, data_precision,
        data_scale, nullable FROM all_tab_columns
        WHERE owner='LEO_WEST' AND table_name='DATAVALUESLICOR' ORDER BY column_id""")
    descriptions = [dict(zip(("name", "type", "length", "precision", "scale", "nullable"), row)) for row in columns]
    names = {c["name"] for c in descriptions}
    if not {"LOCALDATETIME", "SENSORID", "VARIABLEID", "DATAVALUE"} <= names:
        raise ValueError("Required columns absent")
    comments = bounded_select(cur, """SELECT column_name, comments FROM all_col_comments
        WHERE owner='LEO_WEST' AND table_name='DATAVALUESLICOR' ORDER BY column_name""")
    return {"checked_at_utc": utc_now(), "table": TABLE, "columns": descriptions,
            "column_comments": comments,
            "timezone": "not established by data type alone; no conversion applied"}


def stats(cur, start, end):
    params = dict(KEY, start_time=start.to_pydatetime(), end_time=end.to_pydatetime())
    values = bounded_select(cur, STATS_SQL, params)[0]
    return {name: (int(value or 0) if name not in ("first", "last") else value)
            for name, value in zip(STATS_NAMES, values)}


def local_stats(frame):
    return dict(rows=len(frame), first=None if frame.empty else frame.localdatetime.min().to_pydatetime(),
                last=None if frame.empty else frame.localdatetime.max().to_pydatetime(),
                null_values=int(frame.datavalue.isna().sum()), service_le_minus9999=int(frame.datavalue.le(-9999).sum()),
                duplicate_time_extras=int(frame.localdatetime.duplicated().sum()))


def extract(output):
    """Sequential month checks, unaggregated raw SELECT, and independent recheck.

    Stop on any slow query, count discrepancy, access error or schema issue.
    A failed directory is retained as incomplete; it is never silently resumed.
    """
    check_inputs()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "incomplete", "started_at_utc": utc_now(), "table": TABLE,
                "sensorid": 1275, "variableid": 56, "sensorcode": "LEO-W_4_0_1_LI-7000",
                "units": "umol/mol", "coordinates_m": [0, 4, .25], "horizontal_offset_m": 4,
                "start_inclusive": str(START), "end_exclusive": str(END), "monthly": [],
                "statistics_query": STATS_SQL, "numeric_storage": "Oracle NUMBER/FLOAT fetched as Decimal and stored as Parquet decimals; no float64 cast",
                "snapshot_consistency": "per-month before/after checks, not a single database transaction snapshot"}
    write_json(output / "provenance.json", manifest)
    conn = None
    try:
        conn = connect()
        cur = conn.cursor()
        try:
            manifest["metadata"] = metadata(cur)
            columns = manifest["metadata"]["columns"]
            if any(not re.fullmatch(r"[A-Z][A-Z0-9_]*", c["name"]) for c in columns):
                raise ValueError("Unsupported column identifier")
            if any(c["type"] not in {"DATE", "NUMBER", "VARCHAR2", "CHAR", "FLOAT", "BINARY_DOUBLE", "BINARY_FLOAT"}
                   and not c["type"].startswith("TIMESTAMP") for c in columns):
                raise ValueError("Unsupported source type; inspect before extraction")
            # All actual columns, including IDs/quality fields; no guessed columns.
            projection = ", ".join('"'+c["name"]+'"' for c in columns)
            sql = f"SELECT {projection} FROM {TABLE} WHERE {FILTER} ORDER BY localdatetime"
            manifest["raw_query"] = sql
            write_json(output / "provenance.json", manifest)
            frames = []
            for start, end in month_windows():
                before = stats(cur, start, end)
                params = dict(KEY, start_time=start.to_pydatetime(), end_time=end.to_pydatetime())
                rows = bounded_select(cur, sql, params)
                frame = pd.DataFrame(rows, columns=[c["name"].lower() for c in columns])
                frame["localdatetime"] = pd.to_datetime(frame.localdatetime)
                after = stats(cur, start, end)
                if before != after or local_stats(frame) != after:
                    raise ValueError(f"Monthly verification failed: {start} — {end}")
                if not frame.localdatetime.ge(start).all() or not frame.localdatetime.lt(end).all():
                    raise ValueError("Boundary leakage")
                if not frame.sensorid.eq(1275).all() or not frame.variableid.eq(56).all():
                    raise ValueError("Channel mismatch")
                if len(frame):
                    frames.append(frame)
                manifest["monthly"].append(dict(start=str(start), end_exclusive=str(end), **after))
                write_json(output / "provenance.json", manifest)
                print(f"Verified {start:%Y-%m}: {len(frame)} rows", flush=True)
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[c["name"].lower() for c in columns])
            if len(combined) != sum(m["rows"] for m in manifest["monthly"]):
                raise ValueError("Total count mismatch")
            path = output / "raw.parquet"
            combined.to_parquet(path, index=False)
            reread = pd.read_parquet(path)
            pd.testing.assert_frame_equal(combined, reread)
            manifest.update(status="complete", finished_at_utc=utc_now(), rows=len(combined),
                            file=path.name, sha256=sha(path), columns=list(combined.columns),
                            totals=local_stats(combined), boundary_partition_verified=True)
            if "valueid" in combined:
                manifest["valueid_nulls"] = int(combined.valueid.isna().sum())
                manifest["valueid_duplicate_extras"] = int(combined.valueid.duplicated().sum())
            write_json(output / "provenance.json", manifest)
        finally:
            cur.close()
    except Exception as exc:
        manifest.update(status="failed", stopped_at_utc=utc_now(), error_type=type(exc).__name__, error=str(exc))
        write_json(output / "provenance.json", manifest)
        raise
    finally:
        if conn is not None:
            conn.close()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initial", "inspect", "extract", "final"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    check_inputs()
    if args.action == "inspect":
        conn = connect()
        try:
            cur = conn.cursor()
            try:
                print(json.dumps(metadata(cur), default=str, indent=2))
            finally:
                cur.close()
        finally:
            conn.close()
        return
    if args.action == "extract":
        result = extract(args.output or STAGE / "air_1275")
        print(json.dumps({k: result[k] for k in ("status", "rows", "sha256")}, indent=2))
        return
    air_path = OLD_AIR
    if args.action == "final":
        manifest = json.loads((STAGE / "air_1275/provenance.json").read_text())
        air_path = STAGE / "air_1275/raw.parquet"
        if manifest["status"] != "complete" or sha(air_path) != manifest["sha256"]:
            raise ValueError("Unverified new air snapshot")
    result = diagnostics(read_channels(air_path), args.output or STAGE / ("coverage_" + args.action),
                         {str(BASE.relative_to(ROOT)): sha(BASE), str(air_path.relative_to(ROOT)): sha(air_path)})
    if args.action == "final":
        comparison = compare_observations(pd.read_parquet(OLD_AIR).reset_index(), pd.read_parquet(air_path))
        write_json((args.output or STAGE / "coverage_final") / "old_air_comparison.json", comparison)
        print(json.dumps(comparison, default=str, indent=2))
    print(json.dumps({k: result[k] for k in ("common_days", "bounds_intersection", "nearest_basalt_seconds")}, default=str, indent=2))


if __name__ == "__main__":
    main()

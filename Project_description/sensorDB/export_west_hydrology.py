"""Export LEO West hydrology as one CO2-compatible Parquet per channel.

Oracle is read only. Local monthly checkpoints retain VALUEID and exact decimal
strings. Reuse a complete local snapshot with --local-only to export any subset.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, mkdtemp
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
ROW = Path(__file__).resolve().parent / "Row_data"
RAW = ROW / "LEO_West_hydrology_raw"
TABLES = ("DATAVALUES", "DATAVALUESREU24")
QUANTITIES = {
    6: ("LEO_West_VWC", "vwc", "%", "VWC", 1),
    7: ("LEO_West_Matric_Potential", "matric_potential", "kPa", "MWP", 2),
    17: ("LEO_West_Mass", "mass", "kg", "Mass", 7),
}
RAW_SCHEMA = pa.schema([
    ("valueid", pa.int64()), ("sensorid", pa.int32()),
    ("variableid", pa.int32()), ("localdatetime", pa.timestamp("us")),
    ("datavalue", pa.float64()), ("datavalue_oracle_decimal", pa.string()),
])
COUNT_SQL = ("SELECT sensorid,variableid,count(*),min(localdatetime),max(localdatetime) "
             "FROM leo_west.{table} WHERE variableid IN (6,7,17){window} "
             "GROUP BY sensorid,variableid")
DATA_SQL = ("SELECT valueid,sensorid,variableid,localdatetime,datavalue "
            "FROM leo_west.{table} WHERE variableid IN (6,7,17) "
            "AND localdatetime>=:a AND localdatetime<:b")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text())


def connect():
    sys.path.insert(0, str(ROOT / "scripts"))
    import oracledb
    from update_co2_sheet import connect as existing_connect
    oracledb.defaults.fetch_decimals = True
    conn = existing_connect()
    conn.call_timeout = 1800000
    return conn


def records(cur, sql, params=None):
    cur.execute(sql, params or {})
    names = [d[0].lower() for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def capture_metadata(work):
    """Initial whole-history counts include actual pairs absent from SENSOROUTPUT."""
    conn = connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SELECT sysdate FROM dual")
            snapshot = str(cur.fetchone()[0])
            for name in ("sensors", "variables", "units", "locations", "sensoroutput", "sensortypes"):
                dump(work / (name + ".json"), records(cur, "SELECT * FROM leo_west." + name))
            cur.execute("SELECT table_name,column_name,data_type,nullable FROM all_tab_columns "
                        "WHERE owner='LEO_WEST' AND table_name IN ('DATAVALUES','DATAVALUESREU24') "
                        "ORDER BY table_name,column_id")
            dump(work / "columns.json", cur.fetchall())
            for table in TABLES:
                cur.execute(COUNT_SQL.format(table=table, window=""))
                keys = ("sensorid", "variableid", "rows", "first_time", "last_time")
                dump(work / (table + "_counts.json"), {
                    "oracle_snapshot": snapshot,
                    "channels": [dict(zip(keys, r)) for r in cur.fetchall()]})
    finally:
        conn.close()


def import_metadata(work, source):
    """Reuse this run's live inspection instead of repeating the large count scan."""
    for name in ("sensors", "variables", "units", "locations", "sensoroutput", "sensortypes", "columns"):
        shutil.copyfile(source / (name + ".json"), work / (name + ".json"))
    dump(work / "DATAVALUES_counts.json", load(source / "counts.json"))
    extra = load(source / "extra_metadata.json")
    dump(work / "DATAVALUESREU24_counts.json", {
        "oracle_snapshot": "separate live inspection; see capture_utc",
        "channels": [dict(zip(("sensorid", "variableid", "rows", "first_time", "last_time"), r))
                     for r in extra["reu_counts"]["rows"]]})
    shutil.copyfile(source / "extra_metadata.json", work / "extra_metadata.json")


def channel_catalog(work):
    sensors = {int(r["sensorid"]): r for r in load(work / "sensors.json")}
    locations = {int(r["locationid"]): r for r in load(work / "locations.json")}
    variables = {int(r["variableid"]): r for r in load(work / "variables.json")}
    units = {int(r["unitsid"]): r for r in load(work / "units.json")}
    declared = set()
    for r in load(work / "sensoroutput.json"):
        sid, vid = int(r["sensorid"]), int(r["outputvariableid"])
        if vid in QUANTITIES:
            table = sensors[sid]["datatablename"].upper()
            if table not in TABLES:
                raise ValueError(f"Unexpected hydrology source table: {table}")
            declared.add((table, sid, vid))
    actual = {}
    for table in TABLES:
        for r in load(work / (table + "_counts.json"))["channels"]:
            actual[(table, int(r["sensorid"]), int(r["variableid"]))] = r
    channels = []
    for table, sid, vid in sorted(declared | set(actual)):
        s, v = sensors[sid], variables[vid]
        loc = locations[int(s["locationid"])]
        folder, suffix, unit, code, stype = QUANTITIES[vid]
        if (units[int(v["variableunitsid"])]["unitsabbreviation"] != unit
                or v["variablecode"] != code or int(s["sensortypeid"]) != stype):
            raise ValueError(f"Physical quantity/type/unit conflict: {table}/{sid}/{vid}")
        stem = s["sensorcode"].rsplit("_", 1)[0]
        coordinate_code_conflict = False
        standard = re.fullmatch(r"LEO-W_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(\d+)", stem)
        if standard:
            y, x, level = standard.groups()
            if (float(x), float(y), str(int(level))) != (
                    float(loc["boxx"]), float(loc["boxy"]), str(loc["dlevel"])):
                coordinate_code_conflict = True
                stem = f"LEO-W_{float(loc['boxy']):g}_{float(loc['boxx']):g}_{loc['dlevel']}"
            spatial_note = "Oracle box coordinates; BOXZ is depth below surface"
        else:
            # CatWalk coordinates are shared registry coordinates, not probe positions.
            stem = f"LEO-W_{float(loc['boxy']):g}_{float(loc['boxx']):g}_{loc['dlevel']}_" + stem.removeprefix("LEO-W_")
            spatial_note = "Shared CatWalk registry location; individual probe position/depth not established"
        r = actual.get((table, sid, vid), {})
        channels.append(dict(
            table="LEO_WEST." + table, sensorid=sid, variableid=vid,
            sensorcode=s["sensorcode"], sensortypeid=stype, locationid=int(s["locationid"]),
            x=loc["boxx"], y=loc["boxy"], depth_m=loc["boxz"], dlevel=loc["dlevel"],
            localz=loc["localz"], boxverticaldatum=loc["boxverticaldatum"],
            locationcode=loc["locationcode"], coordinate_code_conflict=coordinate_code_conflict,
            spatial_note=spatial_note, sensor_note=s["note"], units=unit,
            nodatavalue=v["nodatavalue"], declared_output=(table, sid, vid) in declared,
            registry_table=s["datatablename"], folder=folder, file=stem + "_" + suffix + ".parquet",
            initial_rows=int(r.get("rows", 0)), initial_first=r.get("first_time"),
            initial_last=r.get("last_time")))
    names = Counter((c["folder"], c["file"]) for c in channels)
    for c in channels:
        if names[c["folder"], c["file"]] > 1:
            c["file"] = c["file"].removesuffix(".parquet") + "_" + c["table"].split(".")[1].lower() + f"_s{c['sensorid']}.parquet"
    if len({(c["folder"], c["file"]) for c in channels}) != len(channels):
        raise ValueError("Duplicate target names")
    return channels


def month_blocks(channels):
    result = []
    for table in TABLES:
        selected = [c for c in channels if c["table"] == "LEO_WEST." + table and c["initial_rows"]]
        if not selected:
            continue
        first = min(pd.Timestamp(c["initial_first"]) for c in selected)
        last = max(pd.Timestamp(c["initial_last"]) for c in selected)
        # Calendar months cover both actual end points, including the current month.
        for a in pd.date_range(first.to_period("M").start_time, last.to_period("M").start_time, freq="MS"):
            b = a + pd.offsets.MonthBegin()
            result.append((table, a.to_pydatetime(), b.to_pydatetime()))
    return result


def raw_batch(rows):
    return pa.Table.from_pydict({
        "valueid": [int(r[0]) for r in rows], "sensorid": [int(r[1]) for r in rows],
        "variableid": [int(r[2]) for r in rows], "localdatetime": [r[3] for r in rows],
        "datavalue": [float(r[4]) if r[4] is not None else None for r in rows],
        "datavalue_oracle_decimal": [str(r[4]) if r[4] is not None else None for r in rows],
    }, schema=RAW_SCHEMA)


def download_block(work, table, a, b, catalog):
    destination = work / "blocks" / table / a.strftime("%Y-%m")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest = destination / "manifest.json"
    if manifest.exists():
        saved = load(manifest)
        for r in saved["channels"]:
            p = destination / r["file"]
            if sha(p) != r["sha256"] or pq.ParquetFile(p).metadata.num_rows != r["rows"]:
                raise ValueError(f"Checkpoint integrity failure: {p.name}")
        return saved
    allowed = {(c["sensorid"], c["variableid"]) for c in catalog if c["table"] == "LEO_WEST." + table}
    conn = connect()
    try:
        with TemporaryDirectory(prefix=".block-", dir=destination.parent) as tmp:
            stage = Path(tmp)
            batches = []
            phase = time.monotonic()
            with conn.cursor() as cur:
                    cur.execute("SET TRANSACTION READ ONLY")
                    cur.execute("SELECT sysdate FROM dual")
                    snapshot = str(cur.fetchone()[0])
                    cur.execute(COUNT_SQL.format(table=table, window=" AND localdatetime>=:a AND localdatetime<:b"), {"a": a, "b": b})
                    expected = {(int(s), int(v)): (int(n), first, last) for s, v, n, first, last in cur.fetchall()}
                    count_seconds = time.monotonic() - phase
                    if not set(expected) <= allowed:
                        raise ValueError("New actual channel not covered by initial inventory; refresh metadata")
                    cur.arraysize = 50000
                    cur.prefetchrows = 50000
                    phase = time.monotonic()
                    cur.execute(DATA_SQL.format(table=table), {"a": a, "b": b})
                    while rows := cur.fetchmany(100000):
                        batches.append(raw_batch(rows))
            fetch_seconds = time.monotonic() - phase
            phase = time.monotonic()
            batch = pa.concat_tables(batches).combine_chunks() if batches else pa.Table.from_batches([], schema=RAW_SCHEMA)
            del batches
            groups = batch.select(["sensorid", "variableid"]).to_pandas().groupby(["sensorid", "variableid"], sort=False).indices
            for key, positions in groups.items():
                pq.write_table(batch.take(pa.array(positions)), stage / f"{key[0]}_{key[1]}.parquet", compression="zstd")
            reports = []
            for key, (count, first, last) in sorted(expected.items()):
                path = stage / f"{key[0]}_{key[1]}.parquet"
                data = pd.read_parquet(path, columns=["localdatetime"])
                if len(data) != count or data.localdatetime.min() != first or data.localdatetime.max() != last:
                    raise ValueError(f"Oracle count/bounds mismatch: {table}/{key}/{a:%Y-%m}")
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                reports.append(dict(sensorid=key[0], variableid=key[1], rows=count,
                                    first_time=str(first), last_time=str(last), file=path.name, sha256=sha(path)))
            if set(groups) != set(expected):
                raise ValueError("Downloaded channel set differs from source count query")
            saved = dict(table="LEO_WEST." + table, start=a, end_exclusive=b,
                         oracle_snapshot=snapshot, rows=sum(r["rows"] for r in reports), channels=reports,
                         reader_code_sha256=sha(__file__),
                         timings_seconds=dict(count=count_seconds, fetch=fetch_seconds,
                                              local_write_and_verify=time.monotonic()-phase),
                         count_sql=COUNT_SQL.format(table=table, window=" AND localdatetime>=:a AND localdatetime<:b"),
                         data_sql=DATA_SQL.format(table=table))
            dump(stage / "manifest.json", saved)
            os.rename(stage, destination)
            # TemporaryDirectory handles the now-absent old path.
            return saved
    finally:
        conn.close()


def output_frame(raw):
    raw = raw.sort_values(["localdatetime", "valueid"], kind="stable")
    # Duplicate measurement times are intentionally retained.
    return raw.set_index("localdatetime")[["datavalue"]].astype("float64"), raw


@dataclass(frozen=True)
class RowFilters:
    start: datetime | None = None
    end: datetime | None = None
    keep_service_values: bool = False
    drop_nonfinite: bool = False
    min_value: float | None = None
    max_value: float | None = None

    def overlaps(self, first, end_exclusive):
        return ((self.start is None or self.start < end_exclusive)
                and (self.end is None or self.end > first))

    def apply(self, source, nodatavalue):
        if self.start is not None:
            source = source.loc[source.localdatetime.ge(self.start)]
        if self.end is not None:
            source = source.loc[source.localdatetime.lt(self.end)]
        service = source.datavalue.eq(float(nodatavalue))
        counts = dict(source_rows_in_window=len(source), removed_service_rows=0,
                      removed_nonfinite_rows=0, removed_by_value_range=0)
        if not self.keep_service_values:
            counts["removed_service_rows"] = int(service.sum())
            source = source.loc[~service]
        if self.drop_nonfinite:
            finite = np.isfinite(source.datavalue)
            counts["removed_nonfinite_rows"] = int((~finite).sum())
            source = source.loc[finite]
        if self.min_value is not None or self.max_value is not None:
            valid = source.datavalue.notna()
            if self.min_value is not None:
                valid &= source.datavalue.ge(self.min_value)
            if self.max_value is not None:
                valid &= source.datavalue.le(self.max_value)
            counts["removed_by_value_range"] = int((~valid).sum())
            source = source.loc[valid]
        return source, counts


def build_outputs(work, catalog, blocks, output=ROW, *, filters=RowFilters(), replace=False):
    output = Path(output)
    names = sorted({c["folder"] for c in catalog})
    for name in names:
        existing = output / name
        if existing.exists() or existing.is_symlink():
            if not replace or existing.is_symlink() or not existing.is_dir():
                raise FileExistsError(f"Export directory already exists: {existing}")
            old_files = set(pd.read_csv(existing / "channels.csv")["file"])
            recognized = old_files | {"channels.csv", "duplicate_times.csv", "provenance.json"}
            if {p.name for p in existing.iterdir()} != recognized:
                raise ValueError(f"Existing export contains extra or missing files: {existing}")
    with TemporaryDirectory(prefix=".hydrology-export-", dir=output) as tmp:
        ready = Path(tmp)
        reports = write_outputs(work, catalog, blocks, ready, filters)
        # Keep each previous directory until the new directories have all moved.
        backup_dir = Path(mkdtemp(prefix=".hydrology-previous-", dir=output)) if replace else None
        moved_old, moved_new = [], []
        try:
            for name in names:
                target = output / name
                if target.exists():
                    backup = backup_dir / name
                    os.rename(target, backup)
                    moved_old.append((backup, target))
                os.rename(ready / name, target)
                moved_new.append((target, ready / name))
        except BaseException:
            for source, destination in reversed(moved_new):
                os.rename(source, destination)
            for source, destination in reversed(moved_old):
                os.rename(source, destination)
            if backup_dir is not None:
                backup_dir.rmdir()
            raise
        if backup_dir is not None:
            shutil.rmtree(backup_dir)
    return reports


def write_outputs(work, catalog, blocks, ready, filters):
    duplicate_rows, reports = [], []
    keys = {(c["table"], c["sensorid"], c["variableid"]) for c in catalog}
    by_channel = {}
    source_blocks = []
    for block in blocks:
        if not filters.overlaps(pd.Timestamp(block["start"]), pd.Timestamp(block["end_exclusive"])):
            continue
        selected = [r for r in block["channels"]
                    if (block["table"], r["sensorid"], r["variableid"]) in keys]
        source_blocks.append(dict(table=block["table"], start=block["start"],
                                  end_exclusive=block["end_exclusive"],
                                  oracle_snapshot=block["oracle_snapshot"], channels=selected))
        directory = work / "blocks" / block["table"].split(".")[1] / str(block["start"])[:7]
        for r in selected:
            key = (block["table"], r["sensorid"], r["variableid"])
            by_channel.setdefault(key, []).append((directory / r["file"], r))
    for c in catalog:
        parts = by_channel.get((c["table"], c["sensorid"], c["variableid"]), [])
        frames = []
        for p, part in parts:
            if part.get("sha256") and sha(p) != part["sha256"]:
                raise ValueError(f"Checkpoint integrity failure: {p}")
            data = pd.read_parquet(p, columns=["valueid", "localdatetime", "datavalue"])
            if len(data) != part["rows"]:
                raise ValueError(f"Checkpoint row count mismatch: {p}")
            frames.append(data)
        if frames:
            source = pd.concat(frames, ignore_index=True)
        else:
            source = pa.Table.from_batches([], schema=RAW_SCHEMA).select(["valueid", "localdatetime", "datavalue"]).to_pandas()
        if source.valueid.duplicated().any():
            raise ValueError(f"Repeated source VALUEID for {c['sensorid']}")
        kept, counts = filters.apply(source, c["nodatavalue"])
        frame, ordered = output_frame(kept)
        frame.index = frame.index.as_unit("us", round_ok=False)
        folder = ready / c["folder"]
        folder.mkdir(exist_ok=True)
        path = folder / c["file"]
        frame.to_parquet(path, compression="zstd")
        pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
        dup = ordered[ordered.duplicated("localdatetime", keep=False)]
        for stamp, group in dup.groupby("localdatetime", dropna=False):
            duplicate_rows.append(dict(table=c["table"], sensorid=c["sensorid"], variableid=c["variableid"],
                                       localdatetime=stamp, rows=len(group),
                                       distinct_values=group.datavalue.nunique(dropna=False),
                                       valueids=";".join(map(str, group.valueid))))
        size = path.stat().st_size
        report = dict(c, **counts, source_block_rows=len(source), rows=len(frame),
                      first_time=str(frame.index.min()) if len(frame) else None,
                      last_time=str(frame.index.max()) if len(frame) else None,
                      service_rows=int(frame.datavalue.eq(float(c["nodatavalue"])).sum()),
                      null_or_nonfinite_rows=int((~np.isfinite(frame.datavalue)).sum()),
                      duplicate_time_rows=len(dup), duplicate_times=int(dup.localdatetime.nunique()),
                      duplicate_time_value_rows=int(ordered.duplicated(["localdatetime", "datavalue"], keep=False).sum()),
                      exported_valueids_sha256=hashlib.sha256(ordered.valueid.to_numpy(dtype="<i8").tobytes()).hexdigest(),
                      bytes=size, sha256=sha(path), status=("filtered_out" if counts["source_rows_in_window"] else "no_rows") if not len(frame) else
                      ("only_service_values" if frame.datavalue.eq(float(c["nodatavalue"])).all() else "has_rows"))
        reports.append(report)
    for vid, (name, *_rest) in QUANTITIES.items():
        selected = [r for r in reports if r["variableid"] == vid]
        if not selected:
            continue
        directory = ready / name
        pd.DataFrame(selected).drop(columns="folder").to_csv(directory / "channels.csv", index=False)
        duplicates = [r for r in duplicate_rows if r["variableid"] == vid]
        pd.DataFrame(duplicates, columns=["table", "sensorid", "variableid", "localdatetime", "rows", "distinct_values", "valueids"]).to_csv(directory / "duplicate_times.csv", index=False)
        records = [dict(b, channels=[r for r in b["channels"] if r["variableid"] == vid])
                   for b in source_blocks if any(c["table"] == b["table"] for c in selected)]
        expected = sum(r["rows"] for b in records for r in b["channels"])
        totals = {key: sum(r[key] for r in selected) for key in
                  ("rows", "source_rows_in_window", "source_block_rows", "removed_service_rows",
                   "removed_nonfinite_rows", "removed_by_value_range")}
        if (totals["source_block_rows"] != expected or totals["rows"] != totals["source_rows_in_window"]
                - totals["removed_service_rows"] - totals["removed_nonfinite_rows"] - totals["removed_by_value_range"]):
            raise ValueError("Export counts do not reconcile with source and filter counts")
        dump(directory / "provenance.json", dict(
            capture=load(work / "capture.json"), quantity_variableid=vid, units=QUANTITIES[vid][2],
            exporter_code_sha256=sha(__file__), runs=load(work / "runs.json"),
            **totals, channels=len(selected), channels_with_rows=sum(bool(r["rows"]) for r in selected),
            empty_channels=sum(not r["rows"] for r in selected),
            service_rows=sum(r["service_rows"] for r in selected),
            duplicate_time_rows=sum(r["duplicate_time_rows"] for r in selected),
            files_bytes=sum(r["bytes"] for r in selected),
            initial_rows=sum(r["initial_rows"] for r in selected),
            count_difference_from_initial=(expected - sum(r["initial_rows"] for r in selected)
                                           if filters.start is None and filters.end is None else None),
            numeric_representation="Oracle decimal -> float64, matching the CO2 sample; exact decimal strings and VALUEID retained in the ignored local checkpoints",
            time_representation="Original Oracle DATE -> naive timestamp[us]; no shift, rounding or resampling",
            filters=vars(filters),
            consistency="Each calendar month/table has its own read-only transaction: count and data share a snapshot; not one global SCN snapshot",
            channels_csv_sha256=sha(directory / "channels.csv"),
            duplicate_times_csv_sha256=sha(directory / "duplicate_times.csv"),
            source_blocks=records,
            errors=load(work / "errors.json") if (work / "errors.json").exists() else []))
    return reports


def local_blocks(work, planned):
    completed = load(work / "download_complete.json")
    blocks = []
    for table, start, end in planned:
        manifest = work / "blocks" / table / start.strftime("%Y-%m") / "manifest.json"
        block = load(manifest)
        if (block["table"] != "LEO_WEST." + table or pd.Timestamp(block["start"]) != start
                or pd.Timestamp(block["end_exclusive"]) != end
                or block["rows"] != sum(r["rows"] for r in block["channels"])):
            raise ValueError(f"Checkpoint manifest mismatch: {manifest}")
        blocks.append(block)
    if completed != {"blocks": len(blocks), "rows": sum(b["rows"] for b in blocks)}:
        raise ValueError("Source snapshot is incomplete or does not match its recorded totals")
    return blocks


def local_datetime(value):
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use an ISO date or local date and time") from exc
    if stamp.tzinfo is not None:
        raise argparse.ArgumentTypeError("Use the source clock without a time zone")
    return stamp


def select_channels(catalog, args):
    mappings = {values[1]: vid for vid, values in QUANTITIES.items()}
    variables = {mappings[name] for name in args.quantity or mappings}
    selected = [c for c in catalog if c["variableid"] in variables
                and (args.source_table is None or c["table"] == "LEO_WEST." + args.source_table)
                and (args.sensor_id is None or c["sensorid"] in args.sensor_id)
                and all(getattr(args, field) is None or float(c[field]) == getattr(args, field)
                        for field in ("x", "y", "depth_m"))]
    if args.sensor_id:
        missing = set(args.sensor_id) - {c["sensorid"] for c in selected}
        if missing:
            raise ValueError(f"Sensor IDs not found under the requested channel filters: {sorted(missing)}")
    if not selected:
        raise ValueError("No channels match the requested filters")
    if (args.min_value is not None or args.max_value is not None) and len({c["variableid"] for c in selected}) != 1:
        raise ValueError("Set value limits for a single physical quantity")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Local checkpoint directory name; reuse to resume")
    parser.add_argument("--metadata-cache", type=Path, help="Reuse the live inspection made in this run")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--download-only", action="store_true")
    modes.add_argument("--local-only", action="store_true", help="Use a complete cached version without contacting Oracle")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5), help="Independent read-only month queries (1-4)")
    parser.add_argument("--output-dir", type=Path, help="Destination for a separate export")
    parser.add_argument("--replace", action="store_true", help="Replace existing managed export folders after verification")
    parser.add_argument("--quantity", action="append", choices=[q[1] for q in QUANTITIES.values()])
    parser.add_argument("--source-table", choices=TABLES)
    parser.add_argument("--sensor-id", action="append", type=int)
    parser.add_argument("--x", type=float, help="Exact Oracle BOX X coordinate (m)")
    parser.add_argument("--y", type=float, help="Exact Oracle BOX Y coordinate (m)")
    parser.add_argument("--depth-m", type=float, help="Exact Oracle BOXZ (m)")
    parser.add_argument("--start", type=local_datetime, help="Inclusive source date/time")
    parser.add_argument("--end", type=local_datetime, help="Exclusive source date/time")
    parser.add_argument("--keep-service-values", action="store_true", help="Retain the Oracle NoDataValue code")
    parser.add_argument("--drop-nonfinite", action="store_true", help="Remove null and nonfinite results")
    parser.add_argument("--min-value", type=float, help="Inclusive lower limit in the selected quantity's units")
    parser.add_argument("--max-value", type=float, help="Inclusive upper limit in the selected quantity's units")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.version):
        parser.error("Invalid version name")
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must precede --end")
    if any(v is not None and not math.isfinite(v) for v in
           (args.x, args.y, args.depth_m, args.min_value, args.max_value)):
        parser.error("Selectors and value limits must be finite")
    if args.min_value is not None and args.max_value is not None and args.min_value > args.max_value:
        parser.error("--min-value must not exceed --max-value")
    output = args.output_dir.resolve() if args.output_dir else ROW
    selectors = (args.quantity, args.source_table, args.sensor_id, args.x, args.y,
                 args.depth_m, args.start, args.end, args.min_value, args.max_value)
    if output == ROW and (any(v is not None for v in selectors) or args.drop_nonfinite or args.keep_service_values):
        parser.error("A custom filter selection requires a separate --output-dir")
    filters = RowFilters(args.start, args.end, args.keep_service_values, args.drop_nonfinite, args.min_value, args.max_value)
    RAW.mkdir(exist_ok=True)
    with (RAW / "writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = RAW / args.version
        work.mkdir(exist_ok=True)
        if not (work / "capture.json").exists():
            if args.local_only:
                parser.error(f"No local snapshot with version {args.version}")
            if args.metadata_cache:
                import_metadata(work, args.metadata_cache)
            else:
                capture_metadata(work)
            metadata = sorted(p for p in work.glob("*.json"))
            dump(work / "capture.json", dict(
                version=args.version, capture_utc=datetime.now(timezone.utc).isoformat(),
                base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                code_sha256=sha(__file__), source_tables=["LEO_WEST." + t for t in TABLES],
                initial_count_sql=COUNT_SQL.format(table="{table}", window=""),
                metadata_sha256={p.name: sha(p) for p in metadata}))
        capture = load(work / "capture.json")
        for name, expected in capture["metadata_sha256"].items():
            if sha(work / name) != expected:
                raise ValueError(f"Metadata checksum conflict: {name}")
        catalog = channel_catalog(work)
        planned = month_blocks(catalog)
        if not args.download_only:
            try:
                selected = select_channels(catalog, args)
            except ValueError as exc:
                parser.error(str(exc))
        if args.local_only:
            blocks = local_blocks(work, planned)
            output.mkdir(parents=True, exist_ok=True)
            reports = build_outputs(work, selected, blocks, output, filters=filters, replace=args.replace)
            print("COMPLETE", len(reports), "channels", sum(r["rows"] for r in reports), "rows", flush=True)
            return
        dump(work / "catalog.json", catalog)
        runs = load(work / "runs.json") if (work / "runs.json").exists() else []
        runs.append(dict(utc=datetime.now(timezone.utc).isoformat(), code_sha256=sha(__file__), workers=args.workers))
        dump(work / "runs.json", runs)
        blocks, failures = [], []
        def one_block(item):
            table, a, b = item
            errors = []
            started = time.monotonic()
            for attempt in range(1, 4):
                try:
                    block = download_block(work, table, a, b, catalog)
                    return block, errors, time.monotonic()-started
                except Exception as exc:
                    # Never copy connection strings, credentials or full exception text.
                    codes = re.findall(r"(?:ORA|DPY|DPI)-\d+", str(exc))
                    error = dict(table=table, month=a.strftime("%Y-%m"), attempt=attempt,
                                 error_type=type(exc).__name__, error_codes=codes,
                                 utc=datetime.now(timezone.utc).isoformat())
                    errors.append(error)
            return None, errors, time.monotonic()-started
        # Each worker owns its connection, snapshot and directory. Only the main
        # thread updates progress/errors; no shared Oracle connection or writer.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one_block, item): item for item in planned}
            for future in as_completed(futures):
                block, errors, elapsed = future.result()
                if errors:
                    previous = load(work / "errors.json") if (work / "errors.json").exists() else []
                    dump(work / "errors.json", previous + errors)
                    for error in errors:
                        print("ERROR", error, flush=True)
                if block is None:
                    failures.append(errors[-1])
                else:
                    blocks.append(block)
                    print(f"{len(blocks)}/{len(planned)} {block['table']} {str(block['start'])[:7]} rows={block['rows']} seconds={elapsed:.1f}", flush=True)
                dump(work / "progress.json", dict(completed_blocks=len(blocks), total_blocks=len(planned),
                                                   downloaded_rows=sum(b["rows"] for b in blocks), failures=failures))
        blocks.sort(key=lambda b: (b["table"], str(b["start"])))
        if failures:
            raise RuntimeError("Incomplete archive: failed blocks recorded locally; run again to retry")
        dump(work / "download_complete.json", dict(blocks=len(blocks), rows=sum(b["rows"] for b in blocks)))
        if not args.download_only:
            output.mkdir(parents=True, exist_ok=True)
            reports = build_outputs(work, selected, blocks, output, filters=filters, replace=args.replace)
            dump(work / "export_complete.json", dict(channels=len(reports), rows=sum(r["rows"] for r in reports)))
            print("COMPLETE", len(reports), "channels", sum(r["rows"] for r in reports), "rows", flush=True)


if __name__ == "__main__":
    main()

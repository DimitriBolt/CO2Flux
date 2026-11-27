"""Archive LEO West temperature measurements and build offline Parquet selections."""
import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import load_workbook

from export_west_hydrology import RAW_SCHEMA, ROW, RowFilters, dump, load, local_datetime, output_frame, sha
from west_temperature_source import (
    INVENTORY,
    RAW,
    block_path,
    check_inventory,
    connect,
    download,
    inventory,
    records,
    now,
    planned_blocks,
    read_block,
)

FOLDERS = {
    "soil": "LEO_West_Soil_Temperature",
    "air": "LEO_West_Air_Temperature",
    "other": "LEO_West_Other_Temperature",
    "viz": "LEO_West_Visualization_Temperature",
}
PILOT_START, PILOT_END = pd.Timestamp("2025-05-01"), pd.Timestamp("2026-10-01")
AXES = ("x", "y", "depth_m", "height_m")


def workbook_locations(path=INVENTORY):
    workbook = load_workbook(path, read_only=True, data_only=True)
    result = {}
    try:
        for sheet in workbook:
            rows = iter(sheet.values)
            header = next(rows, ())
            if not {"SENSORID", "VARIABLEID", "SENSORCODE", "X-across", "Y-upslope"} <= set(header):
                continue
            for row in rows:
                r = dict(zip(header, row))
                if not isinstance(r["SENSORID"], int) or not isinstance(r["VARIABLEID"], int):
                    continue
                key = r["SENSORID"], r["VARIABLEID"]
                result[key] = dict(
                    code=r["SENSORCODE"],
                    x=r["X-across"],
                    y=r["Y-upslope"],
                    depth_cm=r.get("Depth [cm]"),
                    height_cm=r.get("Height [cm]"),
                )
    finally:
        workbook.close()
    return result


def classify(table, code, kind, sensor, loc, book):
    name = sensor["sensorcode"]
    datum = (loc.get("boxverticaldatum") or "").strip().lower()
    boxz = float(loc["boxz"]) if loc.get("boxz") is not None else None
    depth, height = None, None
    category, note = "other", "Other/uncertain installation; see registry"
    if code == "soilTemp":
        if "CatWalk" in name:
            note = "Temporary CatWalk installation; barrel/probe geometry, slope depth unconfirmed"
        elif kind in ("5TM", "MPS-2", "TCAV") and datum == "depth below the surface" and boxz and boxz > 0:
            category, depth, note = "soil", boxz, "Basalt depth from LOCATIONS.BOXZ and vertical datum"
    elif (code, kind) in (("AirTemp", "HMP60"), ("SonicTemp", "CSAT3")):
        if datum.startswith("height") and name.startswith("LEO-W_") and "CatWalk" not in name:
            category, note = "air", "Registered mast-air channel; physical installation and height unresolved"
            if book and book["code"] == name and book["height_cm"] is not None:
                height, note = (
                    float(book["height_cm"]) / 100,
                    "Height from LEO-West-Inventory, keyed by sensor/variable/code",
                )
            elif boxz and boxz > 0:
                height, note = boxz, "Height from LOCATIONS.BOXZ and vertical datum"
    elif code == "SurfaceTemp" and "LoadCellSurface" in name:
        note = "Type E thermocouple associated with load cell; basalt contact unconfirmed"
    elif code in ("SonicTempAvg", "AirTemp_AHU"):
        note = "Duct/ventilation air; not an above-slope measurement point"
    elif code == "waterTemp":
        note = "Water-temperature channel of pressure transducer at the base of the hillslope"
    elif code in ("DeviceTemp", "IRGA_Temp", "PyrgeoTemp"):
        note = "Instrument temperature: " + kind
    if table == "DATAVALUESVIZ":
        return "viz", depth, height, "Visualization series (SENSORTYPES.TABLENAMEVIZ); " + note
    return category, depth, height, note


def capture_methods(work, cap):
    """Preserve source method descriptions alongside the immutable metadata inventory."""
    dest = work / "temperature_methods.json"
    if dest.is_file():
        return
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        methods = records(
            cur,
            "SELECT methodid,variableid,sensortypeid,methoddescription "
            "FROM leo_west.methods WHERE " + cap["predicate"],
        )
        quality = records(cur, "SELECT sensorid,description FROM leo_west.qualitycontrol")
    dump(dest, dict(checked=now(), methods=methods, quality=quality))


def save_workbook(work, cap):
    cached = work / "inventory_source.xlsx"
    if not cached.exists():
        if sha(INVENTORY) != cap["inventory_xlsx_sha256"]:
            raise ValueError("Inventory workbook differs from the captured SHA-256")
        tmp = work / "inventory_source.xlsx.tmp"
        shutil.copyfile(INVENTORY, tmp)
        os.replace(tmp, cached)


def catalog(work, cap):
    meta = work / "inventory"
    cached = work / "inventory_source.xlsx"
    snapshot_book = cached if cached.is_file() else INVENTORY
    if sha(snapshot_book) != cap["inventory_xlsx_sha256"]:
        raise ValueError("Saved inventory workbook checksum mismatch")
    book = workbook_locations(snapshot_book)

    def indexed(name, key):
        return {int(r[key]): r for r in load(meta / (name + ".json"))}

    sensors = indexed("sensors", "sensorid")
    types = indexed("sensortypes", "sensortypeid")
    locs = indexed("locations", "locationid")
    units = indexed("units", "unitsid")
    vs = {
        k: v
        for k, v in indexed("variables", "variableid").items()
        if "temp" in (v["variablename"] + v["variablecode"]).lower()
    }
    declared = {
        (sensors[int(r["sensorid"])]["datatablename"].upper(), int(r["sensorid"]), int(r["outputvariableid"]))
        for r in load(meta / "sensoroutput.json")
        if int(r["outputvariableid"]) in vs
    }
    counts = {
        (t, int(r["sensorid"]), int(r["variableid"])): r
        for t in cap["tables"]
        for r in load(meta / (t + ".json"))
    }
    channels = []
    for t, s, v in sorted(declared | counts.keys()):
        sensor = sensors[s]
        loc = locs[int(sensor["locationid"])]
        var = vs[v]
        kind = types[int(sensor["sensortypeid"])]["sensortype"]
        b = book.get((s, v))
        cat, depth, height, note = classify(t, var["variablecode"], kind, sensor, loc, b)
        x, y = (float(loc[n]) if loc[n] is not None else None for n in ("boxx", "boxy"))
        lev = loc["dlevel"]
        base = "_".join(f"{z:g}" if isinstance(z, float) else str(z or "0") for z in (y, x, lev))
        suffix = {"soil": "soil_temperature", "air": "air_temperature"}.get(cat, var["variablecode"].lower())
        actual = counts.get((t, s, v), {})
        coded = re.fullmatch(r"LEO-W_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(\d+)_.*", sensor["sensorcode"])
        code_conflict = bool(coded and (float(coded[1]), float(coded[2]), int(coded[3])) != (y, x, int(lev)))
        discrepancy = bool(
            b
            and (
                b["code"] != sensor["sensorcode"]
                or b["x"] is not None
                and b["x"] != x
                or b["y"] is not None
                and b["y"] != y
                or depth is not None
                and b["depth_cm"] is not None
                and depth != float(b["depth_cm"]) / 100
            )
        )
        channels.append(
            dict(
                table="LEO_WEST." + t,
                sensorid=s,
                variableid=v,
                sensorcode=sensor["sensorcode"],
                sensor_type=kind,
                variablecode=var["variablecode"],
                quantity=cat,
                units=units[int(var["variableunitsid"])]["unitsabbreviation"],
                x=x,
                y=y,
                depth_m=depth,
                height_m=height,
                dlevel=lev,
                boxz=loc["boxz"],
                localz=loc["localz"],
                locationid=sensor["locationid"],
                locationcode=loc["locationcode"],
                locationname=loc["locationname"],
                boxverticaldatum=loc["boxverticaldatum"],
                verticalprojection=loc["verticalprojection"],
                location_comments=loc["comments"],
                installation=note,
                sensor_note=sensor["note"],
                coordinate_code_conflict=code_conflict,
                inventory_geometry_conflict=discrepancy,
                workbook_geometry=json.dumps(b, default=str) if b else None,
                registered=(t, s, v) in declared,
                registry_table=sensor["datatablename"],
                registry_start=sensor["startdatetime"],
                registry_end=sensor["enddatetime"],
                registry_status=sensor["status"],
                available=t not in cap["unavailable"],
                nodatavalue=float(var["nodatavalue"]) if var["nodatavalue"] is not None else None,
                initial_rows=int(actual["n"]) if actual else (0 if t not in cap["unavailable"] else None),
                initial_first=actual.get("first_time"),
                initial_last=actual.get("last_time"),
                file=f"LEO-W_{base}_{suffix}.parquet",
            )
        )
    unique_names(channels)
    return channels


def unique_names(channels):
    collisions = Counter((c["quantity"], c["file"]) for c in channels)
    for c in channels:
        if collisions[c["quantity"], c["file"]] > 1:
            key = f"{c['table'].split('.')[-1].lower()}_s{c['sensorid']}_v{c['variableid']}"
            c["file"] = c["file"][:-8] + "_" + key + ".parquet"
    if len({(c["quantity"], c["file"]) for c in channels}) != len(channels):
        raise ValueError("Channel filename collision")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version")
    modes = p.add_mutually_exclusive_group()
    for flag in ("status", "local-only", "download-only"):
        modes.add_argument("--" + flag, action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--replace", action="store_true")
    p.add_argument("--keep-service-values", action="store_true")
    for flag in ("start", "end"):
        p.add_argument("--" + flag, type=local_datetime)
    for flag in ("min-value", "max-value"):
        p.add_argument("--" + flag, type=float)
    for axis in AXES:
        name = axis.replace("_", "-")
        p.add_argument("--" + name, type=float, action="append")
        for side in ("min", "max"):
            p.add_argument("--" + name + "-" + side, type=float)
    for flag, typ in [
        ("quantity", str),
        ("source-table", str),
        ("sensor-id", int),
        ("sensor-type", str),
        ("variable-id", int),
    ]:
        p.add_argument("--" + flag, type=typ, action="append")
    a = p.parse_args(argv)
    if a.version and (a.version == "selections" or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", a.version)):
        p.error("Invalid version")
    if a.start and a.end and a.start >= a.end:
        p.error("--start must precede --end")
    for lo, hi in [(a.min_value, a.max_value)] + [
        (getattr(a, k + "_min"), getattr(a, k + "_max")) for k in AXES
    ]:
        if (
            any(v is not None and not math.isfinite(v) for v in (lo, hi))
            or lo is not None
            and hi is not None
            and lo > hi
        ):
            p.error("Invalid range")
    if any(not math.isfinite(v) for k in AXES for v in getattr(a, k) or []):
        p.error("Invalid spatial coordinate")
    return a


def select(channels, a):
    result = channels
    for field in ("sensor_id", "variable_id", "sensor_type", "source_table", "quantity"):
        keys = getattr(a, field)
        if not keys:
            continue
        key = {"sensor_id": "sensorid", "variable_id": "variableid", "source_table": "table"}.get(
            field, field
        )
        values = {str(k).upper().removeprefix("LEO_WEST.") for k in keys}
        result = [c for c in result if str(c[key]).upper().removeprefix("LEO_WEST.") in values]
    for k in AXES:
        exact, lo, hi = getattr(a, k), getattr(a, k + "_min"), getattr(a, k + "_max")
        if exact or lo is not None or hi is not None:
            result = [
                c
                for c in result
                if c[k] is not None
                and (not exact or c[k] in exact)
                and (lo is None or c[k] >= lo)
                and (hi is None or c[k] <= hi)
            ]
    if not result:
        raise ValueError("No channels match the selectors")
    if (a.min_value is not None or a.max_value is not None) and len({c["units"] for c in result}) != 1:
        raise ValueError("Temperature limits require channels with a single native unit")
    return result


def parts_index(work, cap, selected, filters, *, verify_all_files=False):
    """Authenticate the complete manifest chain; each selected file is verified when opened."""
    completion = load(work / "download_complete.json")
    signature = sha(work / "inventory" / "capture.json")
    if completion["capture_sha256"] != signature:
        raise ValueError("Inventory does not match completed snapshot")
    index, blocks = defaultdict(list), []
    signatures = {}
    total = 0
    for plan in planned_blocks(work, cap):
        t, a, b = plan
        directory = block_path(work, t, a)
        path = directory / "manifest.json"
        signatures[str(path.relative_to(work))] = sha(path)
        block = read_block(work, plan, cap, verify_parts=verify_all_files)
        total += block["rows"]
        blocks.append(
            dict(
                table=t,
                start=str(a),
                end=str(b),
                snapshot=block["snapshot"],
                rows=block["rows"],
                sha256=signatures[str(path.relative_to(work))],
                reader_sha256=block["reader_sha256"],
            )
        )
        for r in block["channels"]:
            key = "LEO_WEST." + t, r["sensorid"], r["variableid"]
            if key in selected and filters.overlaps(a, b):
                index[key].append((directory, r))
    if (
        signatures != completion["manifests"]
        or len(blocks) != completion["blocks"]
        or total != completion["rows"]
    ):
        raise ValueError("Incomplete snapshot or changed month manifest")
    return index, blocks


def load_channel(parts):
    columns = ["valueid", "localdatetime", "datavalue"]
    frames = []
    for directory, r in parts:
        path = directory / r["file"]
        if sha(path) != r["sha256"]:
            raise ValueError(f"Raw file checksum mismatch: {path}")
        data = pq.read_table(path, columns=columns)
        if len(data) != r["rows"] or data.schema.remove_metadata() != pa.schema(
            [RAW_SCHEMA.field(k) for k in columns]
        ):
            raise ValueError(f"Raw file row count or schema mismatch: {path}")
        source = data.to_pandas()
        source["source_null"] = data["datavalue"].is_null().to_numpy(zero_copy_only=False)
        frames.append(source)
    if frames:
        result = pd.concat(frames, ignore_index=True)
        if result.valueid.duplicated().any():
            raise ValueError("Repeated source VALUEID in one channel")
        return result
    frame = pa.Table.from_batches([], schema=RAW_SCHEMA).select(columns).to_pandas()
    frame["source_null"] = pd.Series(dtype=bool)
    return frame


def sample_interval(index):
    diff = np.diff(index.as_unit("us").asi8)
    positive = diff[diff > 0]
    if not len(positive):
        return None
    values, counts = np.unique(positive, return_counts=True)
    return float(values[counts.argmax()]) / 1e6


def interval_history(index):
    history = {}
    for month, subset in pd.Series(index, index=index).groupby(index.to_period("M")):
        seconds = sample_interval(pd.DatetimeIndex(subset))
        if seconds is not None and (not history or seconds != next(reversed(history.values()))):
            history[str(month)] = seconds
    return json.dumps(history, separators=(",", ":"))


def pilot_report(c, source):
    channel = c["quantity"]
    if channel not in ("air", "soil") or c["x"] is None or c["y"] is None:
        return None
    distance = math.hypot(c["x"] - 1, c["y"] - 4)
    if distance > 1:
        return None
    window = source.loc[source.localdatetime.ge(PILOT_START) & source.localdatetime.lt(PILOT_END)]
    nodata = c["nodatavalue"]
    valid, counts = RowFilters(drop_nonfinite=True, keep_service_values=nodata is None).apply(
        window, nodata or 0
    )
    occupied = valid.localdatetime.dt.floor("15min").nunique()
    raw_occupied = window.localdatetime.dt.floor("15min").nunique()
    slots = int((PILOT_END - PILOT_START) / pd.Timedelta(minutes=15))
    return dict(
        table=c["table"],
        sensorid=c["sensorid"],
        variableid=c["variableid"],
        sensor_type=c["sensor_type"],
        quantity=channel,
        x=c["x"],
        y=c["y"],
        distance_m=distance,
        depth_m=c["depth_m"],
        height_m=c["height_m"],
        first=valid.localdatetime.min(),
        last=valid.localdatetime.max(),
        raw=len(window),
        valid=len(valid),
        excluded_service=counts["removed_service_rows"],
        excluded_nonfinite=counts["removed_nonfinite_rows"],
        occupied_15min=occupied,
        empty_15min=slots - occupied,
        no_source_15min=slots - raw_occupied,
    )


def build_channel(c, parts, folder, filters, dup_writer, full):
    source = load_channel(parts)
    in_window = source
    if filters.start:
        in_window = in_window.loc[in_window.localdatetime.ge(filters.start)]
    if filters.end:
        in_window = in_window.loc[in_window.localdatetime.lt(filters.end)]
    # All currently registered temperature variables have a finite NoDataValue.
    # A future source without a sentinel still receives the finite-value check.
    nodata = c["nodatavalue"]
    if nodata is None:
        clean_filters = RowFilters(**(vars(filters) | {"keep_service_values": True}))
        kept, counts = clean_filters.apply(source, 0)
    else:
        kept, counts = filters.apply(source, nodata)
    frame, _ = output_frame(kept)
    frame.index = frame.index.as_unit("us", round_ok=False)
    nulls = int(in_window.source_null.sum())
    service = int(in_window.datavalue.eq(nodata).sum()) if nodata is not None else 0
    if len(in_window) != len(frame) + sum(
        counts[n] for n in ("removed_service_rows", "removed_nonfinite_rows", "removed_by_value_range")
    ):
        raise ValueError("Channel record balance mismatch")
    if full:
        for k, v in (
            ("valid", len(frame)),
            ("nulls", nulls),
            ("service", service),
            ("nonfinite", counts["removed_nonfinite_rows"] - nulls),
        ):
            if sum(r[k] for _, r in parts) != v:
                raise ValueError(f"Independent raw/clean balance failed: {c['file']}/{k}")
    original_dups = in_window.loc[in_window.duplicated("localdatetime", keep=False)]
    for stamp, g in original_dups.groupby("localdatetime"):
        dup_writer.writerow(
            [
                c["table"],
                c["sensorid"],
                c["variableid"],
                stamp,
                len(g),
                ";".join(map(str, sorted(g.valueid))),
                int(stamp in frame.index),
            ]
        )
    dups = frame.index.duplicated(keep=False)
    if not c["available"]:
        status = "source_unavailable"
    elif len(frame):
        status = "has_rows"
    elif not len(in_window):
        status = "no_source_rows"
    elif service == len(in_window):
        status = "only_service_values"
    elif len(in_window) == counts["removed_service_rows"] + counts["removed_nonfinite_rows"]:
        status = "only_invalid_values"
    else:
        status = "filtered_out"
    path = folder / c["file"]
    frame.to_parquet(
        path,
        compression="zstd",
        use_dictionary=["datavalue"],
        column_encoding={"localdatetime": "DELTA_BINARY_PACKED"},
    )
    restored = pd.read_parquet(path)
    pd.testing.assert_frame_equal(frame, restored, check_exact=True)
    if not restored.index.is_monotonic_increasing or restored.index.hasnans:
        raise ValueError(f"Invalid timestamps in {path}")
    vals = frame.datavalue
    delta = sum(r["rows"] for _, r in parts) - c["initial_rows"] if full and c["available"] else None
    return (
        dict(
            c,
            **counts,
            null_rows=nulls,
            nonfinite_nonnull_rows=int((~np.isfinite(in_window.datavalue)).sum()) - nulls,
            source_service_rows=service,
            rows=len(frame),
            status=status,
            min_value=vals.min(),
            max_value=vals.max(),
            first_time=frame.index.min(),
            last_time=frame.index.max(),
            source_first=in_window.localdatetime.min(),
            source_last=in_window.localdatetime.max(),
            mode_interval_seconds=sample_interval(frame.index),
            monthly_mode_changes_seconds=interval_history(frame.index),
            duplicate_time_rows=int(dups.sum()),
            duplicate_times=frame.index[dups].nunique(),
            bytes=path.stat().st_size,
            sha256=sha(path),
            rows_difference_from_inventory=delta,
        ),
        pilot_report(c, source) if full else None,
    )


def protect_output(output, partial, work):
    output = output.resolve()
    if output == ROW:
        if partial:
            raise ValueError("A filtered export requires a separate --output-dir")
        return
    if output == RAW or output == work or work in output.parents or output in ROW.parents:
        raise ValueError("Output overlaps a protected directory")
    if ROW in output.parents and RAW not in output.parents:
        raise ValueError(
            "Place local selections outside Row_data or under the raw archive's selections directory"
        )
    if RAW in output.parents and (output.relative_to(RAW).parts[0] != "selections"):
        raise ValueError("Use a directory below temperature_raw/selections for local selections")


def recover_publish(output):
    journal = output / ".temperature-publish.json"
    if not journal.exists():
        return
    j = load(journal)
    stage = output / j["staging"]
    if stage.parent != output or not stage.name.startswith(".temperature-stage-"):
        raise ValueError("Invalid temperature publication journal")
    if not j["committed"]:
        for name, existed in reversed(list(j["targets"].items())):
            if name not in FOLDERS.values():
                raise ValueError("Unrecognized publication target")
            target, old = output / name, stage / ("old-" + name)
            if old.exists() or not existed:
                if target.exists():
                    os.rename(target, stage / name)
                if old.exists():
                    os.rename(old, target)
    if stage.exists():
        shutil.rmtree(stage)
    journal.unlink()


def inventory_changes(output, channels):
    """Compare the complete registry with the previous installed version."""
    previous = {}
    for directory in FOLDERS.values():
        path = output / directory / "channels.csv"
        if path.is_file():
            for row in pd.read_csv(path, keep_default_na=False).to_dict("records"):
                previous[(row["table"], int(row["sensorid"]), int(row["variableid"]))] = row
    current = {(c["table"], c["sensorid"], c["variableid"]): c for c in channels}
    if not previous:
        return {"previous_archive": False}
    fields = (
        "sensorcode",
        "sensor_type",
        "quantity",
        "units",
        "x",
        "y",
        "depth_m",
        "height_m",
        "boxz",
        "localz",
        "dlevel",
        "locationcode",
        "registry_table",
        "registry_status",
        "nodatavalue",
    )

    def equivalent(a, b):
        if a in ("", None) or b in ("", None):
            return a in ("", None) and b in ("", None)
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return a == b

    changed = []
    for key in sorted(previous.keys() & current.keys()):
        diff = {
            f: [previous[key][f], current[key][f]]
            for f in fields
            if not equivalent(previous[key][f], current[key][f])
        }
        if diff:
            changed.append({"channel": key, "fields": diff})
    return dict(
        previous_archive=True,
        added=sorted(current.keys() - previous.keys()),
        removed=sorted(previous.keys() - current.keys()),
        metadata_changes=changed,
    )


def install(output, stage, names, replace, full):
    targets = {n: (output / n).exists() for n in names}
    if full:
        targets.update({n: True for n in FOLDERS.values() if (output / n).exists()})
    for name, existed in targets.items():
        if existed:
            folder = output / name
            if folder.is_symlink() or not replace:
                raise FileExistsError(folder)
            prov = load(folder / "provenance.json")
            catalog_file = folder / "channels.csv"
            if prov.get("dataset") != "LEO West temperature" or sha(catalog_file) != prov["catalog_sha256"]:
                raise ValueError(f"Unrecognized or modified catalogue: {folder}")
            for sidecar, key in (
                ("duplicate_times.csv", "duplicates_sha256"),
                ("pilot_coverage.csv", "pilot_sha256"),
            ):
                if prov.get(key) and sha(folder / sidecar) != prov[key]:
                    raise ValueError(f"Modified sidecar: {folder/sidecar}")
            previous = pd.read_csv(catalog_file)
            expected = set(previous.file) | {"channels.csv", "duplicate_times.csv", "provenance.json"}
            if prov.get("pilot_sha256"):
                expected.add("pilot_coverage.csv")
            if {p.name for p in folder.iterdir()} != expected:
                raise ValueError(f"Unexpected folder contents: {folder}")
            if any(sha(folder / r.file) != r.sha256 for r in previous.itertuples()):
                raise ValueError(f"Modified Parquet in previous export: {folder}")
    journal = output / ".temperature-publish.json"
    j = dict(staging=stage.name, targets=targets, committed=False)
    dump(journal, j)
    try:
        for name, existed in targets.items():
            if existed:
                os.rename(output / name, stage / ("old-" + name))
            if name in names:
                os.rename(stage / name, output / name)
        dump(journal, j | {"committed": True})
    finally:
        recover_publish(output)


def build(work, cap, channels, a, output, full):
    filters = RowFilters(a.start, a.end, a.keep_service_values, True, a.min_value, a.max_value)
    selected = {(c["table"], c["sensorid"], c["variableid"]) for c in channels}
    parts, blocks = parts_index(work, cap, selected, filters)
    output.mkdir(parents=True, exist_ok=True)
    recover_publish(output)
    changes = inventory_changes(output, channels) if full else None
    verified_channels = 0
    stage = Path(mkdtemp(prefix=".temperature-stage-", dir=output))
    try:
        groups = defaultdict(list)
        for c in channels:
            groups[c["quantity"]].append(c)
        for category, group in groups.items():
            folder = stage / FOLDERS[category]
            folder.mkdir()
            reports, pilot = [], []
            with (folder / "duplicate_times.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "table",
                        "sensorid",
                        "variableid",
                        "localdatetime",
                        "raw_rows",
                        "source_valueids",
                        "any_kept",
                    ]
                )
                for c in group:
                    key = c["table"], c["sensorid"], c["variableid"]
                    report, p = build_channel(c, parts[key], folder, filters, writer, full)
                    reports.append(report)
                    verified_channels += 1
                    if output == ROW and (verified_channels % 50 == 0 or verified_channels == len(channels)):
                        dump(work / "build_progress.json", dict(channels_checked=verified_channels, channels_total=len(channels)))
                        print(f"{verified_channels}/{len(channels)} working channels verified", flush=True)
                    if p is not None:
                        pilot.append(p)
            frame = pd.DataFrame(reports)
            frame.to_csv(folder / "channels.csv", index=False)
            pilot_sha = None
            if pilot:
                pd.DataFrame(pilot).to_csv(folder / "pilot_coverage.csv", index=False)
                pilot_sha = sha(folder / "pilot_coverage.csv")
            dump(
                folder / "provenance.json",
                dict(
                    dataset="LEO West temperature",
                    version=work.name,
                    created_utc=now(),
                    capture=cap,
                    methods_sha256=sha(work / "temperature_methods.json")
                    if (work / "temperature_methods.json").exists()
                    else None,
                    category=category,
                    inventory_changes=changes,
                    rows=int(frame.rows.sum()),
                    source_rows=int(frame.source_rows_in_window.sum()),
                    excluded=int(frame.source_rows_in_window.sum() - frame.rows.sum()),
                    files_bytes=int(frame.bytes.sum()),
                    full_archive=full,
                    selection=vars(a),
                    filters=vars(filters),
                    source_blocks=blocks,
                    code_sha256=sha(__file__),
                    source_module_sha256=sha(Path(__file__).with_name("west_temperature_source.py")),
                    row_filters_module_sha256=sha(Path(__file__).with_name("export_west_hydrology.py")),
                    catalog_sha256=sha(folder / "channels.csv"),
                    duplicates_sha256=sha(folder / "duplicate_times.csv"),
                    pilot_sha256=pilot_sha,
                    consistency="Separate monthly read-only transactions, fixed source-clock cutoff; no global SCN.",
                    data_sql_template="SELECT VALUEID,SENSORID,VARIABLEID,LOCALDATETIME,TO_CHAR(DATAVALUE,'TM9') with explicit decimal point; exact SQL in raw manifests",
                    limitations="Original source clock and unit. VIZ is separate. Unknown heights and inaccessible tables are marked.",
                    retry_events=[load(p) for p in sorted((work / "errors").glob("*.json"))],
                ),
            )
        install(output, stage, [FOLDERS[k] for k in groups], a.replace, full and output == ROW)
    finally:
        if stage.exists() and not (output / ".temperature-publish.json").exists():
            shutil.rmtree(stage)


def versions():
    return sorted(
        (
            p
            for p in RAW.glob("*")
            if p.is_dir() and p.name != "selections" and not p.is_symlink() and (p / ".lock").is_file()
        ),
        key=lambda p: p.stat().st_mtime_ns,
    )


def choose_version(a):
    if a.version:
        return RAW / a.version
    existing = versions()
    if a.local_only:
        complete = [p for p in existing if (p / "download_complete.json").is_file()]
        if not complete:
            raise ValueError("No completed raw temperature snapshot")
        return complete[-1]
    unfinished = [
        p
        for p in existing
        if not (p / "download_complete.json").is_file()
        or not a.download_only
        and not (p / "export_complete.json").is_file()
    ]
    if unfinished:
        return unfinished[-1]
    return RAW / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def status():
    for work in versions():
        progress_file = work / "progress.json"
        progress = load(progress_file) if progress_file.exists() else {"verified_blocks": 0}
        build_progress = work / "build_progress.json"
        print(
            json.dumps(
                dict(
                    version=work.name,
                    **(load(build_progress) if build_progress.is_file() else {}),
                    **progress,
                    raw_complete=(work / "download_complete.json").exists(),
                    export_complete=(work / "export_complete.json").exists(),
                ),
                ensure_ascii=False,
            )
        )
    if not versions():
        print("No temperature snapshots")


def main(argv=None):
    a = parse(argv)
    if a.status:
        status()
        return
    partial = (
        any(
            getattr(a, k) is not None
            for k in (
                "start",
                "end",
                "min_value",
                "max_value",
                "sensor_type",
                "source_table",
                "quantity",
                "sensor_id",
                "variable_id",
                *AXES,
                *(k + s for k in AXES for s in ("_min", "_max")),
            )
        )
        or a.keep_service_values
    )
    output = (a.output_dir or ROW).resolve()
    if a.download_only and (a.output_dir or partial):
        raise ValueError("--download-only always captures the full raw archive")
    RAW.mkdir(exist_ok=True)
    with (RAW / ".lock").open("a") as global_lock:
        fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        work = choose_version(a)
        protect_output(output, partial, work)
        if a.local_only and not (work / "download_complete.json").is_file():
            raise ValueError("Offline export requires a completed local snapshot")
        work.mkdir(exist_ok=True)
        with (work / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            already_downloaded = (work / "download_complete.json").exists()
            if not a.local_only and not already_downloaded:
                inventory(work)
                capture = check_inventory(work)
                save_workbook(work, capture)
                capture_methods(work, capture)
                download(work, a.workers)
            if a.download_only:
                if already_downloaded:
                    parts_index(work, check_inventory(work), set(), RowFilters(), verify_all_files=True)
                return
            cap = check_inventory(work)
            channels = select(catalog(work, cap), a)
            build(work, cap, channels, a, output, not partial)
            if output == ROW:
                dump(
                    work / "export_complete.json",
                    dict(
                        utc=now(),
                        channels=len(channels),
                        provenance_sha256={
                            c: sha(ROW / FOLDERS[c] / "provenance.json")
                            for c in {r["quantity"] for r in channels}
                        },
                    ),
                )


if __name__ == "__main__":
    main()

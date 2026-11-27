"""Independently reconcile a complete temperature export against every raw Parquet."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from export_west_hydrology import RAW_SCHEMA, ROW, dump, load, sha
from west_temperature_source import RAW, check_inventory, now, planned_blocks


def verify(work, row=ROW):
    cap = check_inventory(work)
    complete = load(work / "download_complete.json")
    installed = load(work / "export_complete.json")
    if complete["capture_sha256"] != sha(work / "inventory/capture.json"):
        raise ValueError("Capture checksum mismatch")
    expected_paths = {
        f"blocks/{table}/{start:%Y-%m}/manifest.json" for table, start, end in planned_blocks(work, cap)
    }
    if expected_paths != complete["manifests"].keys() or len(expected_paths) != complete["blocks"]:
        raise ValueError("Monthly manifest coverage mismatch")
    parts = defaultdict(list)
    raw_total = 0
    for rel in sorted(expected_paths):
        manifest = work / rel
        if sha(manifest) != complete["manifests"][rel]:
            raise ValueError(f"Manifest checksum mismatch: {rel}")
        b = load(manifest)
        if b["rows"] != sum(r["rows"] for r in b["channels"]):
            raise ValueError(f"Monthly row total mismatch: {rel}")
        raw_total += b["rows"]
        for r in b["channels"]:
            parts["LEO_WEST." + b["table"], r["sensorid"], r["variableid"]].append(
                (manifest.parent / r["file"], r)
            )
    if raw_total != complete["rows"]:
        raise ValueError("Source snapshot row count mismatch")
    seen = set()
    kept_total = 0
    paths = sorted(row.glob("LEO_West_*_Temperature/provenance.json"))
    categories = set()
    for provenance in paths:
        folder = provenance.parent
        prov = load(provenance)
        category = prov["category"]
        categories.add(category)
        if (
            prov["version"] != work.name
            or not prov["full_archive"]
            or sha(provenance) != installed["provenance_sha256"][category]
        ):
            raise ValueError(f"Wrong publication version: {folder}")
        for name, key in (
            ("channels.csv", "catalog_sha256"),
            ("duplicate_times.csv", "duplicates_sha256"),
            ("pilot_coverage.csv", "pilot_sha256"),
        ):
            if prov[key] and sha(folder / name) != prov[key]:
                raise ValueError(f"Sidecar checksum mismatch: {folder/name}")
        channels = pd.read_csv(folder / "channels.csv")
        if channels.file.duplicated().any() or set(channels.file) != {
            p.name for p in folder.glob("*.parquet")
        }:
            raise ValueError(f"Output channel filenames mismatch: {folder}")
        for c in channels.itertuples():
            key = c.table, int(c.sensorid), int(c.variableid)
            if key in seen:
                raise ValueError(f"Repeated channel: {key}")
            seen.add(key)
            path = folder / c.file
            if path.stat().st_size != c.bytes or sha(path) != c.sha256:
                raise ValueError(f"Working file checksum mismatch: {path}")
            out = pq.read_table(path)
            if (
                out.schema.names != ["datavalue", "localdatetime"]
                or out[0].type != pa.float64()
                or out[1].type != pa.timestamp("us")
                or len(out) != c.rows
            ):
                raise ValueError(f"Working schema or rows mismatch: {path}")
            offset = nraw = 0
            for raw, r in parts.pop(key, []):
                if sha(raw) != r["sha256"]:
                    raise ValueError(f"Raw checksum mismatch: {raw}")
                data = pq.read_table(raw)
                if data.schema != RAW_SCHEMA or len(data) != r["rows"]:
                    raise ValueError(f"Raw schema or rows mismatch: {raw}")
                source = data["datavalue_oracle_decimal"].cast(pa.float64())
                if not source.equals(data["datavalue"]):
                    raise ValueError(f"Decimal representation mismatch: {raw}")
                t = data["localdatetime"]
                bounds = pc.min_max(t).as_py()
                if (str(bounds["min"]), str(bounds["max"])) != (r["first"], r["last"]):
                    raise ValueError(f"Raw time bounds mismatch: {raw}")
                for name, n in (("sensorid", key[1]), ("variableid", key[2])):
                    if pc.min_max(data[name]).as_py() != {"min": n, "max": n}:
                        raise ValueError(f"Raw channel identity mismatch: {raw}")
                valid = pc.is_finite(source)
                if pd.notna(c.nodatavalue):
                    valid = pc.and_(valid, pc.not_equal(source, float(c.nodatavalue)))
                filtered = data.select(["datavalue", "localdatetime"]).filter(valid)
                if (
                    len(filtered) != r["valid"]
                    or len(data) - len(filtered) != r["nulls"] + r["service"] + r["nonfinite"]
                ):
                    raise ValueError(f"Monthly cleaning balance mismatch: {raw}")
                if not filtered.equals(out.slice(offset, len(filtered))):
                    raise ValueError(f"Source values or timestamps differ: {raw}")
                nraw += len(data)
                offset += len(filtered)
            if (
                offset != len(out)
                or nraw != c.source_rows_in_window
                or nraw != offset + c.removed_service_rows + c.removed_nonfinite_rows
            ):
                raise ValueError(f"Channel cleaning balance mismatch: {path}")
            t = out["localdatetime"].to_numpy()
            if len(t) and (
                np.any(t[1:] < t[:-1])
                or pd.Timestamp(t[0]) != pd.Timestamp(c.first_time)
                or pd.Timestamp(t[-1]) != pd.Timestamp(c.last_time)
            ):
                raise ValueError(f"Working time bounds/order mismatch: {path}")
            kept_total += offset
    if parts or len(seen) != installed["channels"] or categories != installed["provenance_sha256"].keys():
        raise ValueError("Missing publication channels or sources")
    result = dict(
        utc=now(),
        version=work.name,
        channels=len(seen),
        blocks=len(expected_paths),
        raw=raw_total,
        retained=kept_total,
        excluded=raw_total - kept_total,
    )
    dump(work / "independent_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if Path(args.version).name != args.version or args.version in (".", ".."):
        parser.error("Invalid version")
    print(verify(RAW / args.version))

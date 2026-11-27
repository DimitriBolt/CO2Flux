"""Read-only Oracle discovery and resumable monthly temperature checkpoints."""
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from export_west_hydrology import RAW_SCHEMA, ROOT, ROW, connect, dump, load, records, sha

RAW = ROW / "LEO_West_temperature_raw"
METADATA = ("sensors", "sensortypes", "variables", "sensoroutput", "locations", "units")
REQUIRED = {"SENSORID", "VARIABLEID", "LOCALDATETIME", "DATAVALUE"}
HYDRO = ROW / "LEO_West_hydrology_raw" / "20261126_full_history"
INVENTORY = ROOT / "Sensors_Description" / "LEO-West-Inventory.xlsx"


def now():
    return datetime.now(timezone.utc).isoformat()


def identifier(name):
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
        raise ValueError(f"Invalid Oracle identifier {name!r}")
    return name


def inventory(work):
    dest = work / "inventory"
    if dest.exists():
        check_inventory(work)
        return
    reference = {}
    if (HYDRO / "capture.json").is_file():
        for fname, expected in load(HYDRO / "capture.json")["metadata_sha256"].items():
            if sha(HYDRO / fname) != expected:
                raise ValueError(f"Reference metadata checksum mismatch: {fname}")
            reference[fname] = expected
    with TemporaryDirectory(dir=work) as tmp, connect() as conn, conn.cursor() as cur:
        stage = Path(tmp)
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT sysdate FROM dual")
        cutoff = cur.fetchone()[0]
        for name in METADATA:
            dump(stage / (name + ".json"), records(cur, "SELECT * FROM leo_west." + name))
        dump(stage / "sensorlog.json", records(cur, "SELECT sensorid, logdatetime, description FROM leo_west.sensorlog"))
        data = records(cur, """SELECT table_name,column_name,data_type,nullable FROM all_tab_columns
                               WHERE owner='LEO_WEST' ORDER BY table_name,column_id""")
        dump(stage / "columns.json", data)
        columns = defaultdict(set)
        for r in data:
            columns[r["table_name"]].add(r["column_name"])
        variables = [int(r["variableid"]) for r in load(stage / "variables.json")
                     if "temp" in (r["variablename"] + r["variablecode"]).lower()]
        registered = {s["datatablename"].upper() for s in load(stage / "sensors.json") if s["datatablename"]}
        tables = sorted(t for t, cols in columns.items() if REQUIRED <= cols)
        missing = {}
        for t in sorted(registered - set(tables)):
            try:
                cur.execute(f"SELECT 1 FROM leo_west.{identifier(t)} WHERE 1=0")
            except Exception as exc:
                if getattr(exc.args[0], "code", None) != 942:
                    raise
                missing[t] = "ORA-00942"
            else:
                raise RuntimeError(f"Registered source with unsupported layout: {t}")
        predicates = "variableid IN (" + ",".join(map(str, sorted(variables))) + ")"
        for t in tables:
            # A dated snapshot excludes records with future source timestamps.
            sql = (f"SELECT sensorid,variableid,count(*) as n,min(localdatetime) as first_time,"
                   f"max(localdatetime) as last_time,count(localdatetime) as dated FROM leo_west.{identifier(t)} "
                   f"WHERE {predicates} AND (localdatetime<=:cutoff OR localdatetime IS NULL) "
                   "GROUP BY sensorid,variableid")
            counts = records(cur, sql, {"cutoff": cutoff})
            if any(r["dated"] != r["n"] for r in counts):
                raise ValueError(f"Undated measurements in {t}; month assignment undefined")
            dump(stage / (t + ".json"), counts)
            print(t, sum(int(r["n"]) for r in counts), "temperature rows", flush=True)
        snapshot = dict(created_utc=now(), cutoff_local=cutoff, tables=tables, unavailable=missing,
                        predicate=predicates, reference_sha256=reference, inventory_xlsx_sha256=sha(INVENTORY),
                        base_commit=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(),
                        metadata_sha256={p.name:sha(p) for p in stage.glob("*.json")})
        dump(stage / "capture.json", snapshot)
        os.rename(stage, dest)


def check_inventory(work):
    stage = work / "inventory"
    cap = load(stage / "capture.json")
    for name, digest in cap["metadata_sha256"].items():
        if sha(stage / name) != digest:
            raise ValueError(f"Temperature inventory checksum mismatch: {name}")
    return cap


def planned_blocks(work, capture):
    from calendar import monthrange
    from datetime import timedelta
    for table in capture["tables"]:
        counts = load(work / "inventory" / (table + ".json"))
        if not counts:
            continue
        start = datetime.fromisoformat(min(row["first_time"] for row in counts)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0)
        last = datetime.fromisoformat(max(row["last_time"] for row in counts))
        while start <= last:
            end = start + timedelta(days=monthrange(start.year,start.month)[1])
            yield table, start, end
            start = end


def block_path(work, table, start):
    return work / "blocks" / table / start.strftime("%Y-%m")


def read_block(work, plan, capture, *, verify_parts):
    table, start, end = plan
    directory = block_path(work, table, start)
    saved = load(directory / "manifest.json")
    if (saved["table"], saved["start"], saved["end"], saved["capture_sha256"]) != (
        table, str(start), str(end), sha(work / "inventory" / "capture.json")
    ):
        raise ValueError(f"Block snapshot mismatch: {directory}")
    total = 0
    for r in saved["channels"]:
        n = r["rows"]
        if n != r["valid"] + r["nulls"] + r["service"] + r["nonfinite"]:
            raise ValueError(f"Block balance mismatch: {directory}")
        total += n
        if verify_parts:
            p = directory / r["file"]
            pf = pq.ParquetFile(p)
            if pf.metadata.num_rows != n or pf.schema_arrow != RAW_SCHEMA or sha(p) != r["sha256"]:
                raise ValueError(f"Damaged temperature checkpoint: {p}")
    if total != saved["rows"]:
        raise ValueError(f"Block row count mismatch: {directory}")
    return saved


def write_channel(stage, key, pieces, expected, nodata):
    data = pa.concat_tables(pieces)
    count, first, last = expected[key]
    bounds = pc.min_max(data["localdatetime"]).as_py()
    if (len(data), bounds["min"], bounds["max"]) != (count, first, last):
        raise ValueError(f"Oracle row count or time bounds mismatch for {key}")
    v = data["datavalue"]
    service = int(pc.sum(pc.equal(v, float(nodata))).as_py() or 0) if nodata is not None else 0
    finite = int(pc.sum(pc.is_finite(v)).as_py() or 0)
    nonfinite = len(v) - finite - v.null_count
    name = f"{key[0]}_{key[1]}.parquet"
    path = stage / name
    pq.write_table(data,path,compression="zstd",use_dictionary=["datavalue","datavalue_oracle_decimal"],
                   column_encoding={"valueid":"DELTA_BINARY_PACKED","localdatetime":"DELTA_BINARY_PACKED"})
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    if pq.ParquetFile(path).metadata.num_rows != count:
        raise ValueError(f"Parquet row count mismatch: {path}")
    return dict(sensorid=key[0],variableid=key[1],rows=count,first=str(first),last=str(last),
                nulls=v.null_count,service=service,nonfinite=nonfinite,valid=finite-service,
                file=name,sha256=sha(path))


def download_block(work, plan, capture, allowed, codes):
    import numpy as np
    table, start, end = plan
    destination = block_path(work, table, start)
    if destination.exists():
        return read_block(work, plan, capture, verify_parts=True)
    destination.parent.mkdir(exist_ok=True,parents=True)
    where = (f" FROM leo_west.{identifier(table)} WHERE {capture['predicate']}"
             " AND localdatetime>=:a AND localdatetime<:b AND localdatetime<=:cutoff")
    params = dict(a=start,b=end,cutoff=datetime.fromisoformat(capture["cutoff_local"]))
    count_sql = ("SELECT sensorid,variableid,count(*),min(localdatetime),max(localdatetime)" + where
                 + " GROUP BY sensorid,variableid")
    data_sql = ("SELECT cast(valueid as number(18)),cast(sensorid as number(9)),cast(variableid as number(9)),"
                "localdatetime,TO_CHAR(datavalue,'TM9','NLS_NUMERIC_CHARACTERS=''.,''')" + where
                + " ORDER BY sensorid,variableid,localdatetime,valueid")
    with TemporaryDirectory(dir=destination.parent,prefix=".month-") as tmp, connect() as conn, conn.cursor() as cur:
        stage = Path(tmp)
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT sysdate FROM dual")
        snapshot = str(cur.fetchone()[0])
        cur.execute(count_sql, params)
        expected = {(int(s),int(v)):(int(n),a,b) for s,v,n,a,b in cur}
        if not expected.keys() <= allowed:
            raise ValueError(f"New actual temperature channel in {table}; start a new inventory")
        pieces, reports, previous = [], [], None
        for batch in conn.fetch_df_batches(statement=data_sql,parameters=params,size=64000):
            data = pa.table(batch)
            if not len(data):
                continue
            exact = data.column(4).cast(pa.string())
            data = pa.Table.from_arrays([data.column(i).cast(RAW_SCHEMA[i].type) for i in range(4)]
                                        + [exact.cast(pa.float64()),exact], schema=RAW_SCHEMA)
            sensors = data["sensorid"].to_numpy()
            variables = data["variableid"].to_numpy()
            boundaries = np.flatnonzero((sensors[1:] != sensors[:-1]) | (variables[1:] != variables[:-1])) + 1
            for a,b in zip(np.r_[0,boundaries],np.r_[boundaries,len(data)]):
                key = int(sensors[a]),int(variables[a])
                if previous is not None and previous != key:
                    reports.append(write_channel(stage,previous,pieces,expected,codes[previous[1]]))
                    pieces = []
                previous = key
                pieces.append(data.slice(int(a),int(b-a)))
        if pieces:
            reports.append(write_channel(stage,previous,pieces,expected,codes[previous[1]]))
        if len(reports) != len(expected) or {(r["sensorid"],r["variableid"]) for r in reports} != expected.keys():
            raise ValueError("Missing or repeated Oracle channel")
        result = dict(table=table,start=str(start),end=str(end),snapshot=snapshot,rows=sum(r["rows"] for r in reports),
                      capture_sha256=sha(work/"inventory"/"capture.json"),reader_sha256=sha(__file__),
                      count_sql=count_sql,data_sql=data_sql,channels=reports)
        dump(stage/"manifest.json",result)
        os.rename(stage,destination)
        return result


def download(work, workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    capture = check_inventory(work)
    plans = list(planned_blocks(work,capture))
    meta = work/"inventory"
    allowed = {t:{(int(r["sensorid"]),int(r["variableid"])) for r in load(meta/(t+".json"))}
               for t in capture["tables"]}
    codes = {int(r["variableid"]):r["nodatavalue"] for r in load(meta/"variables.json")}
    # Complete the smallest sources first; all months retain their individual source transaction.
    def run(plan):
        for attempt in range(1,4):
            try:
                return download_block(work,plan,capture,allowed[plan[0]],codes)
            except Exception as exc:
                error_dir = work/"errors"
                error_dir.mkdir(exist_ok=True)
                dump(error_dir/f"{plan[0]}-{plan[1]:%Y%m}-{time.time_ns()}.json",
                     dict(utc=now(),attempt=attempt,type=type(exc).__name__,
                          oracle_code=getattr(exc.args[0],"code",None) if exc.args else None))
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    plans.sort(key=lambda p:(p[0]=="DATAVALUES", p[0],p[1]))
    total = 0
    signatures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run,p):p for p in plans}
        try:
            for i,future in enumerate(as_completed(jobs),1):
                block = future.result()
                total += block["rows"]
                table,start,_end = jobs[future]
                path = block_path(work,table,start)/"manifest.json"
                signatures[str(path.relative_to(work))] = sha(path)
                dump(work/"progress.json",dict(verified_blocks=i,planned=len(plans),rows=total))
                if i%10 == 0 or i==len(plans):
                    print(f"{i}/{len(plans)} blocks verified ({total:,} rows)",flush=True)
        except BaseException:
            for job in jobs:
                job.cancel()
            raise
    dump(work/"download_complete.json",dict(blocks=len(plans),rows=total,manifests=signatures,
                                           capture_sha256=sha(meta/"capture.json")))

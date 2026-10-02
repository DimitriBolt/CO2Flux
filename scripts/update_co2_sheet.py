from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import re

from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK_PATH = ROOT / "Sensors_Description" / "variables_schema.xlsx"
ENV_PATH = Path.home() / "Documents" / ".env"
ORACLE_CLIENT_LIB_DIR = Path("/opt/oracle/instantclient_19_26")
TODAY = date.today().isoformat()

# Current CO2 worksheet layout. Never infer IDs from unrelated numeric cells.
EXPECTED_HEADERS = {
    "B": "Physical symbol", "I": "Source system", "K": "Inventory sheet",
    "L": "Series ID", "M": "Exact source channel name",
    "N": "Oracle table / query path", "AD": "v.variableid",
    "AG": "u.unitsabbreviation",
}
SLOPE_SCHEMAS = {"LEO Center": "leo_center", "LEO East": "leo_east", "LEO West": "leo_west"}
CHANNEL_TYPES = {("C_CO2,basalt", "GMM222"): "datavalues",
                 ("C_CO2,air", "LI-COR"): "datavalueslicor"}
UPDATE_COLUMNS = ("O", "X", "Y", "Z", "AA", "AC", "AH")

@dataclass
class SeriesBounds:
    first_dt: datetime | None
    first_val: float | None
    last_dt: datetime | None
    last_val: float | None

    @property
    def has_data(self) -> bool:
        return self.first_dt is not None and self.last_dt is not None


def connect() -> oracledb.Connection:
    import oracledb
    from dotenv import dotenv_values

    cfg = dotenv_values(ENV_PATH)
    oracledb.init_oracle_client(lib_dir=str(ORACLE_CLIENT_LIB_DIR))
    dsn = oracledb.makedsn(cfg["ORACLE_HOST"], int(cfg["ORACLE_PORT"]), sid=cfg["ORACLE_SID"])
    conn = oracledb.connect(
        user=cfg["ORACLE_USER"],
        password=cfg["ORACLE_PASSWORD"],
        dsn=dsn,
    )
    conn.call_timeout = 120_000
    return conn


def fetch_bounds_sensor(
    cur: oracledb.Cursor,
    table: str,
    sensor_id: int,
    variable_id: int,
    time_col: str = "LOCALDATETIME",
    value_col: str = "DATAVALUE",
    start_date: str | None = None,
) -> SeriesBounds:
    start_filter = f"\n          AND dv.{time_col} >= DATE '{start_date}'" if start_date else ""
    q_first = f"""
        SELECT dv.{time_col}, dv.{value_col}
        FROM {table} dv
        WHERE dv.sensorid = :sensor_id
          AND dv.variableid = :variable_id
          {start_filter}
        ORDER BY dv.{time_col}
        FETCH FIRST 1 ROW ONLY
    """
    q_last = f"""
        SELECT dv.{time_col}, dv.{value_col}
        FROM {table} dv
        WHERE dv.sensorid = :sensor_id
          AND dv.variableid = :variable_id
          {start_filter}
        ORDER BY dv.{time_col} DESC
        FETCH FIRST 1 ROW ONLY
    """
    cur.execute(q_first, sensor_id=sensor_id, variable_id=variable_id)
    first_row = cur.fetchone()
    cur.execute(q_last, sensor_id=sensor_id, variable_id=variable_id)
    last_row = cur.fetchone()
    return SeriesBounds(
        first_dt=first_row[0] if first_row else None,
        first_val=first_row[1] if first_row else None,
        last_dt=last_row[0] if last_row else None,
        last_val=last_row[1] if last_row else None,
    )


def date_label(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.date().isoformat()


def end_label(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return "present" if dt.date().isoformat() == TODAY else dt.date().isoformat()


def make_series_query(table: str, sensor_id: int, variable_id: int, start_date: str | None = None) -> str:
    where_lines = [
        f"    dv.sensorid = {sensor_id}",
        f"    AND dv.variableid = {variable_id}",
    ]
    if start_date:
        where_lines.append(f"    AND dv.localdatetime >= DATE '{start_date}'")
    return (
        "SELECT\n"
        "    dv.localdatetime,\n"
        "    dv.datavalue\n"
        f"FROM\n    {table} dv\n"
        "WHERE\n"
        + "\n".join(where_lines)
        + "\nORDER BY\n"
        "    dv.localdatetime;"
    )


def make_variable_query(schema: str, variable_id: int) -> str:
    return (
        "SELECT\n"
        "    v.variableid,\n"
        "    v.variablecode,\n"
        "    v.variablename,\n"
        "    u.unitsabbreviation\n"
        f"FROM\n    {schema}.variables v\n"
        f"    LEFT JOIN {schema}.units u\n"
        "        ON v.variableunitsid = u.unitsid\n"
        "WHERE\n"
        f"    v.variableid = {variable_id}\n"
        "ORDER BY\n"
        "    v.variableid;"
    )


def resolve_sensor_id(record: dict[str, object]) -> int:
    value = record.get("L")
    if isinstance(value, bool) or value is None:
        raise ValueError("Missing or invalid sensorid in column L")
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value.strip()):
        return int(value)
    raise ValueError(f"Invalid sensorid in column L: {value!r}")


def record_from_row(ws, row_index: int, columns: list[str]) -> dict[str, object]:
    return {col: ws[f"{col}{row_index}"].value for col in columns}


def selected_records(ws, slopes):
    """Validate all selected records before issuing queries or changing cells."""
    for col, expected in EXPECTED_HEADERS.items():
        if ws[f"{col}3"].value != expected:
            raise ValueError(f"Unexpected CO2 header {col}3; expected {expected!r}")
    unknown = set(slopes) - SLOPE_SCHEMAS.keys()
    if unknown:
        raise ValueError(f"Unknown slopes: {sorted(unknown)}")
    columns = [get_column_letter(i) for i in range(1, ws.max_column + 1)]
    selected, keys, codes = [], set(), set()
    for row_index in range(4, ws.max_row + 1):
        record = record_from_row(ws, row_index, columns)
        kind = (record["B"], record["K"])
        if record["I"] not in slopes or kind not in CHANNEL_TYPES:
            continue
        sensor_id = resolve_sensor_id(record)
        schema = SLOPE_SCHEMAS[record["I"]]
        table = f"{schema}.{CHANNEL_TYPES[kind]}"
        if record["N"] != table:
            raise ValueError(f"Row {row_index}: source table conflicts with slope/type")
        variable_id = record["AD"]
        if isinstance(variable_id, bool) or not isinstance(variable_id, int):
            raise ValueError(f"Row {row_index}: variableid must be an integer in AD")
        if record["I"] == "LEO West":
            expected_id = 9 if kind[0] == "C_CO2,basalt" else 56
            if variable_id != expected_id:
                raise ValueError(f"Row {row_index}: unexpected West CO2 variableid")
        # Other slopes retain their own explicit workbook metadata.
        if not isinstance(record["M"], str) or not record["M"].strip() or not record["AG"]:
            raise ValueError(f"Row {row_index}: missing sensorcode or source units")
        for col in (*UPDATE_COLUMNS, "L", "M", "N", "AD", "AG"):
            if ws[f"{col}{row_index}"].data_type == "f":
                raise ValueError(f"Row {row_index}: refusing to overwrite/use formula in {col}")
        key = (table, sensor_id, variable_id)
        code_key = (table, record["M"], variable_id)
        if key in keys or code_key in codes:
            raise ValueError(f"Row {row_index}: ambiguous channel identity")
        keys.add(key); codes.add(code_key)
        selected.append((row_index, record))
    return selected


def update_sensor_record(cur, record):
    """Refresh availability/query fields; preserve the channel identity and units."""
    table = record["N"]
    sensor_id = resolve_sensor_id(record)
    variable_id = record["AD"]
    bounds = fetch_bounds_sensor(cur, table, sensor_id, variable_id)
    start = date_label(bounds.first_dt)
    result = dict(record)
    result.update({
        "O": f"dv.sensorid = {sensor_id} AND dv.variableid = {variable_id}",
        "X": start,
        "Y": end_label(bounds.last_dt),
        "AA": make_series_query(table, sensor_id, variable_id, start_date=start or None),
        "AC": (f"Yes — literal AA query validated in Oracle on {TODAY}." if bounds.has_data
               else f"No — literal AA query returns zero rows in Oracle on {TODAY}."),
        "AH": make_variable_query(table.split(".")[0], variable_id),
    })
    note = f"Availability checked in Oracle on {TODAY}; sensorid={sensor_id}, variableid={variable_id}."
    old_note = str(record.get("Z") or "")
    result["Z"] = old_note if note in old_note else "\n".join(filter(None, (old_note, note)))
    return result


def update_workbook(workbook, cur, *, slopes=None):
    """Update existing CO2 rows in place; never rebuild, move or delete rows.

    Other sheets, formulas, styles and non-CO2 series are outside this operation.
    Adding sensors from an inventory requires a separate, explicit operation.
    """
    ws = workbook["CO2"]
    selected = selected_records(ws, set(SLOPE_SCHEMAS) if slopes is None else set(slopes))
    updates = [(row, update_sensor_record(cur, record)) for row, record in selected]
    # Apply only after all reads succeed, without touching identity or unit cells.
    for row, record in updates:
        for col in UPDATE_COLUMNS:
            ws[f"{col}{row}"] = record[col]
    return len(updates)


def main(*, workbook_path=None, slopes=None) -> None:
    path = Path(workbook_path) if workbook_path is not None else WORKBOOK_PATH
    workbook = load_workbook(path)
    try:
        # Fail before opening Oracle when workbook metadata are inconsistent.
        selected_records(workbook["CO2"], set(SLOPE_SCHEMAS) if slopes is None else set(slopes))
        conn = connect()
        try:
            cur = conn.cursor()
            try:
                count = update_workbook(workbook, cur, slopes=slopes)
                workbook.save(path)
                print(f"Updated {count} existing CO2 rows; other rows preserved.")
            finally:
                cur.close()
        finally:
            conn.close()
    finally:
        workbook.close()


if __name__ == "__main__":
    main()

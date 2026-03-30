# climate_control_theorist_schema.xlsx: criteria and assembly algorithm

## Purpose

This note fixes the working rules for assembling and extending
`Sensors_Description/climate_control_theorist_schema.xlsx`.

The workbook is not just a sensor inventory dump. It is a curated catalog of
time-series channels that are physically meaningful for the PDE / inverse
problem and are already mapped to Oracle extraction paths.

For the `CO2` sheet, the current implementation source of truth is:

- `scripts/update_co2_sheet.py`

Helpful companion notes:

- `Sensors_Description/how_to_identify_basalt_co2_rows_for_gradient.txt`
- `Sensors_Description/CO2-описание.txt`

## Oracle connection instructions

Use this workflow when the workbook must be validated against the Oracle
SensorDB database.

### Credentials and paths

- Oracle credentials are stored in:
  `/home/dimitri/Documents/.env`
- Current Oracle Instant Client directory:
  `/opt/oracle/instantclient_19_26`

Expected `.env` keys:

- `ORACLE_HOST`
- `ORACLE_PORT`
- `ORACLE_SID`
- `ORACLE_USER`
- `ORACLE_PASSWORD`

### Important environment rule

In this environment, Oracle connection must use `python-oracledb` in
**thick mode**.

Do not default to thin mode here, because the target database requires native
network encryption / data integrity and thin mode fails with `DPY-3001`.

### Important Linux library rule

This machine exposes:

- `/lib/x86_64-linux-gnu/libaio.so.1t64`

but Oracle Instant Client expects:

- `libaio.so.1`

So before connecting, create a compatibility symlink in `/tmp` and prepend it
to `LD_LIBRARY_PATH`.

Working shell pattern:

```bash
mkdir -p /tmp/ora_compat
ln -sf /lib/x86_64-linux-gnu/libaio.so.1t64 /tmp/ora_compat/libaio.so.1
export LD_LIBRARY_PATH=/opt/oracle/instantclient_19_26:/tmp/ora_compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### Minimal Python connection pattern

```python
from pathlib import Path

import oracledb
from dotenv import dotenv_values

ENV_PATH = Path("/home/dimitri/Documents/.env")
ORACLE_CLIENT_LIB_DIR = "/opt/oracle/instantclient_19_26"

cfg = dotenv_values(ENV_PATH)

oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB_DIR)
dsn = oracledb.makedsn(
    cfg["ORACLE_HOST"],
    int(cfg["ORACLE_PORT"]),
    sid=cfg["ORACLE_SID"],
)

conn = oracledb.connect(
    user=cfg["ORACLE_USER"],
    password=cfg["ORACLE_PASSWORD"],
    dsn=dsn,
)
conn.call_timeout = 120_000
cur = conn.cursor()
```

### Query execution pattern

For time series extraction, use direct literal SQL and fetch rows in ascending
time order:

```python
cur.execute(
    """
    SELECT
        dv.localdatetime,
        dv.datavalue
    FROM
        leo_west.datavalueslicor dv
    WHERE
        dv.sensorid = 1284
        AND dv.variableid = 56
        AND dv.localdatetime >= TO_DATE('2026-03-25 00:00', 'YYYY-MM-DD HH24:MI')
        AND dv.localdatetime <= TO_DATE('2026-03-25 23:59', 'YYYY-MM-DD HH24:MI')
    ORDER BY
        dv.localdatetime
    """
)
rows = cur.fetchall()
```

### Cleanup pattern

Always close both cursor and connection:

```python
cur.close()
conn.close()
```

### Operational reminder

Before debugging workbook contents, always check first whether the issue is:

- bad metadata in Excel
- wrong `sensorid` / `variableid`
- Oracle row absence
- Oracle client library setup

When the goal is to validate workbook rows, prefer reusing the connection style
already used in:

- `scripts/update_co2_sheet.py`

## Core rules

1. One row equals one time series channel.
2. Keep only channels that have a clear physical role in the model,
   calibration, boundary conditions, controls, or validation.
3. Do not keep raw electrical channels if there is no direct physical meaning
   for the project.
4. Every kept row must contain enough metadata to reconstruct a direct Oracle
   query without further manual lookup.
5. Prefer channels that have coordinates or at least a clear spatial meaning.
6. Record actual Oracle availability, not only inventory metadata.

## Meaning of workbook sheets

- `Input`: drivers, controls, forcings, state-support channels that enter the
  PDE or inverse problem from the input side.
- `Output`: measured constraints or calibration targets.
- `CO2`: CO2-specific sheet for basalt concentration profiles, atmospheric CO2
  boundary-condition points, and a small set of support channels needed around
  the CO2 use case.

## Required row fields

For each kept row, populate the following blocks consistently:

- physical meaning: `B:K`
- Oracle extraction path: `L:O`
- units and conversion: `P:R`
- location and geometry: `S:X`
- project-use flags and scientific rationale: `Y:AC`
- ready SQL: `AD`
- Oracle validation flag: `AF`
- variable metadata: `AG:AK`

In practice, the most important operational columns are:

- `B`: physical quantity
- `C`: physical symbol
- `E`: PDE / inverse-problem role
- `F`: how the series enters the scientific workflow
- `G`: source system / slope
- `I`: instrument family
- `J`: series ID / sensor ID
- `K`: exact source channel name
- `L`: Oracle table
- `M`: Oracle selector
- `N`: time column
- `O`: value column
- `U`, `V`, `W`: coordinates
- `X`: human-readable height / depth
- `AA`, `AB`: actual availability window
- `AD`: ready-to-run SQL query
- `AF`: whether the literal `AD` query was validated in Oracle
- `AG:AJ`: variable ID, code, name, units from Oracle metadata

## General assembly algorithm

1. Decide which sheet the row belongs to.
   Put the channel into `Input`, `Output`, or `CO2` according to scientific
   role, not according to where it was found in the raw inventory.

2. Keep only physically meaningful channels.
   For inclusion, the channel must contribute to at least one of:
   boundary conditions, forcing, control, state observation, calibration
   target, transport driver, hydrologic support, or comparison reference.

3. Expand inventories to one row per real measurement point.
   Do not keep one generic row for a whole sensor family if the actual data are
   pointwise. A separate spatial point must get its own row.

4. Fill the semantic columns first.
   Assign physical quantity, symbol, tensor type, role, and a plain-language
   explanation of why the row matters for the project.

5. Fill Oracle access fields.
   Each row must have:
   table, selector, time column, value column, and a concrete SQL query.

6. Fill geometry.
   Use coordinates and height / depth whenever possible. If exact coordinates
   are unavailable, write an honest spatial meaning such as zone proxy,
   site-level reference, or biome-level control forcing.

7. Validate against Oracle.
   Run the literal query, record the real first and last timestamps, and update
   `AA`, `AB`, `AC`, and `AF`.

8. Keep the ready SQL query literal and usable.
   `AD` should work as a copy-paste query with the currently known table,
   sensor ID, variable ID, and start date.

9. Mark minimal-v1 usability honestly.
   If metadata exist but Oracle returns zero rows, set the row to not required
   for minimal v1 unless there is a strong reason to keep it visible.

10. Preserve ordering and row style.
   For generated sections, keep a stable ordering and copy the row formatting
   from an existing template row.

## CO2 sheet: selection rules

The `CO2` sheet is more strict than a generic sensor table.

### Basalt internal CO2 rows

These rows represent CO2 concentration inside basalt pore space.

Required markers:

- `B = Carbon dioxide concentration`
- `C = C_CO2,basalt`
- `I = GMM222`
- `E = Primary state / gradient driver`
- `L = leo_center.datavalues` or `leo_east.datavalues` or `leo_west.datavalues`
- `M = dv.sensorid = ... AND dv.variableid = 9`
- `AH = CO2`
- `AI = Carbon dioxide`

Sensor-code rule used by `scripts/update_co2_sheet.py`:

- `LEO-<slope>_<y>_<x>_<level>_GMM222`

Depth mapping:

- level 1 -> 5 cm below surface
- level 2 -> 20 cm below surface
- level 3 -> 35 cm below surface
- level 4 -> 50 cm below surface

Geometry rule for one basalt vertical:

- same `G`
- same `U`
- same `V`
- different depth (`W` / `X`)

This is the rule used to assemble a vertical basalt profile through one surface
location.

### Atmospheric CO2 rows

These rows represent air CO2 above the slope and act as atmospheric boundary
condition points.

Required markers:

- `C = C_CO2,air`
- `I = LI-COR`
- `L = ...datavalueslicor`
- `E = Upper boundary condition / atmospheric vertical-profile point`

Current Oracle rule fixed in the generator:

- use `variableid = 56`
- keep Oracle variable code `CO2_cellB`

Reason:

- in the current Oracle export, `variableid = 55` is largely a non-physical
  placeholder series, while `variableid = 56` carries the usable atmospheric
  CO2 signal.

### CO2 support rows

The bottom part of the `CO2` sheet may also contain a small number of support
channels if they are directly useful for interpreting CO2 transport or
boundary-layer conditions, for example:

- humidity
- air temperature
- wind
- radiation
- pressure
- hydrologic support
- relevant control forcing

These support rows are allowed only if they are clearly tied to the CO2
scientific workflow.

## CO2 sheet: operational algorithm for updates

When extending or rebuilding the `CO2` sheet:

1. Use `scripts/update_co2_sheet.py` as the reference implementation.
2. Load the workbook and keep the existing header structure.
3. Refresh already existing `LEO Center` basalt rows from Oracle.
4. Rebuild `LEO East` and `LEO West` basalt rows by scanning Oracle sensor
   codes that match the GMM222 pattern.
5. For each basalt sensor:
   parse slope, `x`, `y`, and level from the sensor code;
   convert level to physical depth;
   write `variableid = 9`;
   generate the SQL query for the full available series.
6. Refresh atmospheric LI-COR rows:
   `LEO East` -> `leo_east.datavalueslicor`
   `LEO Center` -> `leo_center.datavalueslicor`
   `LEO West` -> `leo_west.datavalueslicor`
7. For atmospheric CO2 rows, use `variableid = 56` and mark rows with zero
   Oracle output accordingly.
8. Refresh support rows and control rows with their own variable IDs and table
   logic.
9. Recompute:
   `AA` availability start,
   `AB` availability end,
   `AC` validation note,
   `AD` concrete SQL,
   `AF` live-tested status.
10. Write rows back into the worksheet, preserve style, and clear leftover old
    rows below the new output block.

## How to decide whether a row gives a usable CO2 time series near the surface

For a row to be accepted as atmospheric CO2 near the surface, check all of the
following:

1. `C = C_CO2,air`
2. `I = LI-COR`
3. `L = ...datavalueslicor`
4. `AF` confirms that the literal SQL query works in Oracle
5. `AA` and `AB` are filled with a real availability interval
6. `Z-coordinate [m]` is close to the surface, currently most importantly
   `0.25 m` for level 1

Interpretation rule:

- exact surface means `z = 0`
- near-surface currently means the lowest air level above the slope, which in
  this workbook is usually `z = 0.25 m`

## Current conclusion recorded from the workbook on 2026-03-29

For `CO2` rows in the current workbook:

- exact surface atmospheric CO2 rows with `z = 0` are absent
- usable near-surface atmospheric CO2 rows do exist at `z = 0.25 m`
- these near-surface rows are present for `LEO East` and `LEO West`
- `LEO Center` near-surface LI-COR rows exist as metadata, but the current
  Oracle export returns zero rows for them

Useful practical filter for near-surface atmospheric CO2:

- `C = C_CO2,air`
- `I = LI-COR`
- `Z-coordinate [m] = 0.25`
- `AF` starts with `Yes`

## Maintenance note

If the workbook structure or Oracle conventions change, update both:

- this note
- `scripts/update_co2_sheet.py`

The script is the executable truth. This note is the human-readable rulebook.

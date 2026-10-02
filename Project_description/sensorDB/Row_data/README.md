Row_data — how CO2 row data files are created
=============================================

This folder contains raw per-sensor CO2 time-series saved as parquet files. Current layout:

- `LEO_West/`  — existing LEO West parquet files (moved here)
- `LEO_East/`  — placeholder; run the generator to populate
- `LEO_Center/`— placeholder; run the generator to populate

How files are produced
----------------------

1. The workbook `Sensors_Description/variables_schema.xlsx` contains inventory and query templates.
   The script `scripts/update_co2_sheet.py` shows how sensor inventory rows and Oracle queries are constructed.

2. The generator script `sensorDB/save_leo_slope_co2_row_data.py` queries the `sensors` table in each
   slope schema and saves basalt sensors (GMM222) and near-surface air sensors (LI-7000) to parquet.

Usage
-----

Set Oracle credentials in `~/Documents/.env` then run (example):

```bash
python3 Project_description/sensorDB/save_leo_slope_co2_row_data.py west
python3 Project_description/sensorDB/save_leo_slope_co2_row_data.py east
python3 Project_description/sensorDB/save_leo_slope_co2_row_data.py center
```

Notes
-----
- The script will create files named like `LEO-W_4_-4_1_basalt.parquet` and `LEO-W_4_0_1_air.parquet`.
- If Oracle environment (Instant Client) is missing the script will attempt to set `LD_LIBRARY_PATH` and
  restart the process; if that fails, ensure `ORACLE_CLIENT_LIB_DIR` is set in your environment.
- Existing files are preserved; the script will skip files that already exist.


Full versioned LEO West refresh (53 channels)
-------------------------------------------

The permanent command for **«Обнови все 53 канала»** is, from the repository root:

```bash
make co2-refresh
make co2-status
```

`co2-refresh` uses `../co2_refresh.py`. Unlike the legacy generator above, it
**never skips a sensor because an old archive exists**. The existing
`LEO_West/`, research snapshots and previous full-refresh versions are untouched.

- If the latest version is incomplete, resume its durable manifest and raw parts.
- If the latest version is complete (including diagnostics), create a new UTC-named
  version and fetch the entire available Oracle history again.
- A failed diagnostic stage resumes diagnostics using the completed new download.
- `co2-status` reads local metadata only: status, channels complete/with saved rows,
  row counts, blocks, errors, diagnostic status, version path and log path.
- An exclusive file lock prevents two writers from downloading the same version.

Autonomous terminal session:

```bash
tmux new-session -d -s co2-full-refresh -c /home/dimitri/PycharmProjects/CO2Flux 'make co2-refresh'
tmux attach -t co2-full-refresh
# Detach without stopping the job: Ctrl-b, then d.
make co2-status
```

Before launching a second session, check `tmux list-sessions` and `make co2-status`.
The session runs independently of Codex; computer shutdown/sleep still interrupts
network work. After restart, the same `make co2-refresh` command resumes confirmed
progress. No scheduler, incremental refresh or automatic archive deletion is used.
The tmux session may disappear after the command finishes; the manifest and log
remain the authority for completion.

Storage:

```text
Project_description/sensorDB/Row_data/LEO_West_full/
  latest.json
  writer.lock
  <UTC-version>/
    manifest.json
    download.log
    raw/<block-id>/part-000000.parquet
    diagnostics/
```

`latest.json` points to the current version. Every version retains its own raw
parts and SHA-256 manifest. Logs use the host's local clock; manifest timestamps
are explicit UTC. Oracle measurement `LOCALDATETIME` is neither converted nor
assumed to share a verified timezone with another source.

Scope and identity
------------------

The loader reads the current CO2 worksheet without modifying it, reusing
`scripts/update_co2_sheet.selected_records`. It requires exactly 48 West GMM222
and 5 West LI-COR at workbook height 0.25 m. It verifies sensorcode, table, units,
X/Y and basalt depth against Oracle, and checks the actual columns and primary key.
Each source is identified by table + sensorid + variableid (9 basalt, 56 air).
Excel air Z=0.25 m and Oracle `LOCATIONS.BOXZ=0.05` use an unresolved geometry
mapping; both values are retained, without silently rewriting height or cancelling
the authorized raw download. Every alternative air distance is calculated in X/Y.

Bounds are queried from the live Oracle data, with **no imposed starting year**.
Metadata acquisition captures per-channel count/min/max once; these counts describe
that moment. Monthly before/after counts validate the actual downloaded time
blocks. The multi-hour archive is not claimed to be one global SCN snapshot:
later insertions/edits to Oracle can differ from the initial metadata counts.

All actual source columns are selected, including VALUEID. NULL, service codes,
out-of-range values and repeated measurement timestamps/values are preserved.
Oracle NUMBER/FLOAT are fetched as Python Decimal and stored without float64
conversion. Each raw Parquet part has its own inferred exact decimal schema;
read parts individually if their scales differ. A typical read is
`pyarrow.parquet.read_table(part_path).to_pandas()`; do not assume a single inferred
schema over every part directory. Only explicit diagnostic working copies use
float64 for the existing numerical cleaning function.

Recovery and verification
-------------------------

- Initial blocks are half-open calendar months, clipped to actual Oracle bounds.
- Rows stream in batches of 50,000 ordered by `(LOCALDATETIME, VALUEID)`.
- Both tables have a verified enabled VALUEID primary key and nonnull key fields.
- A batch is written to a temporary Parquet, flushed, row-count checked, renamed,
  hashed and atomically entered into the manifest. Data files and manifest directory
  entries are fsynced. Unconfirmed temporary/orphan files are not counted as progress.
- Restart verifies confirmed parts and continues with a **strictly greater** key.
  Distinct VALUEIDs sharing the same timestamp and concentration remain distinct.
- Before/after Oracle counts by sensor must equal saved counts for a completed block.
- Blocks over one million rows split into weeks/days before transfer. Repeatedly
  failing blocks split the remaining range while preserving confirmed pages.
  Dense single days still stream and checkpoint by key.
- Each block gets at most three attempts per invocation; the connection is reopened
  after an error. Unavailable day blocks are recorded as failures; independent
  blocks continue. Running the command again retries an incomplete version.
- Split parents complete only after all children and a full-range count check.
  Failed blocks never count as a completed archive.

The loader uses the existing thick-mode connector and existing private credentials;
no credentials are copied into the archive or Git. `Makefile` supplies the existing
Oracle library paths. Override `PYTHON` or `ORACLE_LIBRARY_PATH` only if needed on
another host. Dependencies: existing oracledb, python-dotenv, openpyxl, pyarrow,
pandas, numpy and the existing scientific dependencies; tmux for detached execution.

Automatic technical diagnostics
-------------------------------

After **all** blocks complete, `../co2_full_diagnostics.py` runs offline. It checks
all 16 triplets, treating 5/20/35 and 5/20/50 cm equally, and computes every distance
to the five air channels. All equally nearest channels are retained as candidates.
Raw archives are immutable; diagnostic source-row references link back to each
raw part. No historical archive is loaded to shorten or replace this refresh.

The approved `chapter02.clean_admission_measurements` pointwise filters are reused.
The two historical November 2017 exclusions apply only to their original three
sensor IDs. Monthly amplitude/H/A gates are not run. Air has an independent
conservative policy with unresolved numeric-QC flags, not a borrowed basalt upper
limit. Constant runs >=1 h remain diagnostics, not automatic exclusions.

Outputs include per-channel working/decision Parquets, raw/retained daily ranges,
yearly and meteorological-season counts (DJF/MAM/JJA/SON, December in the following
winter year), step distributions, gaps above the existing 1.5 × median-step
diagnostic threshold, plateau tables and all 16 four-channel candidates. Candidate
ranking uses actual common 2-hour bins and days, without imputation. Counts of
calendar years having any data are **not** a claim of complete annual cycles;
season/year tables expose the actual coverage. Different clocks remain provisional.

Scientific results are inserted only into the existing stage 0.2 subsection of
`Research_log/details/02_data_checks.md`. No README/PLAN scientific rewrite, FFT,
frequency/phase calculation or final vertical selection is performed by diagnostics.
Generated data remain local; `diagnostics/summary.json` records output checksums.
The loader does not commit or push automatically; publication of its scientific
result is a separate verification step within the authorized task.

Tests (synthetic source, no Oracle access):

```bash
python3 -B -m unittest discover -s Project_description/sensorDB/tests -p 'test_co2_refresh.py' -v
python3 -B -m unittest discover -s Project_description/sensorDB/tests -p 'test_co2_full_diagnostics.py' -v
```

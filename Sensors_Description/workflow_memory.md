# CO2Flux Workflow Memory

## Purpose

This file is the compact working memory for repeated CO2Flux tasks in
`Sensors_Description`.

Use it as the entry point before:

- extending `variables_schema.xlsx`
- building or extending CO2 profile viewers
- connecting to Oracle SensorDB for validation or extraction
- searching for an ideal common period for a basalt CO2 vertical

## Canonical documents

### Workbook assembly

Primary rulebook:

- `Sensors_Description/variables_schema_notes.md`

This document is the canonical memory for:

- how `variables_schema.xlsx` is structured
- how rows are selected and validated
- how the `CO2` sheet is assembled
- how basalt rows differ from atmospheric rows
- how Oracle connection is configured in this environment

### Viewer requirements

Shared requirements source:

- `Sensors_Description/co2_vertical_profile_viewer_requirements.md`

This requirements file applies to:

- `Sensors_Description/co2_vertical_profile_viewer.py`
- `Sensors_Description/co2_viewer_add_surface.py`

### CO2 SQL usage notes

Practical user-facing notes:

- `Sensors_Description/CO2-описание.txt`
- `Sensors_Description/how_to_identify_basalt_co2_rows_for_gradient.txt`

These two files contain mirrored Russian / English notes for:

- identifying basalt vs atmospheric CO2 rows
- grouping basalt verticals
- building SQL for basalt CO2
- building SQL for atmospheric CO2 in air

### Basalt vertical ideal-period analysis

Canonical workflow:

- `Project_description/sensorDB/ideal_vertical_period_workflow.md`

This document is the canonical memory for:

- resolving one basalt vertical from `(slope, x, y)` in `variables_schema.xlsx`
- extracting the shared time axis for a vertical triplet from Oracle SensorDB
- handling duplicate `(sensorid, timestamp)` rows safely
- finding the longest continuous common period with valid values and no
  outliers beyond `15%` from the period mean
- updating the corresponding Jupyter notebook for that vertical

## Repeated workflow: extending variables_schema.xlsx

When working on the workbook again, follow this order:

1. Start with `Sensors_Description/variables_schema_notes.md`.
2. Treat `scripts/update_co2_sheet.py` as the executable source of truth for
   the `CO2` sheet.
3. Preserve the rule: one row = one meaningful time series.
4. Keep Oracle extraction fields complete:
   `N`, `O`, `P`, `Q`, `AA`, `AC`, `AD:AG`.
5. Validate rows against Oracle, not only against inventory metadata.
6. For basalt CO2:
   use `B = C_CO2,basalt`, `K = GMM222`, `variableid = 9`.
7. For atmospheric CO2:
   use `B = C_CO2,air`, `K = LI-COR`, `...datavalueslicor`,
   `variableid = 56`.

## Repeated workflow: extending viewer scripts

Working rule:

- if the existing viewer is known-good, do not edit it unless necessary
- create a clone for new functionality and keep the original stable

Current viewer setup:

- stable original:
  `Sensors_Description/co2_vertical_profile_viewer.py`
- additive clone with near-surface air point:
  `Sensors_Description/co2_viewer_add_surface.py`

Shared rules for both viewers:

- use the shared requirements in
  `Sensors_Description/co2_vertical_profile_viewer_requirements.md`
- use the shared config file
  `Sensors_Description/co2_vertical_profile_viewer_config.toml`

Current extension pattern for the additive viewer:

- basalt profile coordinates come from `[profile]`
- manually chosen nearest air point comes from `[surface_air]`
- air point uses the same slope as the basalt profile
- air point is the LI-COR `Level 1 (25 cm above surface)` point
- frame times are defined by basalt timestamps only
- air CO2 uses the last known value at each basalt frame time

When making future viewer versions, preserve these decisions unless the user
explicitly changes them.

## Repeated workflow: Oracle SensorDB connection

Primary connection memory:

- see `Oracle connection instructions` in
  `Sensors_Description/variables_schema_notes.md`

Environment-specific rules already established:

- credentials live in `/home/dimitri/Documents/.env`
- use `python-oracledb` in thick mode
- Oracle Instant Client directory:
  `/opt/oracle/instantclient_19_26`
- this machine may need `LD_LIBRARY_PATH` to include both:
  `/opt/oracle/instantclient_19_26`
  and `~/.local/lib`

Expected `.env` keys:

- `ORACLE_HOST`
- `ORACLE_PORT`
- `ORACLE_SID`
- `ORACLE_USER`
- `ORACLE_PASSWORD`

## Practical reminder

Before doing the same kind of work again:

1. Read this file.
2. Open the canonical document for the specific task.
3. Reuse the established conventions instead of inventing new ones.
4. Prefer extending by clone when the current scientific script is already
   trusted by the user.

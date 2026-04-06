# Basalt Vertical Ideal-Period Workflow

## Purpose

This note stores the proven workflow for finding an **ideal common period**
for one basalt CO2 vertical and saving the result into a Jupyter notebook.

The workflow now supports two modes:

- `3-sensor mode`: only the basalt triplet
- `4-sensor mode`: the basalt triplet plus one surface-air LI-COR point at
  `25 cm above surface`

Validated examples:

- `3 sensors`
  - slope: `LEO East`
  - basalt x: `1`
  - basalt y: `10`
  - resolved sensor IDs: `408`, `420`, `432`
  - validated ideal period:
    `2024-08-26 06:30:01` to `2024-12-27 07:15:02`
- `4 sensors with freshness-limited air`
  - slope: `LEO West`
  - basalt x: `1`
  - basalt y: `4`
  - air x: `0`
  - air y: `4`
  - resolved sensor IDs: `996`, `1012`, `1035`, `1275`
  - air modal step: `60 min`
  - air freshness limit: `90 min`
  - validated ideal period:
    `2025-09-30 03:15:00` to `2025-10-03 10:45:00`

Important correction for `4-sensor mode`:

- older runs that used air `last known value` for ideal-period search are now
  considered **obsolete**
- they must not be reused as validated ideal periods
- ideal-period search may still use `last known value`, but only within a
  strict freshness window derived from the air sensor cadence

## Input definition

By default, repeated ideal-period tasks should read the target vertical from
`Sensors_Description/config.toml`:

- slope from `[profile].slope`
- basalt x from `[profile].x_coord_m`
- basalt y from `[profile].y_coord_m`
- air x from `[surface_air].x_coord_m`
- air y from `[surface_air].y_coord_m`

Ignore `start_date` and `end_date` from `config.toml` for ideal-period work.

One basalt vertical is defined by fixed:

- slope `I`
- x coordinate `C`
- y coordinate `D`

and by multiple basalt depths:

- `B = C_CO2,basalt`
- `K = GMM222`
- same `(I, C, D)`
- different depth rows

For the extended `4-sensor mode`, also resolve one air point:

- same slope
- x and y from `Sensors_Description/config.toml`
- `B = C_CO2,air`
- `K = LI-COR`
- `Level 1 = 25 cm above surface`

Important config rule:

- for repeated ideal-period work, use `config.toml` as the default source of
  the target coordinates
- ignore `start_date` and `end_date` in that file

## Ideal-period definition

For one vertical, the ideal period is the **longest continuous common time
interval** in which all participating sensors satisfy all of the following:

1. the basalt triplet has a common time axis
2. if an air sensor is included, the air value at each basalt time must be
   backed by a sufficiently **fresh** real air observation
3. no aligned sensor value is `<= 0`
4. within the candidate period, each sensor stays within `15%` of its own
   period mean:
   `abs(value - mean_period) / mean_period <= 0.15`
5. the time axis is continuous with the modal step of that interval
6. no gaps are allowed, even rare ones

### Important timestamp rule for basalt data

For some basalt verticals, raw Oracle timestamps are not clean one-point-per-slot
series. A nominal `15 min` slot can contain multiple raw observations such as:

- `00:15:01`
- `00:15:03`

Treat these as the same practical slot. Therefore:

- floor basalt timestamps to `15-minute` slots
- average all raw values within each `(sensorid, slot)`

This slot-collapsing step is part of the canonical workflow for the
`4-sensor mode`, and it is safe to reuse whenever raw basalt timestamps show
second-level jitter inside one slot.

### Important timestamp rule for air data

For the air point:

- use `last known value` only with a freshness limit
- determine the practical air cadence from the **modal air step rounded to the
  nearest `15 min`**
- use:
  `air_freshness_limit = 1.5 * modal_air_step`
- for a basalt slot timestamp `t`, the air value is eligible only if:
  `t - last_real_air_timestamp <= air_freshness_limit`
- if the last real air observation is older than that limit, the air value is
  considered missing and that timestamp cannot belong to the ideal period

In other words:

- for ideal-period search, every accepted time point must be backed by a real
  measurement from every participating sensor
- for the air sensor, a carried-forward value is allowed only while it is
  still fresh under the cadence-based limit

Why this rounding is required:

- raw LI-COR timestamps often drift by seconds or a minute
- without practical rounding, the same nominal hourly air series can produce a
  misleading modal step such as `59 min`
- rounding to the nearest `15 min` preserves the intended cadence for the
  freshness rule

## Critical Oracle constraints

For basalt `...datavalues` tables, the practical performance bottleneck is:

- the useful index is on `LOCALDATETIME`
- long scans by `sensorid` alone are slow

Therefore, do **not** start by fetching one full basalt history at once.
Instead:

- query basalt by **time windows**
- include all 3 basalt sensor IDs in the same query
- process the result locally

For air `...datavalueslicor`:

- one resolved LI-COR sensor history is usually small enough to preload once
- this is practical because only one air sensor is needed

## Proven workflow

### 1. Resolve the sensors from config and Excel

Start with:

- `Sensors_Description/workflow_memory.md`
- `Sensors_Description/variables_schema_notes.md`
- `Sensors_Description/CO2-описание.txt`
- `Sensors_Description/how_to_identify_basalt_co2_rows_for_gradient.txt`

Then read `Sensors_Description/config.toml` and treat it as the default source
of the target vertical.

Then read `Sensors_Description/variables_schema.xlsx` sheet `CO2`.

For the basalt triplet, keep rows with:

- `B = C_CO2,basalt`
- `K = GMM222`
- requested slope
- requested basalt `x`
- requested basalt `y`

Record for each basalt row:

- depth label
- depth in cm
- `sensorid`
- table name
- sensor code

For the current repeated workflow, also keep the air row with:

- `B = C_CO2,air`
- `K = LI-COR`
- same slope
- air `x`
- air `y`
- `Level 1 / 25 cm above surface`

Record for the air row:

- height
- `sensorid`
- table name
- sensor code

### 2. Fetch basalt data month by month

For each monthly window:

- query `leo_<slope>.datavalues`
- filter by:
  - `localdatetime >= month_start`
  - `localdatetime < month_end`
  - `variableid = 9`
  - `sensorid IN (<triplet ids>)`
- order by `localdatetime, sensorid`

Why monthly windows:

- they align with the existing Oracle index on time
- they are fast enough to run repeatedly
- they let us detect changes in sampling regime

### 3. Preload the air series once when needed

For `4-sensor mode`:

- query the resolved air sensor from `...datavalueslicor`
- use `variableid = 56`
- fetch the full available history for that one sensor
- sort by `localdatetime`

This makes later slot-level air validation cheap.

### 4. Collapse basalt data to practical 15-minute slots

Before building a shared basalt timeline:

- convert `localdatetime` to pandas timestamps
- floor each timestamp to a `15-minute` slot
- group by `(sensorid, slot_ts)`
- replace all raw values in that slot with their mean

This step handles both:

- exact duplicates
- near-duplicates inside one nominal slot

### 5. Build the common basalt slot timeline

For each month:

- pivot to `index = slot_ts`, `columns = sensorid`
- drop rows where any of the 3 basalt sensors is missing
- sort by time

For the shared basalt monthly series, compute:

- `common_count`
- first timestamp
- last timestamp
- rounded step histogram in minutes
- per-sensor:
  - sum
  - min
  - max
  - count of `<= 0`

### 6. Align air onto the basalt slots

For `4-sensor mode`:

- compute the modal rounded air step from the raw air series
- define:
  `air_freshness_limit = 1.5 * modal_air_step`
- use the shared basalt slots as the master time axis
- align the raw air series onto those basalt slots by backward `merge_asof`
- keep only basalt slots where the matched air observation is not older than
  `air_freshness_limit`

The common analysis series for `4-sensor mode` is therefore:

- `3` basalt slot-mean series
- `1` air freshness-limited last-known-value series
- one shared `15-minute` index

### 7. Search coarse candidates first

Use monthly summaries and, when useful, day-level summaries to find the best
candidate windows quickly.

This stage is only a first pass. It is meant to:

- identify promising continuous spans
- avoid brute-forcing the full history at once
- focus exact checks near the best candidate boundaries

### 8. Refine exact boundaries on slot-level data

After finding the best coarse candidate:

- inspect the edge days around it
- locate the exact first and last good slots
- if needed, brute-force a small neighborhood around the candidate start/end

Because the air sensor may stop producing raw observations for long stretches,
this refinement must also inspect whether apparently flat air segments are
still within the freshness limit or have become stale and therefore invalid.

### 9. Verify the final exact interval

Verify on the final exact window:

- every gap equals the modal step
- all aligned sensor values are `> 0`
- each sensor stays within `15%` of its own mean on that exact interval

Only after this check should the interval be called ideal.

## Notebook update pattern

### Triplet only

Create or update **12 cells total**:

- `4` cells per sensor
- `3` basalt sensors

### Triplet plus air

Create or update **16 cells total**:

- `4` cells per sensor
- `3` basalt sensors
- `1` air sensor

For each sensor block:

1. markdown header cell
2. setup cell creating `BasaltCO2Series(...)` or `AirCO2Series(...)`
3. fetch / coverage cell
4. plot cell

Canonical block order for `4-sensor mode`:

1. air block first
2. basalt `5 cm below surface`
3. basalt `20 cm below surface`
4. the deepest basalt block for that vertical

Current basalt workbook reality for the validated `LEO West` examples:

- many verticals use `5 cm`, `20 cm`, `50 cm`
- some other verticals may use `35 cm` instead of `50 cm`
- therefore the notebook order after air should be basalt from shallower to
  deeper, using the actual resolved depths from `variables_schema.xlsx`

The first markdown cell for each sensor block should include:

- slope
- basalt `x`
- basalt `y`
- air `x` and `y` if used
- ideal period start
- ideal period end
- ideal-period step
- ideal-period duration
- depth or height
- table
- `sensorid`
- `variableid`
- sensor code

Plotting rule:

- basalt blocks should plot the `15-minute` slot-mean series for the ideal period
- air blocks should plot the freshness-limited aligned air series on the
  accepted slot index
- when practical, also overlay the real air observations as points so long
  flat carry-forward stretches are visually distinguishable

## Final notebook location and naming

Finished ideal-period notebooks should be saved in:

- `Project_description/sensorDB/Ideal_period/`

Use this filename pattern:

- `co2_ideal_period_<Slope>_x<X>_y<Y>.ipynb`

Example:

- `co2_ideal_period_LEO_West_x-1_y4.ipynb`

Naming rule:

- use the basalt vertical coordinates in the filename
- keep the slope spelling exactly readable for the user
- do not encode the air coordinates into the filename unless the user asks

Version-control rule:

- after saving the final notebook, add it to Git with `git add`
- do not commit unless the user explicitly asks for a commit

## Practical command strategy

When repeating this work through Codex:

1. read the target slope and coordinates from `Sensors_Description/config.toml`
2. resolve the basalt triplet from `variables_schema.xlsx`
3. resolve the air point from `config.toml` plus `variables_schema.xlsx`
4. fetch basalt month-by-month for all 3 sensor IDs together
5. preload the single air history if `4-sensor mode` is active
6. collapse basalt raw data to `15-minute` slot means
7. compute the modal rounded air step from raw air timestamps
8. set `air_freshness_limit = 1.5 * modal_air_step`
9. align air onto the basalt slots by freshness-limited backward matching
10. search the best coarse candidate
11. refine exact start/end on slot-level data
12. verify the final interval
13. save the finished notebook into
    `Project_description/sensorDB/Ideal_period/`
14. add the finished notebook to Git
15. order notebook blocks as:
    air first, then basalt from shallowest to deepest

## Why this workflow should be reused

This workflow avoids the failure modes that cost the most time:

- full-history Oracle scans by basalt `sensorid`
- assuming raw basalt timestamps are already one-point-per-slot
- assuming a carried-forward air value means the air sensor was actually
  producing data

It also separates:

- coarse search on monthly aggregates
- exact refinement on a small number of edge windows

That makes the method practical for the remaining verticals.

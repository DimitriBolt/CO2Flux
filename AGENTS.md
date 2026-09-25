# CO2Flux Project Instructions

## Project Overview

This project focuses on analyzing **CO2 vertical profile data** and measuring CO2 influx/outflux through basalt slopes at Biosphere 2.

**Key objectives:**
- Visualize CO2 concentration profiles in basalt
- Track CO2 influx and outflux dynamics
- Generate animated visualizations for three LEO slopes: LEO Center, LEO East, LEO West

---

## Quick Start

**Python requirements:**
- Python 3.11+ (project uses `tomllib`)
- Virtual environment via `python -m venv venv`
- Install dependencies: `pip install -r requirements.txt`

**Oracle setup:**
- Oracle credentials in `/home/dimitri/Documents/.env` or project `.env`
- Uses `python-oracledb` in **thick mode** (required for native encryption on this database)
- Oracle Instant Client: `/opt/oracle/instantclient_19_26`
- On Linux: prepend `/opt/oracle/instantclient_19_26` and `/tmp/ora_compat` to `LD_LIBRARY_PATH` to resolve libaio compatibility

**Running scripts:**
```bash
# CO2 data viewer
python3 Sensors_Description/co2_vertical_profile_viewer.py

# CO2 data viewer with surface point
python3 Sensors_Description/co2_viewer_add_surface.py

# Update CO2 sheet in variables_schema.xlsx
python3 scripts/update_co2_sheet.py
```

---

## Project Structure

### `Sensors_Description/`

**Configuration & Scripts:**
- `co2_vertical_profile_viewer.py` — Main CO2 visualization script
- `co2_viewer_add_surface.py` — Viewer with surface air point
- `co2_vertical_profile_viewer_config.toml` — Default configuration
- `co2_vertical_profile_viewer_config.local.toml.example` — Local overrides template
- `co2_vertical_profile_viewer_requirements.md` — Viewer design specifications

**Inventory & Metadata:**
- `LEO-Center-Inventory.xlsx` — LEO Center slope sensor inventory
- `LEO-East-Inventory.xlsx` — LEO East slope sensor inventory
- `LEO-West-Inventory.xlsx` — LEO West slope sensor inventory
- `LEOSensorDBdescription.pdf` — Database schema and sensor details

**Data & SQL:**
- `CO2_air.sql` — Query for atmospheric CO2 boundary conditions
- `CO2basalt.sql` — Query for CO2 profiles in basalt
- `temp_basalt.sql` — Soil temperature query (auxiliary)
- `humidity_basalt.sql` — Soil volumetric water content query (auxiliary)
- `variables_schema.xlsx` — Master workbook (CO2 sheet: basalt CO2 profiles + atmospheric CO2)
- `workflow_memory.md` — Technical reference for CO2 data

**Output Visualizations:**
- `co2_profile_LEO_*.gif` — Animated CO2 profiles by slope and location
- `co2_profile_LEO_*.jpg` — Final frames with annotations

### `scripts/`

- `update_co2_sheet.py` — Regenerate CO2 sheet in `variables_schema.xlsx` from Oracle inventory

---

## Configuration Pattern

Scripts use TOML for configuration:

- **Default config:** `Sensors_Description/co2_vertical_profile_viewer_config.toml`
- **Local overrides:** `Sensors_Description/co2_vertical_profile_viewer_config.local.toml` (create for machine-specific settings)
- **Pattern:** Scripts load defaults first, then apply local overrides if file exists

**Config sections:**
- `[profile]`: slope, x/y coordinates, date range for visualization
- `[surface_air]`: coordinates for atmospheric CO2 reference point
- `[oracle]`: machine-specific Oracle Instant Client path and environment setup

---

## Viewers

### Design Patterns

- **Visualization:** Horizontal bars with depth on y-axis, CO2 concentration (ppm) on x-axis
- **CO2 axis range:** Fixed `0...8000 ppm` across all plots for visual consistency
- **Animation:** Every available measurement (no smoothing, resampling, or filtering)
- **Time synchronization:** Use "last known value" at each frame time
- **Missing data:** Display available depths only, no interpolation
- **Output:** Animated GIF + final JPEG with annotations (time, slope, coordinates, values)

### Script Architecture

- Parameters defined at top of script (not CLI arguments)
- Clone-based approach: `co2_vertical_profile_viewer.py` (stable), `co2_viewer_add_surface.py` (additive features)
- When extending viewers, create a clone instead of editing the original

### Shared Requirements

See `Sensors_Description/co2_vertical_profile_viewer_requirements.md` for full specifications.

---

## Database

### Oracle Connection

- Credentials: `/home/dimitri/Documents/.env`
- Use **thick mode**: `oracledb.init_oracle_client(lib_dir=ORACLE_CLIENT_LIB_DIR)`
- Linux compatibility: symlink for `libaio.so.1` at `/tmp/ora_compat`
- Query pattern: fetch time series by time windows (not full sensor history at once)
- Always close: `cur.close()` and `conn.close()`

### Data Sources

**LEO Slopes:**
- **LEO Center:** `leo_center.datavalues` table
- **LEO East:** `leo_east.datavalues` table
- **LEO West:** `leo_west.datavalues` table

**CO2 measurements:**
- Variable ID 1 = CO2 concentration (ppm)
- Search by sensor ID to locate specific depths and slopes

---

## Workbook: variables_schema.xlsx

### CO2 Sheet Structure

**Focus:** Basalt CO2 profiles + atmospheric CO2 boundary conditions

Columns include:
- Physical description (A–B)
- Spatial coordinates (C–F)
- Role in problem (G–H)
- Source & inventory (I–M)
- Oracle query path (N–Q)
- Units & conversion (R–T)
- Location & geometry (U–V)
- Scientific rationale (W)
- Data availability (X–Y)
- Technical metadata (Z–AG)

### Update Procedure

```bash
python3 scripts/update_co2_sheet.py
```

This regenerates the CO2 sheet from Oracle inventory and validates data availability.

### Technical Reference

See `Sensors_Description/workflow_memory.md` for detailed CO2 data workflow and query patterns.

---

## Related Independent Projects

**Note:** This repository was separated from a larger Biosphere 2 project in May 2026.

- **Separate project:** `DigitalTwin` — RainForest climate control modeling and LSTM training
  - Location: `/home/dimitri/PycharmProjects/DigitalTwin/`
  - Focus: 64 input climate control parameters × 36 RainForest output sensors
  - Not covered by this document

---

## References

- `Sensors_Description/workflow_memory.md` — CO2 data workflow details
- `Sensors_Description/co2_vertical_profile_viewer_requirements.md` — Viewer specifications
- `Sensors_Description/LEOSensorDBdescription.pdf` — Database schema
- `/home/dimitri/PycharmProjects/DigitalTwin/` — RainForest project (separate repository)

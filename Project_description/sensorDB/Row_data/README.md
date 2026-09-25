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


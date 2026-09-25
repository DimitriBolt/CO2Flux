"""
Run a partial update of the CO2 sheet for LEO East and LEO Center only.
This is a thin wrapper around `update_co2_sheet.py` that prints progress and
logs to stdout so you can follow the generation sequentially.

Usage:
    python3 scripts/run_update_east_center.py

This script expects the same Oracle credentials file at ~/Documents/.env and the
Oracle instant client available at the path used by `update_co2_sheet.py`.
"""
from __future__ import annotations

import sys
from datetime import datetime

from openpyxl import load_workbook

import update_co2_sheet as u


def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def main() -> None:
    log("Starting partial update: East and Center only")
    conn = u.connect()
    cur = conn.cursor()

    wb = load_workbook(u.WORKBOOK_PATH)
    ws = wb["CO2"]
    columns = [u.get_column_letter(i) for i in range(1, ws.max_column + 1)]

    log("Reading existing rows from workbook")
    # Build center basalt rows by querying the inventory in Oracle (GMM222 sensors)
    log("Fetching center basalt inventory rows from Oracle (GMM222 sensors)")
    center_basalt = u.fetch_gmm222_rows(cur, "LEO_CENTER", "LEO Center", "LEO-Center-Inventory.xlsx")
    log(f"  Fetched {len(center_basalt)} center basalt rows")
    east_air = [u.record_from_row(ws, row, columns) for row in range(52, 79)]
    center_air = [u.record_from_row(ws, row, columns) for row in range(79, 106)]
    support_rows = [u.record_from_row(ws, row, columns) for row in range(133, 153)]

    # Update center basalt
    log(f"Updating center basalt sensors ({len(center_basalt)} records)")
    for i, record in enumerate(center_basalt, start=1):
        log(f"  Center basalt: processing record {i}/{len(center_basalt)} sensorid={record.get('J')}")
        u.update_sensor_record(
            cur,
            record,
            table="leo_center.datavalues",
            variable_id=9,
            variable_code="CO2",
            variable_name="Carbon dioxide",
            units="ppm",
            note_with_data=f"Literal AD query validated in Oracle on {u.TODAY}. Inventory dates were checked against the current Oracle series.",
            note_no_data=f"Literal AD query unexpectedly returned zero rows in Oracle on {u.TODAY}.",
        )

    # Fetch east basalt rows from Oracle
    log("Fetching east basalt inventory rows from Oracle (GMM222 sensors)")
    east_basalt = u.fetch_gmm222_rows(cur, "LEO_EAST", "LEO East", "LEO-East-Inventory.xlsx")
    log(f"  Fetched {len(east_basalt)} east basalt rows")

    # Update east air records
    east_air_note = (
        f"Literal AD query validated in Oracle on {u.TODAY}. For LI-7000 atmospheric CO2, variableid=56 "
        "(CO2_cellB) is retained because variableid=55 in the current Oracle export is almost entirely an "
        "exact 28.0/0.0 series, while variableid=56 carries the physically scaled concentration series. "
        "The Oracle table currently exposes the 2024 campaign window only."
    )
    log(f"Updating east air records ({len(east_air)} records)")
    for i, record in enumerate(east_air, start=1):
        log(f"  East air: processing record {i}/{len(east_air)} sensorid={record.get('J')}")
        u.update_sensor_record(
            cur,
            record,
            table="leo_east.datavalueslicor",
            variable_id=56,
            variable_code="CO2_cellB",
            variable_name="Carbon dioxide",
            units="umol/mol",
            note_with_data=east_air_note,
            note_no_data=f"Literal AD query returns zero rows in current Oracle export as of {u.TODAY}.",
        )

    # Update center air records
    center_air_note = (
        f"Sensor metadata exist, but these center-slope LI-7000 points return zero rows in the current "
        f"Oracle export as of {u.TODAY}. leo_center.datavalueslicor currently contains only sensorid 1309 "
        "(LEO-G_CTest_LI-7000), not these slope-point sensors."
    )
    log(f"Updating center air records ({len(center_air)} records)")
    for i, record in enumerate(center_air, start=1):
        log(f"  Center air: processing record {i}/{len(center_air)} sensorid={record.get('J')}")
        u.update_sensor_record(
            cur,
            record,
            table="leo_center.datavalueslicor",
            variable_id=56,
            variable_code="CO2_cellB",
            variable_name="Carbon dioxide",
            units="umol/mol",
            note_with_data=center_air_note,
            note_no_data=center_air_note,
            use_in_v1_if_missing="No",
        )
        record["Y"] = "No"

    # Update support rows (controls / other series)
    log(f"Updating support rows ({len(support_rows)} records)")
    for i, record in enumerate(support_rows, start=1):
        log(f"  Support row: processing {i}/{len(support_rows)} J={record.get('J')} G={record.get('G')}")
        series_id = int(record["J"]) if isinstance(record["J"], (int, float)) else None
        if series_id == 1275 and record["B"] == "Water vapor concentration":
            u.update_sensor_record(
                cur,
                record,
                table="leo_center.datavalueslicor",
                variable_id=58,
                variable_code="H2O_cellB",
                variable_name="Water vapor concentration",
                units="mmol/mol",
                note_with_data=center_air_note,
                note_no_data=center_air_note,
                use_in_v1_if_missing="No",
            )
            record["Y"] = "No"
            continue

        if str(record["G"]) == "Bio2 Controls":
            u.update_control_record(
                cur,
                record,
                note=f"Literal AD query validated in Oracle on {u.TODAY}. Query updated to return the full available series from the live control table.",
            )
            continue

        u.update_full_series_sensor_record(
            cur,
            record,
            note=f"Literal AD query validated in Oracle on {u.TODAY}. Query updated to return the full available series.",
        )

    # Assemble output rows (only center and east + their air + support)
    output_rows = center_basalt + east_basalt + east_air + center_air + support_rows
    start_row = 4
    template_row = 4
    last_output_row = start_row + len(output_rows) - 1

    log(f"Writing {len(output_rows)} output rows to workbook (rows {start_row}..{last_output_row})")
    if last_output_row > ws.max_row:
        ws.insert_rows(ws.max_row + 1, amount=last_output_row - ws.max_row)

    for row_index, record in enumerate(output_rows, start=start_row):
        u.apply_row_style(ws, template_row, row_index)
        for col in columns:
            ws[f"{col}{row_index}"] = record.get(col, None)

    for row_index in range(last_output_row + 1, ws.max_row + 1):
        u.clear_row_values(ws, row_index)

    wb.save(u.WORKBOOK_PATH)
    log("Workbook saved")

    cur.close()
    conn.close()
    log("Oracle connection closed")
    log("Partial update finished successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        raise


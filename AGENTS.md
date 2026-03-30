# Project Instructions

## Sensors_Description memory

For tasks in `Sensors_Description` related to:

- `climate_control_theorist_schema.xlsx`
- CO2 viewer scripts
- Oracle SensorDB access

read this file first:

- `Sensors_Description/workflow_memory.md`

Then open the canonical task-specific documents referenced from that file.

## Established conventions

### Workbook

- The canonical human-readable rulebook for extending
  `Sensors_Description/climate_control_theorist_schema.xlsx` is:
  `Sensors_Description/climate_control_theorist_schema_notes.md`
- The executable source of truth for the `CO2` sheet is:
  `scripts/update_co2_sheet.py`

### Viewers

- Shared viewer requirements live in:
  `Sensors_Description/co2_vertical_profile_viewer_requirements.md`
- If a viewer is known-good and trusted by the user, prefer creating a clone
  for new functionality instead of editing the original script.
- Current stable original:
  `Sensors_Description/co2_vertical_profile_viewer.py`
- Current additive clone:
  `Sensors_Description/co2_viewer_add_surface.py`

### Oracle

- Credentials live in:
  `/home/dimitri/Documents/.env`
- Connection instructions for this environment are documented in:
  `Sensors_Description/climate_control_theorist_schema_notes.md`

## Working style for repeated tasks

- Reuse the established conventions from the canonical docs instead of
  re-deriving them.
- When extending user-facing scientific scripts, preserve trusted behavior and
  isolate new behavior in a clone unless the user explicitly wants in-place
  edits.

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Row_data"
COMPRESSION = "zstd"
DEFAULT_ORACLE_CLIENT_LIB_DIR = Path("/opt/oracle/instantclient_19_26")
LOCAL_LIB_DIR = Path.home() / ".local/lib"
ORA_COMPAT_DIR = Path("/tmp/ora_compat")
ORA_COMPAT_LINK = ORA_COMPAT_DIR / "libaio.so.1"
ORA_COMPAT_TARGET = Path("/lib/x86_64-linux-gnu/libaio.so.1t64")
ORACLE_ENV_READY_FLAG = "CO2_ROW_DATA_ORACLE_ENV_READY"


def ensure_oracle_runtime_env() -> None:
    if os.getenv(ORACLE_ENV_READY_FLAG) == "1":
        return

    oracle_client_lib_dir = Path(
        os.getenv("ORACLE_CLIENT_LIB_DIR", str(DEFAULT_ORACLE_CLIENT_LIB_DIR))
    ).expanduser()
    if not oracle_client_lib_dir.exists():
        return

    required_entries = [str(oracle_client_lib_dir)]

    if ORA_COMPAT_TARGET.exists():
        ORA_COMPAT_DIR.mkdir(parents=True, exist_ok=True)
        if ORA_COMPAT_LINK.is_symlink() or ORA_COMPAT_LINK.exists():
            if ORA_COMPAT_LINK.resolve() != ORA_COMPAT_TARGET.resolve():
                ORA_COMPAT_LINK.unlink()
                ORA_COMPAT_LINK.symlink_to(ORA_COMPAT_TARGET)
        else:
            ORA_COMPAT_LINK.symlink_to(ORA_COMPAT_TARGET)
        required_entries.append(str(ORA_COMPAT_DIR))

    if LOCAL_LIB_DIR.exists():
        required_entries.append(str(LOCAL_LIB_DIR))

    current_entries = [
        entry for entry in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    missing_entries = [entry for entry in required_entries if entry not in current_entries]
    if not missing_entries:
        os.environ.setdefault("ORACLE_CLIENT_LIB_DIR", str(oracle_client_lib_dir))
        os.environ.setdefault(ORACLE_ENV_READY_FLAG, "1")
        return

    new_env = os.environ.copy()
    new_env["ORACLE_CLIENT_LIB_DIR"] = str(oracle_client_lib_dir)
    new_env["LD_LIBRARY_PATH"] = os.pathsep.join(required_entries + current_entries)
    new_env[ORACLE_ENV_READY_FLAG] = "1"
    restart_argv = list(getattr(sys, "orig_argv", [])) or [sys.executable, *sys.argv]
    if restart_argv:
        restart_argv[0] = sys.executable
    os.execve(sys.executable, restart_argv, new_env)


ensure_oracle_runtime_env()

from sensorDB import SensorDB  # type: ignore
from basalt_co2_series import BasaltCO2Series  # type: ignore
from air_co2_series import AirCO2Series  # type: ignore

GMM222_PATTERN = re.compile(r"^LEO-(?P<slope>[A-Z])_(?P<y>-?\d+)_(?P<x>-?\d+)_(?P<level>\d+)_GMM222$")
DEPTH_CM_BY_LEVEL = {1: 5, 2: 20, 3: 35, 4: 50}


def build_output_name(sensor_code: str, replacement: str) -> str:
    if replacement == "basalt":
        return f"{sensor_code.replace('GMM222', 'basalt')}.parquet"
    if replacement == "air":
        return f"{sensor_code.replace('LI-7000', 'air')}.parquet"
    raise ValueError(f"Unsupported replacement kind: {replacement}")


def save_series(sensor, output_path: Path) -> None:
    if output_path.exists():
        print(f"Skipping {output_path.name}: already exists", flush=True)
        return
    print(f"Fetching {output_path.name} ...", flush=True)
    series = sensor.fetch_series()
    series.to_frame().to_parquet(output_path, compression=COMPRESSION)
    print(f"Saved {output_path.name}: {len(series):,} rows", flush=True)


def fetch_and_save_basalt(cur, schema_prefix: str, slope_letter: str, out_dir: Path) -> None:
    pattern = f"LEO-{slope_letter}_%GMM222"
    cur.execute(
        f"SELECT sensorid, sensorcode FROM {schema_prefix}.sensors WHERE sensorcode LIKE :pattern ORDER BY sensorid",
        pattern=pattern,
    )
    for sensor_id, sensor_code in cur.fetchall():
        m = GMM222_PATTERN.match(sensor_code)
        if not m:
            continue
        y = int(m.group("y"))
        x = int(m.group("x"))
        level = int(m.group("level"))
        depth_cm = DEPTH_CM_BY_LEVEL.get(level)
        table = f"{schema_prefix}.datavalues"
        sensor = BasaltCO2Series(
            table_name=table,
            sensor_id=int(sensor_id),
            variable_id=9,
            slope=f"LEO {slope_letter}",
            x_coord_m=x,
            y_coord_m=y,
            depth_cm=depth_cm,
            units="ppm",
        )
        output_name = build_output_name(sensor_code, "basalt")
        save_series(sensor, out_dir / output_name)


def fetch_and_save_air(cur, schema_prefix: str, slope_letter: str, out_dir: Path) -> None:
    pattern = f"LEO-{slope_letter}_%LI-7000"
    cur.execute(
        f"SELECT sensorid, sensorcode FROM {schema_prefix}.sensors WHERE sensorcode LIKE :pattern ORDER BY sensorid",
        pattern=pattern,
    )
    for sensor_id, sensor_code in cur.fetchall():
        table = f"{schema_prefix}.datavalueslicor"
        sensor = AirCO2Series(
            table_name=table,
            sensor_id=int(sensor_id),
            sensor_code=sensor_code,
            variable_id=56,
            slope=f"LEO {slope_letter}",
            x_coord_m=None,
            y_coord_m=None,
            height_m=0.25,
            units="umol/mol",
        )
        output_name = build_output_name(sensor_code, "air")
        save_series(sensor, out_dir / output_name)


def main(slope: str) -> None:
    mapping = {
        "west": ("leo_west", "W"),
        "east": ("leo_east", "E"),
        "center": ("leo_center", "C"),
    }
    if slope not in mapping:
        raise SystemExit(f"Unknown slope: {slope}. Choose one of: {', '.join(mapping)}")

    schema_prefix, slope_letter = mapping[slope]
    out_dir = OUTPUT_DIR / f"LEO_{slope.capitalize()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving LEO {slope.capitalize()} CO2 row data to {out_dir}")
    with SensorDB() as sdb:
        with sdb.cursor() as cur:
            fetch_and_save_basalt(cur, schema_prefix, slope_letter, out_dir)
            fetch_and_save_air(cur, schema_prefix, slope_letter, out_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python save_leo_slope_co2_row_data.py [west|east|center]")
        raise SystemExit(2)
    main(sys.argv[1].lower())


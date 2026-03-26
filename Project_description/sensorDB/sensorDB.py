import os
import re
import sys
from datetime import datetime
from pathlib import Path

from pandas import DataFrame

DEFAULT_ORACLE_CLIENT_LIB_DIR = Path("/opt/oracle/instantclient_19_26")
LOCAL_LIB_DIR = Path.home() / ".local/lib"
ORACLE_ENV_READY_FLAG = "SENSORDB_ORACLE_ENV_READY"


def _ensure_oracle_runtime_env():
    # If Oracle client libraries are installed locally, restart the script with
    # the required shared-library paths so python-oracledb can use Thick mode.
    oracle_client_lib_dir = Path(
        os.getenv("ORACLE_CLIENT_LIB_DIR", str(DEFAULT_ORACLE_CLIENT_LIB_DIR))
    ).expanduser()
    if not oracle_client_lib_dir.exists():
        return

    required_entries = [str(oracle_client_lib_dir)]
    if LOCAL_LIB_DIR.exists():
        required_entries.append(str(LOCAL_LIB_DIR))

    current_entries = [
        entry for entry in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    missing_entries = [entry for entry in required_entries if entry not in current_entries]

    os.environ.setdefault("ORACLE_CLIENT_LIB_DIR", str(oracle_client_lib_dir))
    if not missing_entries or os.getenv(ORACLE_ENV_READY_FLAG) == "1":
        return

    new_env = os.environ.copy()
    new_env["ORACLE_CLIENT_LIB_DIR"] = str(oracle_client_lib_dir)
    new_env["LD_LIBRARY_PATH"] = os.pathsep.join(required_entries + current_entries)
    new_env[ORACLE_ENV_READY_FLAG] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], new_env)


_ensure_oracle_runtime_env()

import oracledb
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path.home() / "Documents" / ".env")
TABLE_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$#]*(\.[A-Z][A-Z0-9_$#]*)?$", re.IGNORECASE)

DEFAULT_START_DATE = "2025-12-20"
DEFAULT_END_DATE = "2026-01-01"


class SensorDB:
    """Minimal helper for reading a sensor series from Oracle into pandas."""

    def __init__(self):
        self._user = os.getenv("ORACLE_USER")
        self._password = os.getenv("ORACLE_PASSWORD")
        self._host = os.getenv("ORACLE_HOST")
        self._port = int(os.getenv("ORACLE_PORT", 1521))
        self._sid = os.getenv("ORACLE_SID")
        self._connection = None
        self._oracle_client_lib_dir = os.getenv("ORACLE_CLIENT_LIB_DIR")
        self._oracle_client_initialized = False

    def _init_oracle_client(self):
        if self._oracle_client_initialized:
            return
        if not self._oracle_client_lib_dir:
            raise RuntimeError(
                "Oracle Thick mode is required for this database connection. "
                "Set ORACLE_CLIENT_LIB_DIR or install Oracle Instant Client."
            )

        # SensorDB requires Oracle Thick mode, so the client library must be initialized.
        oracledb.init_oracle_client(lib_dir=self._oracle_client_lib_dir)
        self._oracle_client_initialized = True

    def connect(self):
        # Connection parameters are loaded from ~/Documents/.env.
        self._init_oracle_client()
        dsn = oracledb.makedsn(self._host, self._port, sid=self._sid)
        self._connection = oracledb.connect(user=self._user, password=self._password, dsn=dsn)
        self._connection.call_timeout = 30_000  # 30 seconds

    def disconnect(self):
        if self._connection:
            self._connection.close()
            self._connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def get_value_series(
        self,
        table_name: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = DEFAULT_END_DATE,
        row_limit: int | None = None,
    ) -> pd.DataFrame:
        """Return two columns: timestamp as text and value_si in SI units."""
        normalized_table_name = table_name.strip().upper()
        if not TABLE_NAME_PATTERN.fullmatch(normalized_table_name):
            raise ValueError(f"Invalid Oracle table name: {table_name}")
        if row_limit is not None and row_limit <= 0:
            raise ValueError(f"row_limit must be positive, got {row_limit}")

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        # query = f"""
        #     -- Return timestamp as text so it behaves like a normal pandas column
        #     -- in this Python/pandas environment.
        #     SELECT TO_CHAR(t.TIMESTAMP, 'YYYY-MM-DD HH24:MI:SS.FF6') AS timestamp,
        #            t.VALUE * 1e-6 AS value_si
        #     FROM {normalized_table_name} t
        #     WHERE t.TIMESTAMP >= :start_date
        #       AND t.TIMESTAMP < :end_date
        #     ORDER BY t.TIMESTAMP
        # """

        query = f"""
                    -- Return timestamp as text so it behaves like a normal pandas column
                    -- in this Python/pandas environment.
                    SELECT
                        TO_CHAR(dv.localdatetime, 'YYYY-MM-DD HH24:MI:SS') AS localdatetime,
                        dv.datavalue
                    FROM
                        {normalized_table_name} dv
                    WHERE
                        dv.sensorid = 994
                        AND dv.variableid = 9
                        AND dv.localdatetime >= :start_date
                        AND dv.localdatetime < :end_date
                    ORDER BY dv.localdatetime
                    {f'FETCH FIRST {row_limit} ROWS ONLY' if row_limit else ''}
                """


        with self._connection.cursor() as cursor:
            cursor.arraysize = 10000  # fetch 10k rows per round-trip
            cursor.execute(query, {
                "start_date": start_dt,
                "end_date": end_dt,
            })
            result_columns = [description[0].lower() for description in cursor.description]
            frame = pd.DataFrame.from_records(cursor.fetchall(), columns=result_columns)

        return frame

if __name__ == "__main__":
    with SensorDB() as sensor_db:
        CO2: DataFrame = sensor_db.get_value_series(
            table_name="leo_center.datavalues",
            start_date=DEFAULT_START_DATE,
            end_date=DEFAULT_END_DATE,
            row_limit=100_000,  # limit to 100k rows for testing
        )

    print(f"{len(CO2)} rows, columns: {', '.join(CO2.columns)}")
    print("First timestamps:", CO2["localdatetime"].head().tolist())
    print("First values:", CO2["datavalue"].head().tolist())

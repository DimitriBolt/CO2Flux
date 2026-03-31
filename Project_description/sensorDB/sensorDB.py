from __future__ import annotations

import pickle
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import oracledb
import pandas as pd
from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path.home() / "Documents" / ".env"
DEFAULT_ORACLE_CLIENT_LIB_DIR = Path("/opt/oracle/instantclient_19_26")
LOCAL_LIB_DIR = Path.home() / ".local/lib"
ORA_COMPAT_DIR = Path("/tmp/ora_compat")
ORA_COMPAT_LINK = ORA_COMPAT_DIR / "libaio.so.1"
ORA_COMPAT_TARGET = Path("/lib/x86_64-linux-gnu/libaio.so.1t64")
DEFAULT_CALL_TIMEOUT_MS = 120_000
DEFAULT_ARRAYSIZE = 10_000
TABLE_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$#]*(\.[A-Z][A-Z0-9_$#]*)?$", re.IGNORECASE)

_ORACLE_CLIENT_INITIALIZED = False


def _ensure_libaio_compat_link() -> Path | None:
    if not ORA_COMPAT_TARGET.exists():
        return None

    ORA_COMPAT_DIR.mkdir(parents=True, exist_ok=True)
    if ORA_COMPAT_LINK.is_symlink() or ORA_COMPAT_LINK.exists():
        if ORA_COMPAT_LINK.resolve() != ORA_COMPAT_TARGET.resolve():
            ORA_COMPAT_LINK.unlink()
            ORA_COMPAT_LINK.symlink_to(ORA_COMPAT_TARGET)
    else:
        ORA_COMPAT_LINK.symlink_to(ORA_COMPAT_TARGET)

    return ORA_COMPAT_DIR


def _runtime_library_entries() -> list[str]:
    oracle_client_lib_dir = Path(
        os.getenv("ORACLE_CLIENT_LIB_DIR", str(DEFAULT_ORACLE_CLIENT_LIB_DIR))
    ).expanduser()
    if not oracle_client_lib_dir.exists():
        return []

    os.environ.setdefault("ORACLE_CLIENT_LIB_DIR", str(oracle_client_lib_dir))

    required_entries = [str(oracle_client_lib_dir)]
    compat_dir = _ensure_libaio_compat_link()
    if compat_dir is not None:
        required_entries.append(str(compat_dir))
    if LOCAL_LIB_DIR.exists():
        required_entries.append(str(LOCAL_LIB_DIR))

    return required_entries


def _prepare_oracle_runtime_env() -> str | None:
    required_entries = _runtime_library_entries()
    if not required_entries:
        return None

    current_entries = [
        entry for entry in os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    missing_entries = [entry for entry in required_entries if entry not in current_entries]
    if missing_entries:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(required_entries + current_entries)

    return os.environ.get("ORACLE_CLIENT_LIB_DIR")


load_dotenv(DEFAULT_ENV_PATH)
_prepare_oracle_runtime_env()


def validate_table_name(table_name: str) -> str:
    normalized = table_name.strip()
    if not TABLE_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(f"Invalid Oracle table name: {table_name}")
    return normalized


def _should_use_worker(exc: Exception) -> bool:
    text = str(exc)
    return "DPI-1047" in text or "Cannot locate a 64-bit Oracle Client library" in text


def _build_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    required_entries = _runtime_library_entries()
    if required_entries:
        current_entries = [
            entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
        ]
        merged_entries = required_entries + [
            entry for entry in current_entries if entry not in required_entries
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(merged_entries)
        env["ORACLE_CLIENT_LIB_DIR"] = required_entries[0]
    return env


@dataclass(frozen=True, slots=True)
class SensorDBConfig:
    user: str
    password: str
    host: str
    port: int
    sid: str
    oracle_client_lib_dir: str

    @classmethod
    def from_env(cls) -> "SensorDBConfig":
        user = os.getenv("ORACLE_USER")
        password = os.getenv("ORACLE_PASSWORD")
        host = os.getenv("ORACLE_HOST")
        sid = os.getenv("ORACLE_SID")
        port_raw = os.getenv("ORACLE_PORT", "1521")
        oracle_client_lib_dir = os.getenv("ORACLE_CLIENT_LIB_DIR")

        missing = [
            name
            for name, value in (
                ("ORACLE_USER", user),
                ("ORACLE_PASSWORD", password),
                ("ORACLE_HOST", host),
                ("ORACLE_SID", sid),
                ("ORACLE_CLIENT_LIB_DIR", oracle_client_lib_dir),
            )
            if not value
        ]
        if missing:
            missing_text = ", ".join(missing)
            raise RuntimeError(
                f"Missing Oracle configuration in {DEFAULT_ENV_PATH}: {missing_text}."
            )

        return cls(
            user=user,
            password=password,
            host=host,
            port=int(port_raw),
            sid=sid,
            oracle_client_lib_dir=oracle_client_lib_dir,
        )


class SensorDB:
    """Infrastructure helper for Oracle SensorDB connections and SELECT queries."""

    def __init__(
        self,
        config: SensorDBConfig | None = None,
        *,
        call_timeout_ms: int = DEFAULT_CALL_TIMEOUT_MS,
        arraysize: int = DEFAULT_ARRAYSIZE,
    ) -> None:
        self.config = config or SensorDBConfig.from_env()
        self.call_timeout_ms = call_timeout_ms
        self.arraysize = arraysize
        self._connection: oracledb.Connection | None = None

    def _init_oracle_client(self) -> None:
        global _ORACLE_CLIENT_INITIALIZED

        if _ORACLE_CLIENT_INITIALIZED:
            return

        try:
            oracledb.init_oracle_client(lib_dir=self.config.oracle_client_lib_dir)
        except oracledb.ProgrammingError as exc:
            if "already initialized" not in str(exc).lower():
                raise
        _ORACLE_CLIENT_INITIALIZED = True

    def connect(self) -> None:
        if self._connection is not None:
            return

        self._init_oracle_client()
        dsn = oracledb.makedsn(self.config.host, self.config.port, sid=self.config.sid)
        self._connection = oracledb.connect(
            user=self.config.user,
            password=self.config.password,
            dsn=dsn,
        )
        self._connection.call_timeout = self.call_timeout_ms

    def disconnect(self) -> None:
        if self._connection is None:
            return

        self._connection.close()
        self._connection = None

    def __enter__(self) -> "SensorDB":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.disconnect()

    @contextmanager
    def cursor(self) -> Iterator[oracledb.Cursor]:
        self.connect()
        assert self._connection is not None
        cursor = self._connection.cursor()
        cursor.arraysize = self.arraysize
        try:
            yield cursor
        finally:
            cursor.close()

    def fetch_dataframe(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        try:
            with self.cursor() as cursor:
                cursor.execute(query, params or {})
                columns = [description[0].lower() for description in cursor.description or ()]
                return pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        except oracledb.DatabaseError as exc:
            if not _should_use_worker(exc):
                raise
            return self._fetch_dataframe_via_worker(query, params)

    def fetch_one(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        frame = self.fetch_dataframe(query, params)
        if frame.empty:
            return None
        return frame.iloc[0].to_dict()

    def _fetch_dataframe_via_worker(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        payload = {
            "query": query,
            "params": dict(params or {}),
            "call_timeout_ms": self.call_timeout_ms,
            "arraysize": self.arraysize,
        }
        completed = subprocess.run(
            [sys.executable, __file__, "--worker"],
            input=pickle.dumps(payload),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_build_worker_env(),
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Oracle worker failed: {stderr}") from None

        return pickle.loads(completed.stdout)


def _worker_main() -> int:
    request = pickle.loads(sys.stdin.buffer.read())
    config = SensorDBConfig.from_env()

    try:
        oracledb.init_oracle_client(lib_dir=config.oracle_client_lib_dir)
    except oracledb.ProgrammingError as exc:
        if "already initialized" not in str(exc).lower():
            raise

    dsn = oracledb.makedsn(config.host, config.port, sid=config.sid)
    connection = oracledb.connect(
        user=config.user,
        password=config.password,
        dsn=dsn,
    )
    connection.call_timeout = int(request["call_timeout_ms"])
    cursor = connection.cursor()
    cursor.arraysize = int(request["arraysize"])
    try:
        cursor.execute(request["query"], request["params"])
        columns = [description[0].lower() for description in cursor.description or ()]
        frame = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
        sys.stdout.buffer.write(pickle.dumps(frame))
        return 0
    finally:
        cursor.close()
        connection.close()


__all__ = [
    "DEFAULT_CALL_TIMEOUT_MS",
    "DEFAULT_ENV_PATH",
    "DEFAULT_ORACLE_CLIENT_LIB_DIR",
    "SensorDB",
    "SensorDBConfig",
    "validate_table_name",
]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_worker_main())

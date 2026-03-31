from __future__ import annotations

import pickle
import os
import re
import subprocess
import sys
import warnings
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
WINDOWS_ORACLE_BASE_DIRS = (
    Path(r"C:\oracle"),
    Path(r"C:\oracle\instantclient"),
    Path(r"C:\instantclient"),
    Path(r"C:\Program Files\Oracle"),
)
LOCAL_LIB_DIR = Path.home() / ".local/lib"
ORA_COMPAT_DIR = Path("/tmp/ora_compat")
ORA_COMPAT_LINK = ORA_COMPAT_DIR / "libaio.so.1"
ORA_COMPAT_TARGET = Path("/lib/x86_64-linux-gnu/libaio.so.1t64")
DEFAULT_CALL_TIMEOUT_MS = 120_000
DEFAULT_ARRAYSIZE = 10_000
TABLE_NAME_PATTERN = re.compile(r"^[A-Z][A-Z0-9_$#]*(\.[A-Z][A-Z0-9_$#]*)?$", re.IGNORECASE)

_ORACLE_CLIENT_INITIALIZED = False


def _oracle_client_library_name() -> str:
    if os.name == "nt":
        return "oci.dll"
    if sys.platform == "darwin":
        return "libclntsh.dylib"
    return "libclntsh.so"


def _normalize_existing_dir(raw_path: str | os.PathLike[str] | None) -> Path | None:
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if path.is_file():
        path = path.parent
    if not path.exists():
        return None
    return path


def _directory_has_oracle_client(path: Path) -> bool:
    candidate = path if path.is_dir() else path.parent
    return (candidate / _oracle_client_library_name()).exists()


def _discover_oracle_client_lib_dir() -> Path | None:
    candidates: list[Path] = []

    configured = _normalize_existing_dir(os.getenv("ORACLE_CLIENT_LIB_DIR"))
    if configured is not None:
        candidates.append(configured)

    if DEFAULT_ORACLE_CLIENT_LIB_DIR.exists():
        candidates.append(DEFAULT_ORACLE_CLIENT_LIB_DIR)

    if os.name == "nt":
        for entry in os.getenv("PATH", "").split(os.pathsep):
            candidate = _normalize_existing_dir(entry)
            if candidate is not None:
                candidates.append(candidate)

        for base in (*WINDOWS_ORACLE_BASE_DIRS, Path.home() / "oracle"):
            if not base.exists():
                continue
            candidates.append(base)
            try:
                for dll_path in base.rglob("oci.dll"):
                    if dll_path.is_file():
                        candidates.append(dll_path.parent)
            except OSError:
                continue

    seen: set[str] = set()
    for candidate in candidates:
        candidate_text = str(candidate)
        if candidate_text in seen:
            continue
        seen.add(candidate_text)
        if _directory_has_oracle_client(candidate):
            os.environ.setdefault("ORACLE_CLIENT_LIB_DIR", candidate_text)
            return candidate

    return None


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
    oracle_client_lib_dir = _discover_oracle_client_lib_dir()
    if oracle_client_lib_dir is None:
        return []

    os.environ.setdefault("ORACLE_CLIENT_LIB_DIR", str(oracle_client_lib_dir))

    required_entries = [str(oracle_client_lib_dir)]
    if os.name == "nt":
        return required_entries

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


def _is_call_timeout_error(exc: Exception) -> bool:
    text = str(exc)
    return (
        "DPY-4024" in text
        or "DPI-1067" in text
        or "ORA-03156" in text
        or "call timeout" in text.lower()
    )


def _build_worker_env() -> dict[str, str]:
    env = os.environ.copy()
    required_entries = _runtime_library_entries()
    if required_entries:
        env["ORACLE_CLIENT_LIB_DIR"] = required_entries[0]
        if os.name == "nt":
            current_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
            merged_entries = required_entries + [
                entry for entry in current_entries if entry not in required_entries
            ]
            env["PATH"] = os.pathsep.join(merged_entries)
            return env

        current_entries = [
            entry for entry in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
        ]
        merged_entries = required_entries + [
            entry for entry in current_entries if entry not in required_entries
        ]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(merged_entries)
    return env


@dataclass(frozen=True, slots=True)
class SensorDBConfig:
    user: str
    password: str
    host: str
    port: int
    sid: str
    oracle_client_lib_dir: str | None

    @classmethod
    def from_env(cls) -> "SensorDBConfig":
        user = os.getenv("ORACLE_USER")
        password = os.getenv("ORACLE_PASSWORD")
        host = os.getenv("ORACLE_HOST")
        sid = os.getenv("ORACLE_SID")
        port_raw = os.getenv("ORACLE_PORT", "1521")
        discovered_client_dir = _discover_oracle_client_lib_dir()
        oracle_client_lib_dir = (
            str(discovered_client_dir)
            if discovered_client_dir is not None
            else None
        )

        missing = [
            name
            for name, value in (
                ("ORACLE_USER", user),
                ("ORACLE_PASSWORD", password),
                ("ORACLE_HOST", host),
                ("ORACLE_SID", sid),
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
            init_kwargs: dict[str, Any] = {}
            if self.config.oracle_client_lib_dir:
                init_kwargs["lib_dir"] = self.config.oracle_client_lib_dir
            oracledb.init_oracle_client(**init_kwargs)
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
            if _should_use_worker(exc):
                return self._fetch_dataframe_via_worker(query, params)
            if self.call_timeout_ms > 0 and _is_call_timeout_error(exc):
                warnings.warn(
                    (
                        f"Oracle call exceeded configured timeout of {self.call_timeout_ms} ms; "
                        "retrying without a call timeout."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.disconnect()
                return self._fetch_dataframe_via_worker(query, params, call_timeout_ms=0)
            raise

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
        *,
        call_timeout_ms: int | None = None,
    ) -> pd.DataFrame:
        payload = {
            "query": query,
            "params": dict(params or {}),
            "call_timeout_ms": (
                self.call_timeout_ms if call_timeout_ms is None else int(call_timeout_ms)
            ),
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
        init_kwargs: dict[str, Any] = {}
        if config.oracle_client_lib_dir:
            init_kwargs["lib_dir"] = config.oracle_client_lib_dir
        oracledb.init_oracle_client(**init_kwargs)
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

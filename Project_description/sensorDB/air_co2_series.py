from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import warnings

import pandas as pd

from sensorDB import SensorDB, validate_table_name


DEFAULT_AIR_VARIABLE_ID = 56
DEFAULT_AIR_HEIGHT_M = 0.25
DEFAULT_START_DATETIME = "2010-01-01"
NO_DATA_VALUE = -9999.0
VALUE_COLUMN = "datavalue"
TIME_COLUMN = "localdatetime"


def _coerce_datetime(value: str | datetime | pd.Timestamp | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, datetime):
        return value
    return pd.Timestamp(value).to_pydatetime()


@dataclass(slots=True)
class AirCO2Series:
    table_name: str
    sensor_id: int
    variable_id: int = DEFAULT_AIR_VARIABLE_ID
    slope: str | None = None
    x_coord_m: float | None = None
    y_coord_m: float | None = None
    height_m: float = DEFAULT_AIR_HEIGHT_M
    units: str = "ppm"
    sensor_code: str | None = None
    _validated_table_name: str = field(init=False, repr=False)
    _last_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _last_series: pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validated_table_name = validate_table_name(self.table_name)
        if self.variable_id != DEFAULT_AIR_VARIABLE_ID:
            raise ValueError(
                f"AirCO2Series expects variable_id={DEFAULT_AIR_VARIABLE_ID}, "
                f"got {self.variable_id}."
            )

    def _base_params(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        include_no_data: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sensor_id": self.sensor_id,
            "variable_id": self.variable_id,
            "start_datetime": _coerce_datetime(start_datetime),
        }
        end_dt = _coerce_datetime(end_datetime)
        if end_dt is not None:
            params["end_datetime"] = end_dt
        if not include_no_data:
            params["no_data_value"] = NO_DATA_VALUE
        return params

    def _build_fetch_query(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        row_limit: int | None = None,
        include_no_data: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        if row_limit is not None and row_limit <= 0:
            raise ValueError(f"row_limit must be positive, got {row_limit}")

        params = self._base_params(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            include_no_data=include_no_data,
        )
        end_clause = ""
        if "end_datetime" in params:
            end_clause = "AND dv.localdatetime <= :end_datetime"

        no_data_clause = ""
        if not include_no_data:
            no_data_clause = "AND dv.datavalue <> :no_data_value"

        row_limit_clause = f"\nFETCH FIRST {row_limit} ROWS ONLY" if row_limit else ""
        query = f"""
            SELECT
                dv.localdatetime,
                dv.datavalue
            FROM
                {self._validated_table_name} dv
            WHERE
                dv.sensorid = :sensor_id
                AND dv.variableid = :variable_id
                AND dv.localdatetime >= :start_datetime
                {end_clause}
                {no_data_clause}
            ORDER BY
                dv.localdatetime{row_limit_clause}
        """
        return query, params

    def fetch_dataframe(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        row_limit: int | None = None,
        include_no_data: bool = False,
    ) -> pd.DataFrame:
        query, params = self._build_fetch_query(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            row_limit=row_limit,
            include_no_data=include_no_data,
        )
        with SensorDB() as sensor_db:
            frame = sensor_db.fetch_dataframe(query, params)

        if TIME_COLUMN in frame.columns:
            frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN])
        if VALUE_COLUMN in frame.columns:
            frame[VALUE_COLUMN] = pd.to_numeric(frame[VALUE_COLUMN], errors="coerce")

        self._last_frame = frame
        return frame

    def fetch_series(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        row_limit: int | None = None,
        include_no_data: bool = False,
    ) -> pd.Series:
        frame = self.fetch_dataframe(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            row_limit=row_limit,
            include_no_data=include_no_data,
        )
        if TIME_COLUMN not in frame.columns or VALUE_COLUMN not in frame.columns:
            raise ValueError(
                f"Expected columns {TIME_COLUMN!r} and {VALUE_COLUMN!r}, got {list(frame.columns)!r}."
            )

        series = pd.Series(
            data=frame[VALUE_COLUMN].to_numpy(copy=False),
            index=pd.DatetimeIndex(frame[TIME_COLUMN], name=TIME_COLUMN),
            name=VALUE_COLUMN,
        )
        self._last_series = series
        return series

    def fetch(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        row_limit: int | None = None,
        include_no_data: bool = False,
        prefer_series: bool = True,
    ) -> pd.Series | pd.DataFrame:
        if not prefer_series:
            return self.fetch_dataframe(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
                include_no_data=include_no_data,
            )

        try:
            return self.fetch_series(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
                include_no_data=include_no_data,
            )
        except (TypeError, ValueError) as exc:
            warnings.warn(
                f"Falling back to DataFrame because Series conversion failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.fetch_dataframe(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
                include_no_data=include_no_data,
            )

    def describe_time_coverage(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        include_row_count: bool = False,
        include_no_data: bool = False,
    ) -> dict[str, Any]:
        params = self._base_params(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            include_no_data=include_no_data,
        )
        end_clause = ""
        if "end_datetime" in params:
            end_clause = "AND dv.localdatetime <= :end_datetime"

        no_data_clause = ""
        if not include_no_data:
            no_data_clause = "AND dv.datavalue <> :no_data_value"

        first_query = f"""
            SELECT
                dv.localdatetime AS first_timestamp
            FROM
                {self._validated_table_name} dv
            WHERE
                dv.sensorid = :sensor_id
                AND dv.variableid = :variable_id
                AND dv.localdatetime >= :start_datetime
                {end_clause}
                {no_data_clause}
            ORDER BY
                dv.localdatetime
            FETCH FIRST 1 ROW ONLY
        """
        last_query = f"""
            SELECT
                dv.localdatetime AS last_timestamp
            FROM
                {self._validated_table_name} dv
            WHERE
                dv.sensorid = :sensor_id
                AND dv.variableid = :variable_id
                AND dv.localdatetime >= :start_datetime
                {end_clause}
                {no_data_clause}
            ORDER BY
                dv.localdatetime DESC
            FETCH FIRST 1 ROW ONLY
        """

        with SensorDB() as sensor_db:
            coverage: dict[str, Any] = {}
            first_row = sensor_db.fetch_one(first_query, params) or {}
            last_row = sensor_db.fetch_one(last_query, params) or {}
            coverage.update(first_row)
            coverage.update(last_row)
            if include_row_count:
                count_query = f"""
                    SELECT
                        COUNT(*) AS row_count
                    FROM
                        {self._validated_table_name} dv
                    WHERE
                        dv.sensorid = :sensor_id
                        AND dv.variableid = :variable_id
                        AND dv.localdatetime >= :start_datetime
                        {end_clause}
                        {no_data_clause}
                """
                count_row = sensor_db.fetch_one(count_query, params) or {}
                coverage.update(count_row)

        if "first_timestamp" in coverage and coverage["first_timestamp"] is not None:
            coverage["first_timestamp"] = pd.Timestamp(coverage["first_timestamp"])
        if "last_timestamp" in coverage and coverage["last_timestamp"] is not None:
            coverage["last_timestamp"] = pd.Timestamp(coverage["last_timestamp"])

        coverage.update(
            {
                "table_name": self._validated_table_name,
                "sensor_id": self.sensor_id,
                "sensor_code": self.sensor_code,
                "variable_id": self.variable_id,
                "slope": self.slope,
                "x_coord_m": self.x_coord_m,
                "y_coord_m": self.y_coord_m,
                "height_m": self.height_m,
                "units": self.units,
            }
        )
        return coverage

    @property
    def label(self) -> str:
        parts = [
            self.slope or "Air CO2 sensor",
            f"sensorid={self.sensor_id}",
            f"variableid={self.variable_id}",
        ]
        if self.sensor_code:
            parts.append(self.sensor_code)
        if self.x_coord_m is not None and self.y_coord_m is not None:
            parts.append(f"(x={self.x_coord_m}, y={self.y_coord_m})")
        parts.append(f"height={self.height_m} m")
        return ", ".join(parts)


__all__ = [
    "AirCO2Series",
    "DEFAULT_AIR_HEIGHT_M",
    "DEFAULT_AIR_VARIABLE_ID",
    "DEFAULT_START_DATETIME",
    "NO_DATA_VALUE",
]

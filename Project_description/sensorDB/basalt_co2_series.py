from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from models import SensorSpec
from sensorDB import SensorDB, validate_table_name


DEFAULT_BASALT_VARIABLE_ID = 9
DEFAULT_START_DATETIME = "2010-01-01"
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
class BasaltCO2Series:
    table_name: str
    sensor_id: int
    variable_id: int = DEFAULT_BASALT_VARIABLE_ID
    slope: str | None = None
    x_coord_m: float | None = None
    y_coord_m: float | None = None
    depth_cm: float | None = None
    units: str = "ppm"
    _validated_table_name: str = field(init=False, repr=False)
    _last_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _last_series: pd.Series | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validated_table_name = validate_table_name(self.table_name)
        if self.variable_id != DEFAULT_BASALT_VARIABLE_ID:
            raise ValueError(
                f"BasaltCO2Series expects variable_id={DEFAULT_BASALT_VARIABLE_ID}, "
                f"got {self.variable_id}."
            )

    @classmethod
    def from_sensor_spec(cls, spec: SensorSpec) -> "BasaltCO2Series":
        return cls(
            table_name=spec.table_name,
            sensor_id=spec.sensor_id,
            variable_id=spec.variable_id,
            slope=spec.slope,
            x_coord_m=spec.x_coord_m,
            y_coord_m=spec.y_coord_m,
            depth_cm=spec.depth_cm,
            units=spec.units or "ppm",
        )

    def _base_params(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "sensor_id": self.sensor_id,
            "variable_id": self.variable_id,
            "start_datetime": _coerce_datetime(start_datetime),
        }
        end_dt = _coerce_datetime(end_datetime)
        if end_dt is not None:
            params["end_datetime"] = end_dt
        return params

    def _build_fetch_query(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        row_limit: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if row_limit is not None and row_limit <= 0:
            raise ValueError(f"row_limit must be positive, got {row_limit}")

        params = self._base_params(start_datetime, end_datetime)
        end_clause = ""
        if "end_datetime" in params:
            end_clause = "AND dv.localdatetime <= :end_datetime"

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
    ) -> pd.DataFrame:
        query, params = self._build_fetch_query(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            row_limit=row_limit,
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
    ) -> pd.Series:
        frame = self.fetch_dataframe(
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            row_limit=row_limit,
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
        prefer_series: bool = True,
    ) -> pd.Series | pd.DataFrame:
        if not prefer_series:
            return self.fetch_dataframe(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
            )

        try:
            return self.fetch_series(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
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
            )

    def describe_time_coverage(
        self,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        *,
        include_row_count: bool = False,
    ) -> dict[str, Any]:
        params = self._base_params(start_datetime, end_datetime)
        end_clause = ""
        if "end_datetime" in params:
            end_clause = "AND dv.localdatetime <= :end_datetime"

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
                "variable_id": self.variable_id,
                "slope": self.slope,
                "x_coord_m": self.x_coord_m,
                "y_coord_m": self.y_coord_m,
                "depth_cm": self.depth_cm,
                "units": self.units,
            }
        )
        return coverage

    def plot(
        self,
        series: pd.Series | None = None,
        *,
        start_datetime: str | datetime | pd.Timestamp = DEFAULT_START_DATETIME,
        end_datetime: str | datetime | pd.Timestamp | None = None,
        row_limit: int | None = None,
        ax: plt.Axes | None = None,
        color: str = "#2E5EAA",
        linewidth: float = 0.9,
        title: str | None = None,
    ) -> tuple[plt.Figure, plt.Axes]:
        values = series
        if values is None:
            values = self.fetch_series(
                start_datetime=start_datetime,
                end_datetime=end_datetime,
                row_limit=row_limit,
            )

        if ax is None:
            figure, ax = plt.subplots(figsize=(12, 4))
        else:
            figure = ax.figure

        ax.plot(values.index, values.to_numpy(copy=False), color=color, linewidth=linewidth)
        ax.set_xlabel("Local datetime")
        ax.set_ylabel(f"CO2 ({self.units})")
        ax.set_title(title or self.label)
        ax.grid(True, alpha=0.25)
        figure.tight_layout()
        return figure, ax

    @property
    def label(self) -> str:
        parts = [
            self.slope or "Basalt CO2 sensor",
            f"sensorid={self.sensor_id}",
            f"variableid={self.variable_id}",
        ]
        if self.x_coord_m is not None and self.y_coord_m is not None:
            parts.append(f"(x={self.x_coord_m}, y={self.y_coord_m})")
        if self.depth_cm is not None:
            parts.append(f"depth={self.depth_cm} cm")
        return ", ".join(parts)


__all__ = [
    "BasaltCO2Series",
    "DEFAULT_BASALT_VARIABLE_ID",
    "DEFAULT_START_DATETIME",
]

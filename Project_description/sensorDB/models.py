from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True, slots=True)
class SensorSpec:
    table_name: str
    sensor_id: int
    variable_id: int = 9
    slope: str | None = None
    x_coord_m: float | None = None
    y_coord_m: float | None = None
    depth_cm: float | None = None
    units: str | None = "ppm"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShapiroResult:
    statistic: float
    p_value: float
    sample_size: int
    alpha: float = 0.05
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reject_normality(self) -> bool:
        return self.p_value < self.alpha

    @property
    def looks_normal(self) -> bool:
        return not self.reject_normality

    def to_dict(self) -> dict[str, object]:
        return {
            "statistic": self.statistic,
            "p_value": self.p_value,
            "sample_size": self.sample_size,
            "alpha": self.alpha,
            "reject_normality": self.reject_normality,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class NormalityReport:
    sample_size: int
    excluded_non_finite: int
    mean: float
    standard_deviation: float
    shapiro: ShapiroResult
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_size": self.sample_size,
            "excluded_non_finite": self.excluded_non_finite,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
            "shapiro": self.shapiro.to_dict(),
            "notes": list(self.notes),
        }

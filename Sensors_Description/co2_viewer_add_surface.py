from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import co2_vertical_profile_viewer as base
import pandas as pd


SENSOR_DB_DIR = Path(__file__).resolve().parent.parent / "Project_description" / "sensorDB"
if str(SENSOR_DB_DIR) not in sys.path:
    sys.path.insert(0, str(SENSOR_DB_DIR))


AIR_LEVEL_M = 0.25
AIR_BAR_COLOR = "#D98324"
PLOT_Y_MAX_M = 0.30
DISPLAY_LEVELS_M = [AIR_LEVEL_M, *base.DISPLAY_DEPTHS_M]
BAR_LEVELS_M = [AIR_LEVEL_M, *base.DISPLAY_DEPTHS_M[1:]]


@dataclass(frozen=True)
class SurfaceAirSelectionConfig:
    x_coord_m: float
    y_coord_m: float


def load_surface_air_config(config_path: Path) -> SurfaceAirSelectionConfig:
    raw_config = base._load_toml_config(config_path)
    if base.LOCAL_CONFIG_PATH.exists():
        raw_config = base._merge_config(raw_config, base._load_toml_config(base.LOCAL_CONFIG_PATH))

    section = base._require_section(raw_config, "surface_air")
    return SurfaceAirSelectionConfig(
        x_coord_m=base._require_number(section, "x_coord_m"),
        y_coord_m=base._require_number(section, "y_coord_m"),
    )


# These will be initialized in main()
SURFACE_AIR_CONFIG: SurfaceAirSelectionConfig = None
AIR_X_COORD_M: float = None
AIR_Y_COORD_M: float = None


def load_surface_air_sensor() -> "AirCO2Series":
    from air_co2_catalog import AirCO2Catalog

    catalog = AirCO2Catalog(workbook_path=base.WORKBOOK_PATH)
    return catalog.get_sensor(
        slope=base.SLOPE,
        x_coord_m=AIR_X_COORD_M,
        y_coord_m=AIR_Y_COORD_M,
        height_m=AIR_LEVEL_M,
    )


def build_frames_with_surface_air(
    basalt_frames: list[tuple[base.datetime, dict[float, float]]],
    air_series: pd.Series | list[base.Measurement],
) -> list[tuple[base.datetime, dict[float, float]]]:
    if isinstance(air_series, pd.Series):
        normalized_air_series = [
            (
                timestamp.to_pydatetime() if isinstance(timestamp, pd.Timestamp) else timestamp,
                float(value),
            )
            for timestamp, value in air_series.sort_index().items()
        ]
    else:
        normalized_air_series = [
            (measurement.timestamp, measurement.value) for measurement in air_series
        ]

    air_index = 0
    last_air_value: float | None = None
    frames: list[tuple[base.datetime, dict[float, float]]] = []

    for frame_time, basalt_values in basalt_frames:
        while (
            air_index < len(normalized_air_series)
            and normalized_air_series[air_index][0] <= frame_time
        ):
            last_air_value = normalized_air_series[air_index][1]
            air_index += 1

        combined_values = basalt_values.copy()
        if last_air_value is not None:
            combined_values[AIR_LEVEL_M] = last_air_value
        frames.append((frame_time, combined_values))

    return frames


def format_depth_tick(depth_m: float) -> str:
    if depth_m > 0:
        return f"+{depth_m:.2f}"
    if depth_m == 0:
        return "0.00"
    return f"{depth_m:.2f}"


def draw_frame(
    figure: base.plt.Figure,
    axis: base.plt.Axes,
    frame_time: base.datetime,
    current_values: dict[float, float],
) -> None:
    axis.clear()

    axis.set_xlim(base.CO2_AXIS_MIN_PPM, base.CO2_AXIS_MAX_PPM)
    axis.set_ylim(-0.55, PLOT_Y_MAX_M)
    axis.set_xlabel("CO2 concentration [ppm]")
    axis.set_ylabel("Depth / height [m]")
    axis.set_yticks(DISPLAY_LEVELS_M)
    axis.set_yticklabels([format_depth_tick(level) for level in DISPLAY_LEVELS_M])
    axis.grid(axis="x", color=base.GRID_COLOR, linewidth=0.8, alpha=0.7)
    axis.set_axisbelow(True)
    axis.axhline(0.0, color=base.SURFACE_LINE_COLOR, linewidth=1.3)

    for level_m in BAR_LEVELS_M:
        if level_m not in current_values:
            continue

        value = current_values[level_m]
        visible_width = max(value, 0.0)
        color = AIR_BAR_COLOR if level_m > 0 else base.BAR_COLOR
        axis.barh(
            y=level_m,
            width=visible_width,
            height=base.BAR_HEIGHT_M,
            left=0.0,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )

        if value >= base.CO2_AXIS_MAX_PPM * 0.92:
            label_x = base.CO2_AXIS_MAX_PPM * 0.98
            ha = "right"
        else:
            label_x = max(visible_width + 90, base.CO2_AXIS_MIN_PPM + 50)
            ha = "left"
        axis.text(
            label_x,
            level_m,
            f"{value:.0f} ppm",
            va="center",
            ha=ha,
            fontsize=10,
            clip_on=False,
        )

    values_text: list[str] = []
    if AIR_LEVEL_M in current_values:
        values_text.append(
            f"air +0.25 m @ X={base.format_coordinate(AIR_X_COORD_M)}, "
            f"Y={base.format_coordinate(AIR_Y_COORD_M)} = {current_values[AIR_LEVEL_M]:.0f} ppm"
        )
    for depth_m in base.DISPLAY_DEPTHS_M[1:]:
        if depth_m in current_values:
            values_text.append(f"{depth_m:.2f} m = {current_values[depth_m]:.0f} ppm")
    values_block = "\n".join(values_text) if values_text else "No values available in this frame"

    axis.set_title(
        f"{base.SLOPE} | Basalt X={base.format_coordinate(base.X_COORD_M)} m | "
        f"Basalt Y={base.format_coordinate(base.Y_COORD_M)} m | "
        f"Air X={base.format_coordinate(AIR_X_COORD_M)} m | "
        f"Air Y={base.format_coordinate(AIR_Y_COORD_M)} m | "
        f"{base.format_datetime_for_display(frame_time)}",
        fontsize=12,
        pad=12,
    )
    axis.text(
        0.98,
        0.97,
        values_block,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "#888888", "alpha": 0.92, "boxstyle": "round,pad=0.35"},
    )

    figure.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CO2 vertical profile viewer with atmospheric surface point"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base.DEFAULT_CONFIG_PATH,
        help=f"Path to TOML configuration file (default: {base.DEFAULT_CONFIG_PATH})",
    )
    return parser.parse_args()


def main() -> None:
    global SURFACE_AIR_CONFIG, AIR_X_COORD_M, AIR_Y_COORD_M

    # Parse command line arguments
    args = parse_args()

    # Initialize base module configuration
    base.VIEWER_CONFIG = base.load_viewer_config(args.config)
    base.DEFAULT_ORACLE_CLIENT_LIB_DIR = base.VIEWER_CONFIG.oracle.default_client_lib_dir
    base.LOCAL_LIB_DIR = base.VIEWER_CONFIG.oracle.local_lib_dir
    base.ORACLE_ENV_READY_FLAG = base.VIEWER_CONFIG.oracle.env_ready_flag
    base.SLOPE = base.VIEWER_CONFIG.profile.slope
    base.X_COORD_M = base.VIEWER_CONFIG.profile.x_coord_m
    base.Y_COORD_M = base.VIEWER_CONFIG.profile.y_coord_m
    base.START_DATE = base.VIEWER_CONFIG.profile.start_date
    base.END_DATE = base.VIEWER_CONFIG.profile.end_date

    # Load surface air configuration
    SURFACE_AIR_CONFIG = load_surface_air_config(args.config)
    AIR_X_COORD_M = SURFACE_AIR_CONFIG.x_coord_m
    AIR_Y_COORD_M = SURFACE_AIR_CONFIG.y_coord_m

    # Ensure Oracle runtime environment is set up
    base._ensure_oracle_runtime_env()

    print(f"Using config file: {args.config}")
    print(f"Basalt profile: {base.SLOPE}, X={base.X_COORD_M}, Y={base.Y_COORD_M}")
    print(f"Surface air point: X={AIR_X_COORD_M}, Y={AIR_Y_COORD_M}")
    print(f"Time range: {base.START_DATE} to {base.END_DATE}")
    print()

    start_dt = base.parse_user_datetime(base.START_DATE, is_end=False)
    if start_dt is None:
        raise ValueError("START_DATE must be provided.")
    requested_end_dt = base.parse_user_datetime(base.END_DATE, is_end=True)
    if requested_end_dt is not None and requested_end_dt < start_dt:
        raise ValueError("END_DATE must be greater than or equal to START_DATE.")

    profile_sensors = base.load_profile_sensors()
    air_sensor = load_surface_air_sensor()

    connection = base.connect_to_oracle()
    cursor = connection.cursor()
    try:
        basalt_series_by_depth = {
            sensor.depth_m: base.fetch_measurements(cursor, sensor, start_dt, requested_end_dt)
            for sensor in profile_sensors
        }
    finally:
        cursor.close()
        connection.close()

    air_series = air_sensor.fetch_series(
        start_datetime=start_dt,
        end_datetime=requested_end_dt,
    )

    resolved_end_dt = base.resolve_end_datetime(requested_end_dt, basalt_series_by_depth)
    basalt_frames = base.build_frames(basalt_series_by_depth, resolved_end_dt)
    frames = build_frames_with_surface_air(basalt_frames, air_series)

    if air_series.empty:
        print(
            "Warning: no air observations were found in the selected interval for "
            f"{air_sensor.sensor_code or air_sensor.sensor_id}. The animation will contain basalt only."
        )

    output_stem = base.build_output_stem(start_dt, resolved_end_dt)
    output_gif = base.OUTPUT_DIR / f"{output_stem}.gif"
    output_jpg = base.OUTPUT_DIR / f"{output_stem}.jpg"

    base.plt.rcParams.update(
        {
            "figure.figsize": (10.5, 6.5),
            "figure.dpi": 120,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )

    figure, axis = base.plt.subplots()

    def update(frame_index: int) -> list[object]:
        frame_time, current_values = frames[frame_index]
        draw_frame(figure, axis, frame_time, current_values)
        return []

    animation = base.FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=base.FRAME_DURATION_MS,
        repeat=False,
        blit=False,
    )

    fps = max(1, round(1000 / base.FRAME_DURATION_MS))
    writer = base.SinglePlayPillowWriter(fps=fps)
    print(f"Saving GIF to {output_gif}")
    animation.save(
        output_gif,
        writer=writer,
        dpi=160,
        progress_callback=lambda current_frame, total_frames: (
            print(f"Frame {current_frame + 1}/{total_frames}", end="\r")
            if (current_frame + 1) % 10 == 0 or current_frame + 1 == total_frames
            else None
        ),
    )
    print()

    update(len(frames) - 1)
    print(f"Saving final JPEG to {output_jpg}")
    figure.savefig(output_jpg, format="jpeg", dpi=200)
    base.plt.close(figure)

    print(f"Done. Frames: {len(frames)}")
    print(f"Air sensor: {air_sensor.sensor_code}")
    print(f"GIF: {output_gif}")
    print(f"JPEG: {output_jpg}")


if __name__ == "__main__":
    main()

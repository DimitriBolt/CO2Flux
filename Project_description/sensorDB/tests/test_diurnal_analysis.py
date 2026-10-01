"""Проверки смысла фаз: известные задержки, полночь, разные часы и пропуски."""

import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diurnal_analysis import analyze_diurnal, clock_text, fit_diurnal, observation_intervals, wrap_hours
from plot_column_timeseries import analyze_column, trend_window_mask


def signal(times, peak, amplitude=20, offset=0):
    hours = (times - times.normalize()).total_seconds().to_numpy() / 3600
    return offset + amplitude * np.cos(2 * np.pi * (hours - peak) / 24)


class DiurnalTests(unittest.TestCase):
    def test_full_trend_window_requires_twelve_hours_on_each_side(self):
        times = pd.date_range("2017-11-01", "2017-11-04", freq="5min")
        frame = pd.DataFrame({"localdatetime": times, "datavalue": 400.})
        mask = trend_window_mask(frame)
        expected = (times >= "2017-11-01 12:00") & (times <= "2017-11-03 12:00")
        np.testing.assert_array_equal(mask, expected)

    def test_masked_stuck_values_block_surrounding_trend_windows(self):
        times = pd.date_range("2017-11-01", "2017-11-05", freq="5min")
        frame = pd.DataFrame({"localdatetime": times, "datavalue": 400.})
        frame.loc[frame.localdatetime.between("2017-11-03 06:00", "2017-11-03 08:00"), "datavalue"] = np.nan
        before = frame.copy(deep=True)
        mask = trend_window_mask(frame)
        expected = (((times >= "2017-11-01 12:00") & (times <= "2017-11-02 17:55"))
                    | ((times >= "2017-11-03 20:05") & (times <= "2017-11-04 12:00")))
        np.testing.assert_array_equal(mask, expected)
        pd.testing.assert_frame_equal(frame, before)

    def test_repeated_timestamps_do_not_fill_an_observation_gap(self):
        times = pd.date_range("2017-11-01", "2017-11-04", freq="5min")
        times = times[(times < "2017-11-02 06:00") | (times > "2017-11-02 08:00")]
        repeated = times.repeat(2)
        intervals = observation_intervals(repeated, np.ones(len(repeated), dtype=bool))
        self.assertEqual(intervals, [(pd.Timestamp("2017-11-01"), pd.Timestamp("2017-11-02 05:55")),
                                     (pd.Timestamp("2017-11-02 08:05"), pd.Timestamp("2017-11-04"))])

    def test_known_amplitude_phase_and_offset_on_irregular_timestamps(self):
        times = pd.date_range("2017-10-01", periods=1000, freq="5min")
        times += pd.to_timedelta(np.arange(len(times)) % 7, unit="s")
        fit = fit_diurnal(times, signal(times, peak=23.5, amplitude=17, offset=42))
        self.assertAlmostEqual(fit["amplitude_ppm"], 17)
        self.assertAlmostEqual(fit["peak_hour_local"], 23.5)
        self.assertAlmostEqual(fit["offset_ppm"], 42)
        self.assertAlmostEqual(fit["r_squared"], 1)
        self.assertEqual(fit["peak_time_local"], "23:30")

    def test_delay_sign_and_midnight_crossing(self):
        self.assertAlmostEqual(float(wrap_hours(2 - 23)), 3)
        self.assertAlmostEqual(float(wrap_hours(23 - 2)), -3)
        self.assertEqual(clock_text(23 + 59.9 / 60), "00:00")

    def test_constant_has_no_defined_phase(self):
        times = pd.date_range("2017-10-01", periods=288, freq="5min")
        fit = fit_diurnal(times, np.full(len(times), 350.0))
        self.assertTrue(np.isnan(fit["peak_hour_local"]))
        self.assertTrue(np.isnan(fit["r_squared"]))

    def test_shared_intervals_do_not_require_identical_timestamps(self):
        times = pd.date_range("2017-10-01", "2017-10-06", freq="5min")
        frames = {}
        for index, (level, peak) in enumerate([("D1", 23), ("D2", 2), ("D3", 6)]):
            shifted = times + pd.Timedelta(seconds=index)
            frames[level] = pd.DataFrame({"localdatetime": shifted,
                                          "residual": signal(shifted, peak)})
        missing = frames["D2"]["localdatetime"].between("2017-10-03 11:00", "2017-10-03 14:00")
        frames["D2"].loc[missing, "residual"] = np.nan
        result = analyze_diurnal(frames)
        np.testing.assert_allclose(result.lags["lag_hours"], [3, 4, 7], atol=1e-10)
        included = result.day_coverage.loc[result.day_coverage["included"], "date"]
        self.assertEqual(included.dt.strftime("%Y-%m-%d").tolist(),
                         ["2017-10-02", "2017-10-04", "2017-10-05"])
        self.assertEqual(len(result.daily), 9)
        np.testing.assert_allclose(result.daily_lags["lag_hours"], np.repeat([3, 4, 7], 3), atol=1e-10)

    def test_no_complete_days_is_reported_without_inventing_daily_results(self):
        times = pd.date_range("2017-10-01 06:00", periods=100, freq="5min")
        frames = {level: pd.DataFrame({"localdatetime": times, "residual": signal(times, peak)})
                  for level, peak in [("D1", 14), ("D2", 18), ("D3", 22)]}
        result = analyze_diurnal(frames)
        self.assertTrue(result.daily.empty)
        self.assertFalse(result.day_coverage["included"].any())
        self.assertTrue(result.summary["daily_peak_iqr_hours"].isna().all())

    def test_duplicate_timestamps_are_not_silently_averaged(self):
        times = pd.date_range("2017-10-01", periods=10, freq="5min")
        frame = pd.DataFrame({"localdatetime": times, "residual": signal(times, 14)})
        frames = {level: frame.copy() for level in ["D1", "D2", "D3"]}
        frames["D2"] = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Повторные"):
            analyze_diurnal(frames)


class ColumnSelectionTests(unittest.TestCase):
    """Только искусственные сигналы; другие реальные колонки не исследуются."""

    def setUp(self):
        # Намеренно нестандартные названия/порядок; ID — условные тестовые числа.
        self.sensors = {"D4": 303, "D2": 202, "D1": 101}
        self.peaks = [23., 2., 6.]
        self.amplitudes = [9., 20., 14.]
        times = pd.date_range("2017-10-01", "2017-10-08", freq="h", inclusive="left")
        self.raw = pd.concat([
            pd.DataFrame({"localdatetime": times + pd.Timedelta(seconds=i),
                          "sensorid": sid,
                          "datavalue": signal(times + pd.Timedelta(seconds=i), peak, amplitude, 500.)})
            for i, (sid, peak, amplitude) in enumerate(zip(self.sensors.values(), self.peaks, self.amplitudes))
        ], ignore_index=True)

    def analyze(self, raw=None, start="2017-10-01", end="2017-10-08"):
        return analyze_column(self.raw if raw is None else raw, self.sensors, start, end)

    def test_actual_levels_ids_order_and_known_daily_vectors(self):
        before = self.raw.copy(deep=True)
        result = self.analyze()
        self.assertEqual(result.sensor_ids, self.sensors)
        self.assertEqual(result.H.columns.tolist(), list(self.sensors))
        self.assertEqual(result.A.columns.tolist(), list(self.sensors))
        expected_days = pd.date_range("2017-10-02", "2017-10-06", name="date")
        pd.testing.assert_index_equal(result.H.index, expected_days)
        pd.testing.assert_index_equal(result.A.index, expected_days)
        np.testing.assert_allclose(result.H.to_numpy(), np.tile(self.peaks, (5, 1)), atol=1e-10)
        np.testing.assert_allclose(result.A.to_numpy(), np.tile(self.amplitudes, (5, 1)), atol=1e-10)
        self.assertEqual(result.lags.pair.tolist(), ["D4 → D2", "D2 → D1", "D4 → D1"])
        np.testing.assert_allclose(result.lags.lag_hours, [3., 4., 7.], atol=1e-10)
        pd.testing.assert_frame_equal(self.raw, before)

    def test_only_selected_sensors_and_half_open_interval_are_used(self):
        start, end = pd.Timestamp("2017-10-02 06:00"), pd.Timestamp("2017-10-07 18:00")
        inside = self.raw.localdatetime.ge(start) & self.raw.localdatetime.lt(end)
        expected = self.analyze(self.raw.loc[inside], start, end)
        contaminated = self.raw.copy(deep=True)
        contaminated.loc[~inside, "datavalue"] = 50000.
        at_end = pd.DataFrame({"localdatetime": [end, end], "sensorid": [303, 303],
                               "datavalue": [-9999., 99999.]})
        unrelated = self.raw.assign(sensorid=999, datavalue=-9999.)
        contaminated = pd.concat([contaminated, at_end, unrelated], ignore_index=True)
        before = contaminated.copy(deep=True)
        actual = self.analyze(contaminated, start, end)
        for name in ("summary", "lags", "daily", "daily_lags", "day_coverage", "H", "A"):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))
        self.assertEqual(actual.common_intervals, expected.common_intervals)
        self.assertEqual(actual.start, start)
        self.assertEqual(actual.end_exclusive, end)
        self.assertEqual(actual.common_intervals[0][0], start + pd.Timedelta(hours=12, seconds=2))
        self.assertEqual(actual.H.index.tolist(), pd.date_range("2017-10-03", "2017-10-06").tolist())
        pd.testing.assert_frame_equal(contaminated, before)

    def test_calendar_retains_days_without_records_at_both_ends(self):
        result = self.analyze(start="2017-09-30", end="2017-10-09")
        coverage = result.day_coverage.set_index("date")
        self.assertEqual(coverage.index.tolist(), pd.date_range("2017-09-30", "2017-10-08").tolist())
        for day in ("2017-09-30", "2017-10-08"):
            self.assertFalse(coverage.loc[day, "included"])
            self.assertEqual(coverage.loc[day, "reason"], "неполное окно или пропуск")
        self.assertEqual(len(result.H), 5)

    def test_gaps_and_service_codes_are_not_filled(self):
        raw = self.raw.loc[~(self.raw.sensorid.eq(303)
                            & self.raw.localdatetime.eq(pd.Timestamp("2017-10-04 08:00")))].copy()
        raw.loc[raw.sensorid.eq(303) & raw.localdatetime.eq(pd.Timestamp("2017-10-05 08:00")), "datavalue"] = -9999.
        before = raw.copy(deep=True)
        result = self.analyze(raw)
        self.assertEqual(result.H.index.tolist(), pd.to_datetime(["2017-10-02", "2017-10-06"]).tolist())
        self.assertEqual(result.A.index.tolist(), result.H.index.tolist())
        for day in pd.date_range("2017-10-03", "2017-10-05"):
            self.assertFalse(result.day_coverage.set_index("date").loc[day, "included"])
        pd.testing.assert_frame_equal(raw, before)

    def test_no_complete_days_produces_empty_three_component_sequences(self):
        result = self.analyze(end="2017-10-02 18:00")
        self.assertEqual(result.H.shape, (0, 3))
        self.assertEqual(result.A.shape, (0, 3))
        self.assertEqual(result.H.columns.tolist(), list(self.sensors))
        self.assertFalse(result.day_coverage.included.any())

    def test_constant_input_does_not_invent_peak_times(self):
        result = self.analyze(self.raw.assign(datavalue=500.))
        self.assertTrue(result.H.isna().all().all())
        np.testing.assert_allclose(result.A, 0., atol=1e-10)
        self.assertTrue(result.daily.r_squared.isna().all())

    def test_repeated_records_require_a_decision(self):
        for delta in (0., 10.):
            with self.subTest(delta=delta):
                repeated = self.raw.iloc[[30]].copy()
                repeated["datavalue"] += delta
                raw = pd.concat([self.raw, repeated], ignore_index=True)
                with self.assertRaisesRegex(ValueError, "Повторные"):
                    self.analyze(raw)

    def test_missing_channel_and_no_common_window_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "нет измерений D4"):
            self.analyze(self.raw.loc[self.raw.sensorid.ne(303)])
        with self.assertRaisesRegex(ValueError, "Нет общих допустимых"):
            self.analyze(end="2017-10-01 18:00")

    def test_invalid_selection_and_time_bounds_are_rejected(self):
        selections = ({"D1": 101, "D2": 202}, {"D1": 101, "D2": 101, "D4": 303},
                      {"": 101, "D2": 202, "D4": 303}, {"D1": True, "D2": 202, "D4": 303})
        for sensors in selections:
            with self.subTest(sensors=sensors), self.assertRaises(ValueError):
                analyze_column(self.raw, sensors, "2017-10-01", "2017-10-08")
        for start, end in (("2017-10-08", "2017-10-01"), ("2017-10-01", "2017-10-01"),
                           (None, "2017-10-08"), ("2017-10-01T00:00Z", "2017-10-08T00:00Z")):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.analyze(start=start, end=end)
        aware = self.raw.copy()
        aware["localdatetime"] = aware.localdatetime.dt.tz_localize("UTC")
        with self.assertRaisesRegex(ValueError, "без часового пояса"):
            self.analyze(aware)


class OctoberReferenceTests(unittest.TestCase):
    """Единственные реальные измерения в тестах — неизменяемый эталон октября."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parents[2] / "Research_log/data/ch01"
        cls.manifest = json.loads((cls.directory / "provenance.json").read_text())
        for name, expected in cls.manifest["files_sha256"].items():
            if hashlib.sha256((cls.directory / name).read_bytes()).hexdigest() != expected:
                raise AssertionError(f"Изменён октябрьский снимок или эталон: {name}")
        cls.raw = pd.read_csv(cls.directory / "measurements.csv", parse_dates=["localdatetime"])
        cls.sensors = {level: info["sensor_id"] for level, info in cls.manifest["sensors"].items()}
        cls.result = analyze_column(cls.raw, cls.sensors, cls.manifest["start_inclusive"],
                                    cls.manifest["end_exclusive"])

    def assert_reference(self, suffix, actual, dates=()):
        expected = pd.read_csv(self.directory / f"reference_diurnal_{suffix}.csv", parse_dates=list(dates))
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected,
                                      check_dtype=False, check_exact=False, atol=1e-7, rtol=1e-7)

    def test_01_summary_reference(self):
        self.assert_reference("summary", self.result.summary.reset_index())
        self.assertEqual(self.result.summary.n_observations.tolist(), [7662, 7662, 7662])

    def test_02_lags_reference(self):
        self.assert_reference("lags", self.result.lags)

    def test_03_daily_reference(self):
        self.assert_reference("daily", self.result.daily, ["date"])
        self.assertEqual(self.result.H.shape, (22, 3))
        self.assertEqual(self.result.A.shape, (22, 3))

    def test_04_daily_lags_reference(self):
        self.assert_reference("daily_lags", self.result.daily_lags, ["date"])

    def test_05_day_coverage_reference(self):
        self.assert_reference("day_coverage", self.result.day_coverage, ["date"])
        self.assertEqual(int(self.result.day_coverage.included.sum()), 22)

    def test_06_common_intervals_reference(self):
        self.assert_reference("common_intervals", pd.DataFrame(self.result.common_intervals,
                              columns=["start", "end"]), ["start", "end"])
        self.assertEqual(len(self.result.common_intervals), 4)


if __name__ == "__main__":
    unittest.main()

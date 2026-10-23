"""Final stage 0.2 admission: monthly gates and real common-bin windows.

No Fourier coefficient, phase, database connection or archive read is needed.
The exact monthly-boundary tests mock daily fits, avoiding trigonometric
roundoff at the approved inclusive inequalities.
"""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SENSOR_DB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SENSOR_DB))
sys.path.insert(0, str(SENSOR_DB.parent / "Research_log"))

import chapter02 as ch
import stage02_final_admission as final


IDS = (1275, 996, 1012, 1035)


def monthly_rows(months=("2025-01", "2025-02"), overrides=None):
    overrides = overrides or {}
    rows = []
    for month in months:
        for sid in IDS:
            status = overrides.get((month, sid), "pass")
            rows.append(dict(month=month, sensorid=sid, status=status,
                             reason=f"synthetic_{status}_{sid}"))
    return pd.DataFrame(rows)


def bin_grid(start="2025-01-25", days=30, occupied=None):
    index = pd.date_range(start, periods=12 * days, freq="2h", name="bin_start")
    grid = pd.DataFrame(index=index)
    grid["complete"] = True if occupied is None else False
    if occupied is not None:
        grid.iloc[list(occupied), grid.columns.get_loc("complete")] = True
    for sid in IDS:
        grid[f"n_{sid}"] = grid.complete.astype(int)
        grid[f"doubtful_{sid}"] = False
        grid[f"plateau_review_{sid}"] = False
    return grid


class JointMonthlyAdmissionTests(unittest.TestCase):
    def test_joint_precedence_and_individual_reasons_are_preserved(self):
        source = monthly_rows(("2025-01", "2025-02", "2025-03"), {
            ("2025-02", 1275): "undetermined",
            ("2025-03", 1275): "undetermined",
            ("2025-03", 1035): "fail",
        })
        before = source.copy(deep=True)
        result = final.joint_months(source).set_index("month")
        self.assertEqual(result.status.tolist(), ["pass", "undetermined", "fail"])
        for month in result.index:
            for sid in IDS:
                original = source.loc[source.month.eq(month) & source.sensorid.eq(sid)].iloc[0]
                self.assertEqual(result.loc[month, f"status_{sid}"], original.status)
                self.assertEqual(result.loc[month, f"reason_{sid}"], original.reason)
        self.assertTrue(result.reason.str.len().gt(0).all())
        pd.testing.assert_frame_equal(source, before)

    def test_missing_duplicate_or_invalid_month_metadata_is_rejected(self):
        valid = monthly_rows()
        cases = {
            "missing_sensor_month": valid.iloc[1:].copy(),
            "duplicate_sensor_month": pd.concat([valid, valid.iloc[[0]]], ignore_index=True),
            "invalid_status": valid.assign(status="unknown"),
            "undefined_month": valid.assign(month=np.nan),
            "undefined_reason": valid.assign(reason=None),
        }
        for label, source in cases.items():
            with self.subTest(case=label), self.assertRaises(ValueError):
                final.joint_months(source)


class WindowAdmissionTests(unittest.TestCase):
    def test_original_calendar_has_492_half_open_30_day_windows(self):
        months = [str(m) for m in pd.period_range("2025-04", "2026-10", freq="M")]
        grid = bin_grid("2025-04-29", days=521)
        result = final.classify_windows(grid, monthly_rows(months))
        self.assertEqual(len(result), 492)
        self.assertEqual(pd.Timestamp(result.start.iloc[0]), pd.Timestamp("2025-04-29"))
        self.assertEqual(pd.Timestamp(result.end_exclusive.iloc[0]), pd.Timestamp("2025-05-29"))
        self.assertEqual(pd.Timestamp(result.start.iloc[-1]), pd.Timestamp("2026-09-02"))
        self.assertEqual(pd.Timestamp(result.end_exclusive.iloc[-1]), pd.Timestamp("2026-10-02"))
        self.assertTrue(result.common_bins.eq(360).all())
        self.assertTrue(result.ig_common_bins.eq(360).all())
        self.assertTrue(result.technical_computable.all())
        self.assertTrue(result.status.eq("pass").all())

    def test_technical_computability_uses_existing_two_cell_boundary(self):
        for count in (0, 1, 2):
            with self.subTest(common_cells=count):
                result = final.classify_windows(bin_grid(occupied=range(count)), monthly_rows()).iloc[0]
                self.assertEqual(result.common_bins, count)
                self.assertEqual(bool(result.technical_computable), count >= 2)
                self.assertEqual(result.status, "pass" if count >= 2 else "fail")
                self.assertFalse(result.requires_decision)
                if count < 2:
                    self.assertTrue(result.reason)

    def test_sparse_observation_coverage_does_not_introduce_a_new_cutoff(self):
        # Two real common cells, 20 days apart: no fill and no percent gate.
        grid = bin_grid(occupied=[0, 240])
        before = grid.copy(deep=True)
        result = final.classify_windows(grid, monthly_rows()).iloc[0]
        self.assertEqual(result.common_bins, 2)
        self.assertEqual(result.ig_common_bins, 2)
        self.assertTrue(result.technical_computable)
        self.assertEqual(result.status, "pass")
        pd.testing.assert_frame_equal(grid, before)

    def test_fail_month_is_not_silently_removed_from_the_original_window(self):
        grid = bin_grid(occupied=[0, 120])  # One January and one February cell.
        months = monthly_rows(overrides={("2025-02", 1275): "fail"})
        result = final.classify_windows(grid, months).iloc[0]
        self.assertEqual(result.common_bins, 2)
        self.assertEqual(result.ig_common_bins, 1)
        self.assertTrue(result.technical_computable)
        self.assertEqual(result.status, "fail")
        self.assertFalse(result.requires_decision)
        self.assertIn("2025-02", result.reason)
        self.assertEqual(json.loads(result.contributing_months), ["2025-01", "2025-02"])

    def test_undetermined_month_is_not_pass_and_fail_takes_priority(self):
        grid = bin_grid(occupied=[0, 120])
        for status, other_status, expected in (("undetermined", "pass", "undetermined"),
                                                ("undetermined", "fail", "fail")):
            with self.subTest(other_status=other_status):
                months = monthly_rows(overrides={
                    ("2025-01", 1275): status,
                    ("2025-02", 1035): other_status,
                })
                result = final.classify_windows(grid, months).iloc[0]
                self.assertEqual(result.status, expected)
                self.assertTrue(result.reason)
                self.assertEqual(result.common_bins, 2)

    def test_calendar_month_without_contributing_cells_does_not_supply_data(self):
        grid = bin_grid(occupied=[0, 1])
        months = monthly_rows(overrides={("2025-02", 1275): "fail"})
        result = final.classify_windows(grid, months).iloc[0]
        self.assertEqual(result.status, "pass")
        self.assertEqual(json.loads(result.contributing_months), ["2025-01"])
        calendar_statuses = json.loads(result.calendar_month_statuses)
        self.assertIn("2025-02", json.dumps(calendar_statuses))
        self.assertEqual(result.ig_common_bins, 2)

    def test_unresolved_excursion_inside_common_cells_requires_a_decision(self):
        grid = bin_grid(occupied=[0, 1])
        grid.loc[grid.index[0], "doubtful_1275"] = True
        result = final.classify_windows(grid, monthly_rows()).iloc[0]
        self.assertEqual(result.ig_common_bins, 2)
        self.assertTrue(result.has_unresolved_excursions)
        self.assertTrue(result.requires_decision)
        self.assertEqual(result.status, "undetermined")
        self.assertTrue(result.reason)

    def test_unresolved_flag_outside_common_cells_does_not_enter_the_result(self):
        grid = bin_grid(occupied=[0, 1])
        grid.loc[grid.index[120], "doubtful_1275"] = True
        result = final.classify_windows(grid, monthly_rows()).iloc[0]
        self.assertFalse(result.has_unresolved_excursions)
        self.assertFalse(result.requires_decision)
        self.assertEqual(result.status, "pass")

    def test_constant_plateau_diagnostic_does_not_become_an_exclusion_rule(self):
        grid = bin_grid(occupied=[0, 1])
        grid["plateau_review_1035"] = True
        result = final.classify_windows(grid, monthly_rows()).iloc[0]
        self.assertFalse(result.has_unresolved_excursions)
        self.assertFalse(result.requires_decision)
        self.assertEqual(result.status, "pass")

    def test_short_calendar_and_invalid_window_parameters_are_rejected(self):
        for days, width, step in ((29, 30, 1), (30, 0, 1), (30, 30, 0),
                                  (30, 1.5, 1), (30, 30, 1.5)):
            with self.subTest(days=days, width=width, step=step), self.assertRaises(ValueError):
                final.classify_windows(bin_grid(days=days), monthly_rows(),
                                       window_days=width, step_days=step)


class ExactMonthlyBoundaryTests(unittest.TestCase):
    def mocked_month(self, days=10, concentration=500.0, amplitude=8.0):
        frame = pd.DataFrame(dict(
            localdatetime=pd.date_range("2025-01-01", periods=days, freq="D"),
            sensorid=1275, variableid=56, level="C_air", datavalue=concentration,
            doubt_reason="",
        ))
        fit = dict(concentration=concentration, amplitude=amplitude,
                   n_obs=8, r_squared=0.0)
        bounds = dict(c_lo=concentration, c_hi=concentration,
                      a_lo=amplitude, a_hi=amplitude, guaranteed=True, bounds="exact")
        with patch.object(ch, "monthly_day_fit", return_value=fit), \
             patch.object(ch, "admission_day_bounds", return_value=bounds):
            monthly, _ = ch.monthly_admission(frame, {"C_air": 1275},
                                              "2025-01-01", "2025-02-01")
        return monthly.iloc[0]

    def test_exact_amplitude_eight_is_inclusive_without_a_tolerance(self):
        at_boundary = self.mocked_month(amplitude=8.0)
        just_below = self.mocked_month(amplitude=np.nextafter(8.0, -np.inf))
        self.assertEqual(at_boundary.status, "pass")
        self.assertEqual(just_below.status, "fail")
        self.assertIn("median_raw_amplitude_below_8", just_below.reason)

    def test_exact_concentration_boundaries_are_inclusive_without_a_tolerance(self):
        for concentration, expected in ((100.0, "pass"), (2000.0, "pass"),
                                       (np.nextafter(100.0, -np.inf), "fail"),
                                       (np.nextafter(2000.0, np.inf), "fail")):
            with self.subTest(concentration=concentration):
                result = self.mocked_month(concentration=concentration)
                self.assertEqual(result.status, expected)

    def test_nine_days_fail_and_exactly_ten_days_pass(self):
        nine, ten = self.mocked_month(days=9), self.mocked_month(days=10)
        self.assertEqual(nine.status, "fail")
        self.assertIn("fewer_than_10_valid_days", nine.reason)
        self.assertEqual(ten.status, "pass")
        self.assertEqual(ten.valid_days, 10)


if __name__ == "__main__":
    unittest.main()

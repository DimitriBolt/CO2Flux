"""Candidate threshold, calendar and actual-input checks; no phase calculation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stage02_candidate_selection as candidate
import stage02_final_admission as strict


def metric(**overrides):
    row = dict(month='2025-01', sensorid=1275, valid_days=10, minimum_valid_days=10,
               concentration_lower=100., concentration_upper=2000.,
               amplitude_lower=2.5, amplitude_upper=2.5, status='fail', reason='original')
    row.update(overrides)
    return pd.DataFrame([row])


def measurements():
    return pd.DataFrame(dict(localdatetime=pd.to_datetime([
        '2025-01-01 00:10', '2025-01-01 00:20', '2025-01-01 02:10',
        '2025-01-01 04:10', '2025-01-01 06:10']),
        datavalue=[100., 200., 300., 400., 500.], technically_retained=[True, True, True, True, False],
        doubtful=[False, False, False, True, False], plateau_review=False))


class CandidateMonthlyTests(unittest.TestCase):
    def test_inclusive_2_5_and_neighboring_float(self):
        for value, expected in [(np.nextafter(2.5, 0), 'fail'), (2.5, 'pass'),
                                (np.nextafter(2.5, np.inf), 'pass')]:
            result = candidate.classify_months(metric(amplitude_lower=value, amplitude_upper=value), 2.5)
            self.assertEqual(result.status.iloc[0], expected)

    def test_straddling_uncertainty_never_becomes_pass(self):
        result = candidate.classify_months(metric(amplitude_lower=2.49, amplitude_upper=2.51), 2.5)
        self.assertEqual(result.status.iloc[0], 'undetermined')

    def test_days_and_concentration_criteria_are_unchanged(self):
        for overrides in [dict(valid_days=9), dict(concentration_upper=99., concentration_lower=99.),
                          dict(concentration_lower=2001., concentration_upper=2001.)]:
            self.assertEqual(candidate.classify_months(metric(**overrides), 2.5).status.iloc[0], 'fail')
        self.assertEqual(candidate.classify_months(metric(minimum_valid_days=9), 2.5).status.iloc[0], 'undetermined')
        self.assertEqual(candidate.classify_months(metric(), 2.5).status.iloc[0], 'pass')

    def test_strict8_has_original_reason_and_boundary(self):
        result = candidate.classify_months(metric(), 8)
        self.assertEqual(result.reason.iloc[0], 'median_raw_amplitude_below_8')
        self.assertEqual(candidate.classify_months(metric(amplitude_lower=8., amplitude_upper=8.), 8).status.iloc[0], 'pass')

    def test_inputs_are_not_mutated_and_metrics_stay_identical(self):
        frame = metric()
        before = frame.copy(deep=True)
        result = candidate.classify_months(frame, 2.5)
        pd.testing.assert_frame_equal(frame, before)
        pd.testing.assert_frame_equal(result.drop(columns=['status', 'reason']), before.drop(columns=['status', 'reason']))

    def test_invalid_or_missing_bounds_rejected(self):
        for threshold in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                candidate.classify_months(metric(), threshold)
        with self.assertRaises(ValueError):
            candidate.classify_months(metric(amplitude_lower=np.nan), 2.5)

    def test_frozen_strict8_all_632_decisions_and_daily_metrics_reproduce(self):
        metrics = pd.read_csv(strict.OUT / 'channel_months.csv')
        daily = pd.read_csv(strict.OUT / 'channel_days.csv')
        before = metrics.copy(deep=True)
        check = candidate.reproduce_strict(metrics, daily)
        self.assertEqual(check['sensor_month_decisions'], 632)
        self.assertEqual(check['numeric_metrics_checked'], 632 * 8)
        pd.testing.assert_frame_equal(metrics, before)


class CandidateCalendarTests(unittest.TestCase):
    def test_calendar_13_month_run_includes_year_transition(self):
        months = pd.period_range('2025-05', '2026-05', freq='M').astype(str).tolist()
        self.assertEqual(candidate.month_runs(months), [months])
        self.assertEqual(len(candidate.month_runs(months[:-1])[0]), 12)

    def test_gap_splits_positive_months_and_duplicates_do_not_extend_run(self):
        result = candidate.month_runs(['2025-01', '2025-03', '2025-04', '2025-04'])
        self.assertEqual(result, [['2025-01'], ['2025-03', '2025-04']])

    def test_partial_month_and_month_without_actual_common_data_are_ineligible(self):
        joint = pd.DataFrame(dict(month=['2025-01', '2025-02', '2025-03', '2025-04'], status='pass'))
        grid = pd.DataFrame(dict(complete=[True, True, False, True]),
                            index=pd.to_datetime(['2025-01-15', '2025-02-15', '2025-03-15', '2025-04-01']))
        result = candidate.eligible_months(joint, grid, pd.Timestamp('2025-01-15'), pd.Timestamp('2025-04-02'))
        self.assertEqual(result.eligible.tolist(), [False, True, False, False])

    def test_selector_omits_doubts_and_point_exclusions_without_mutating_original(self):
        work = measurements()
        before = work.copy(deep=True)
        selected = candidate.candidate_points(work, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'))
        self.assertEqual(selected.datavalue.tolist(), [100., 200., 300.])
        pd.testing.assert_frame_equal(work, before)

    def test_bins_use_actual_means_and_real_centers_without_filling(self):
        works = {sid: measurements() for sid in strict.IDS}
        grid = candidate.candidate_grid(works, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'))
        self.assertEqual(len(grid), 12)
        self.assertEqual(grid.complete.sum(), 2)
        self.assertEqual(grid.C_1275.iloc[:2].tolist(), [150., 300.])
        self.assertTrue(grid.C_1275.iloc[2:].isna().all())
        self.assertEqual(grid.bin_center.iloc[0], pd.Timestamp('2025-01-01 01:00'))
        self.assertFalse(grid[[f'doubtful_{sid}' for sid in strict.IDS]].any().any())
        self.assertEqual(candidate.longest_missing_cells(grid.complete), 10)

    def test_old_window_precheck_retains_sparse_cells_without_new_percentage_gate(self):
        works = {sid: measurements() for sid in strict.IDS}
        grid = candidate.candidate_grid(works, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-31'))
        rows = pd.concat([metric(sensorid=sid, status='pass', reason='all_three_criteria_invariant_to_flagged_samples') for sid in strict.IDS])
        result = strict.classify_windows(grid, rows)
        self.assertTrue(result.admitted.iloc[0])
        self.assertEqual(result.common_bins.iloc[0], 2)
        self.assertEqual(result.possible_bins.iloc[0], 360)

    def test_meteorological_seasons_assign_december_to_following_winter(self):
        self.assertEqual(candidate.season('2025-12-15'), '2026-DJF')
        self.assertEqual(candidate.season('2026-02-28'), '2026-DJF')
        self.assertEqual(candidate.season('2026-03-01'), '2026-MAM')


if __name__ == '__main__':
    unittest.main()

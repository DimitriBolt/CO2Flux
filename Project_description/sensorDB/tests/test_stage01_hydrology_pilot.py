"""Scientific preservation checks for the limited stage 0.1-B hydrology input."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stage01_hydrology_pilot import channel_coverage, nonservice, plot_arrays


def observations(times, values):
    return pd.DataFrame({'localdatetime': pd.to_datetime(times), 'datavalue': values})


class TestHydrologyPreservation(unittest.TestCase):
    def test_service_value_and_missing_observation_remain_distinct(self):
        g = observations(['2025-05-01 00:00:01', '2025-05-01 00:15:00',
                          '2025-05-01 00:45:01'], [-10.1, -9999.0, -9.9])
        original = g.copy(deep=True)
        result = channel_coverage(g, '2025-05-01', '2025-05-01 01:00')
        self.assertEqual((result['expected_slots'], result['observed_slots'],
                          result['service_rows'], result['missing_observation_slots'],
                          result['nonservice_rows']), (4, 3, 1, 1, 2))
        t, y = plot_arrays(g)
        self.assertEqual(np.isnan(y).sum(), 2)  # one service and one missing-slot separator
        np.testing.assert_array_equal(y[np.isfinite(y)], [-10.1, -9.9])
        np.testing.assert_array_equal(t[np.isfinite(y)], g.localdatetime.iloc[[0, 2]].to_numpy())
        pd.testing.assert_frame_equal(g, original, check_exact=True)

    def test_native_second_jitter_does_not_create_a_gap_or_change_time(self):
        g = observations(['2025-05-01 00:00:01', '2025-05-01 00:15:00',
                          '2025-05-01 00:30:01'], [12.1, 12.2, 12.3])
        t, y = plot_arrays(g)
        np.testing.assert_array_equal(t, g.localdatetime.to_numpy())
        np.testing.assert_array_equal(y, g.datavalue.to_numpy())
        self.assertEqual(channel_coverage(g, '2025-05-01', '2025-05-01 00:45')['missing_observation_slots'], 0)

    def test_multiple_observations_in_one_slot_are_reported_without_averaging(self):
        g = observations(['2025-05-01 00:00:00', '2025-05-01 00:00:01',
                          '2025-05-01 00:15:00'], [10.0, 20.0, 30.0])
        original = g.copy(deep=True)
        c = channel_coverage(g, '2025-05-01', '2025-05-01 00:30')
        self.assertEqual(c['rows'], 3)
        self.assertEqual(c['ambiguous_slots'], 1)
        self.assertEqual(c['observed_slots'], 2)
        pd.testing.assert_frame_equal(g, original, check_exact=True)

    def test_empty_interval_has_missing_slots_not_measured_zeros(self):
        g = observations([], [])
        c = channel_coverage(g, '2025-05-01', '2025-05-01 01:00')
        self.assertEqual((c['rows'], c['nonservice_rows'], c['missing_observation_slots']), (0, 0, 4))
        self.assertTrue(pd.isna(c['first_time']))
        self.assertTrue(pd.isna(c['min_nonservice']))

    def test_negative_matric_potential_is_a_nonservice_measurement(self):
        values = pd.Series([-11.4, -9.5, -9999.0, np.nan])
        self.assertEqual(nonservice(values).tolist(), [True, True, False, False])
        self.assertEqual(values.iloc[2], -9999.0)


if __name__ == '__main__':
    unittest.main()

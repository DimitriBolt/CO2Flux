"""Synthetic checks of Stage 0.5 preprocessing, support and signed lag inference."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import short_lag_resolution as lag


class ShortLagTests(unittest.TestCase):
    pair = lag.PAIRS[0]
    start = lag.START + lag.DAY
    index = pd.date_range(start, periods=7*96, freq="15min")

    def synthetic(self, delay=4, gap=False, independent=False, strong_background=True):
        rng = np.random.default_rng(2950)
        kernel = np.exp(-np.arange(-8,9)**2 / 10)
        full = np.convolve(rng.normal(size=800), kernel, "same")
        t = np.arange(len(self.index))
        upper = full[64+t]
        lower = full[64+t-delay] if not independent else np.convolve(rng.normal(size=800), kernel, "same")[64+t]
        by_slot = np.sin(2*np.pi*t/96) if strong_background else 0
        drift = t*.009 if strong_background else 0
        a,b = self.pair
        data = pd.DataFrame({f"time_{a}": self.index + pd.Timedelta(seconds=4),
            f"time_{b}": self.index + pd.Timedelta(seconds=14),
            f"C_{a}": 400+upper+30*by_slot+drift, f"C_{b}":500+lower+17*by_slot-2*drift}, index=self.index)
        return data.drop(index=self.index[[155,411,501]]) if gap else data

    def test_phase_conversion(self):
        np.testing.assert_allclose(lag.phase_minutes([0,math.pi/12,-math.pi/6,-math.pi]),
                                   [0,60,120,720], atol=1e-12)

    def test_signed_zero_and_short_delay_with_background_and_gaps(self):
        for delay in (0, 4, -5):
            with self.subTest(delay=delay):
                data = self.synthetic(delay, gap=True, strong_background=delay != 0)
                prepared, designs = lag.window_design(data,self.pair,self.start)
                self.assertFalse(designs[15].reason)
                self.assertEqual(lag.maxima(designs[15].correlation(),designs[15].shifts),[delay*15])
                self.assertTrue(np.isnan(prepared[1][[155,411,501]]).all())
                self.assertEqual(len(designs[120].anchors),79)
                if delay == 4:
                    self.assertEqual(lag.maxima(designs[120].correlation(),designs[120].shifts),[0])
                for h, design in designs.items():
                    errors = lag.scalar_check(design, lag.scalar_baseline_rates(prepared,self.pair)[h])
                    self.assertLess(max(errors),5e-14)

    def test_zero_with_different_drifts_can_shift_raw_peak_but_remains_unresolved(self):
        data = self.synthetic(0, gap=True, strong_background=True)
        tables, _, _, _, _ = lag.analyze_window(data,self.pair,self.start,"synthetic")
        fine = tables[0][tables[0].step_min == 15].iloc[0]
        self.assertEqual(fine.base_maxima,[-15])
        self.assertIn(0,fine.competing)
        self.assertTrue(pd.isna(fine.stable_lag_min))

    def test_coarse_grid_cannot_represent_short_shift(self):
        data = self.synthetic(delay=4)
        _, designs = lag.window_design(data,self.pair,self.start)
        fine, coarse = designs[15], designs[120]
        self.assertEqual(lag.maxima(fine.correlation(),fine.shifts),[60])
        self.assertEqual(coarse.shifts.tolist(),[-240,-120,0,120,240])
        self.assertNotIn(60,lag.maxima(coarse.correlation(),coarse.shifts))
        data = self.synthetic(delay=8)
        _, designs = lag.window_design(data,self.pair,self.start)
        self.assertEqual(lag.maxima(designs[120].correlation(),designs[120].shifts),[120])

    def test_independent_dynamics_and_perturbations(self):
        data = self.synthetic(independent=True, gap=True)
        tables, _, _, _, _ = lag.analyze_window(data,self.pair,self.start,"synthetic")
        summary,support,_,_ = tables
        fine = summary[summary.step_min == 15].iloc[0]
        self.assertTrue(fine.base_maxima)
        self.assertIsNone(fine.stable_lag_min)
        self.assertGreater(len(fine.competing),1)
        for sensor in self.pair:
            case = support[(support.step_min == 15) & (support.variant == f"drop_max_abs_q_{sensor}")].iloc[0]
            first,second = map(pd.Timestamp,case.removed_cells.split("|"))
            self.assertEqual(second-first,pd.Timedelta(minutes=15))
            self.assertEqual(case.occupied_cells,669-2)

    def test_full_only_filters_both_grids_and_retains_source_time(self):
        data = self.synthetic(gap=True)
        prepared, design = lag.window_design(data,self.pair,self.start,full_only=True)
        self.assertEqual(design[15].occupied, (84-3)*8)
        self.assertEqual(design[120].occupied,84-3)
        self.assertTrue(np.isin(prepared[4],[0,8]).all())
        row = next(lag.coarse_cells(prepared,self.pair,self.start,"full_2h_only"))
        self.assertEqual(len(row["source_times_996"].split("|")),8)
        self.assertEqual(row["mean_source_time_996"],self.start+pd.Timedelta(minutes=52.5,seconds=4))
        self.assertEqual(row["mean_source_time_1012"],self.start+pd.Timedelta(minutes=52.5,seconds=14))

    def test_missing_gap_is_not_crossed_or_sampled_at_other_shifts(self):
        data = self.synthetic().drop(index=self.index[220:500])
        _, designs = lag.window_design(data,self.pair,self.start)
        self.assertTrue(designs[15].reason)
        self.assertTrue(designs[120].reason)
        self.assertTrue(np.isnan(designs[15].q[219:500]).all())
        self.assertFalse(any(219 <= i+s//15 < 500 for i in designs[15].anchors for s in designs[15].shifts))

    def test_broad_maximum_and_boundary_are_unresolved(self):
        shifts = np.arange(-240,241,120)
        one = np.array([.1,.5,.5,.49,.2])
        two = np.array([.1,.4,.47,.49,.2])
        r = lag.describe_peak(one,[one,two],shifts,True)
        self.assertEqual(r["base_maxima"],[-120,0])
        self.assertEqual(r["competing"],[-120,0,120])
        self.assertIsNone(r["stable_lag_min"])
        r = lag.describe_peak(np.array([1,.1,0,0,0]),[np.array([1,.1,0,0,0])],shifts,True)
        self.assertEqual(r["status"],"boundary_peak")


if __name__ == "__main__":
    unittest.main()

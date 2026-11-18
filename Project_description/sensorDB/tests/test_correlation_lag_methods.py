"""Known shifts, fixed support, archived deletions and the two-hour gate."""
from pathlib import Path
import json
from statistics import correlation, linear_regression, median
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import correlation_lag_methods as lag


class LevelLagTests(unittest.TestCase):
    start = pd.Timestamp("2025-05-02")
    pair = lag.archive.PAIRS[0]

    def synthetic(self, delay):
        index = pd.date_range(self.start - pd.Timedelta(days=1), periods=9*96, freq="15min")
        t = np.arange(len(index))
        latent = np.random.default_rng(56).normal(size=len(t)+40)
        common = 3*np.cos(2*np.pi*t/96) + .0008*t
        # Exact shifted observations, with background plus irregular variation.
        upper = common + latent[t+20]
        lower = 3*np.cos(2*np.pi*(t-delay)/96) + .0008*(t-delay) + latent[t+20-delay]
        frame = pd.DataFrame({"C_996":400+upper, "C_1012":500+lower}, index=index)
        return frame.drop(index=index[[157, 444]])

    def variants(self):
        names = ["base","window_-1d","window_+1d",*lag.BACKGROUNDS,
                 "drop_day_2","drop_min_C_996","drop_max_abs_q_1012","full_2h_only"]
        rows = []
        for name in names:
            shift = {"window_-1d":-1, "window_+1d":1}.get(name,0)
            removed = {"drop_day_2":pd.date_range(self.start+lag.archive.DAY, periods=96, freq="15min"),
                       "drop_min_C_996":[self.start], "drop_max_abs_q_1012":[self.start,self.start+pd.Timedelta(minutes=15)]}.get(name,[])
            rows.append(dict(variant=name,start=self.start+shift*lag.archive.DAY,
                             removed_cells="|".join(t.isoformat() for t in removed)))
        return pd.DataFrame(rows)

    def test_known_signed_delays_through_background_and_gaps(self):
        for steps in (8,-8,0):
            for method in "AB":
                with self.subTest(steps=steps, method=method):
                    result, comparison = lag.analyze_method(self.synthetic(steps), self.pair,
                                                          self.variants(), method)
                    self.assertEqual(result[15][0]["stable_lag_min"],steps*15)
                    self.assertEqual(result[120][0]["stable_lag_min"],steps*15)
                    self.assertEqual(comparison,"compatible_nearest")
                    labels = set(result[15][2].variant)
                    self.assertEqual(labels.intersection(lag.BACKGROUNDS),
                                     set(lag.BACKGROUNDS) if method=="B" else set())

    def test_one_support_across_shifts_and_no_wrap_or_gap_filling(self):
        values = np.random.default_rng(7).normal(size=(672,2))
        values[[79,331],0] = np.nan
        values[540,1] = np.nan
        shifts, curve, anchors, _, reason = lag.correlation_curve(values,15)
        self.assertEqual(reason,"")
        expected = [i for i in range(16,len(values)-16)
                    if np.isfinite(values[i-16:i+17]).all()]
        np.testing.assert_array_equal(anchors, expected)
        for shift,r in zip(shifts,curve):
            other = [i+int(shift)//15 for i in expected]
            self.assertAlmostEqual(r,correlation(values[expected,0],values[other,1]),places=13)

    def test_paired_exclusion_and_full_bins_on_both_grids(self):
        frame = self.synthetic(4)
        variants = self.variants().set_index("variant",drop=False)
        for name in ("drop_max_abs_q_1012","full_2h_only"):
            v = variants.loc[name]
            if name=="drop_max_abs_q_1012":
                v = v.copy()
                v['removed_cells'] = self.start.isoformat()+"|"+(self.start+pd.Timedelta(minutes=15)).isoformat()
            fine,mask = lag.level_signals(frame,self.pair,v,"B",15)
            coarse,second = lag.level_signals(frame,self.pair,v,"B",120)
            np.testing.assert_equal(mask,second)
            if name=="full_2h_only":
                self.assertTrue(np.all(np.isin(mask.reshape(-1,8).sum(axis=1),[0,8])))
            else:
                self.assertFalse(mask[:2].any())
            for k,part in enumerate(fine.reshape(-1,8,2)):
                good = mask[k*8:k*8+8]
                if good.any():
                    np.testing.assert_allclose(coarse[k],part[good].mean(axis=0))
                else:
                    self.assertTrue(np.isnan(coarse[k]).all())

    def test_unresolved_fine_grid_never_calls_coarse_correlations(self):
        frame = self.synthetic(4)
        frame.loc[:,:] = 1.0
        with patch.object(lag,"analyze_grid",wraps=lag.analyze_grid) as observed:
            results, outcome = lag.analyze_method(frame,self.pair,self.variants(),"A")
        self.assertEqual(list(results),[15])
        self.assertEqual(observed.call_count,1)
        self.assertEqual(outcome,"no_fine_estimate")
        self.assertEqual(results[15][0]["status"],"no_defined_curve")

    def test_resolution_outcomes_include_midpoint_and_missing_support(self):
        fine = {"stable_lag_min":60}
        for shift, status, expected in [
            (120,"stable_nonzero_grid_peak","compatible_tied_nearest"),
            (0,"stable_zero_grid_peak","compatible_tied_nearest"),
            (-120,"stable_nonzero_grid_peak","different_discrete_estimate"),
            (None,"competing_or_wide_peak","lost_resolvability"),
            (None,"required_perturbation_unavailable","undetermined")]:
            with self.subTest(shift=shift,status=status):
                self.assertEqual(lag.compare_grids(fine,dict(stable_lag_min=shift,status=status)),expected)

    def test_published_baselines_against_scalar_reconstruction(self):
        # The archived selected points are a second input path to the four base episodes.
        points = lag.archive.read_csv(lag.archive.OUT/"selected_source_points.csv.gz",
                                      parse_dates=["fine_start"])
        curves = lag.archive.read_csv(lag.OUT/"curves_ab.csv.gz")
        audit = lag.archive.read_csv(lag.OUT/"support_ab.csv.gz")
        summary = lag.archive.read_csv(lag.OUT/"summary.csv")
        for (pair, selection, start), part in points.groupby(["pair","selection","baseline_start"]):
            pos = ((part.fine_start-pd.Timestamp(start))/pd.Timedelta(minutes=15)).astype(int)
            ids = next(ids for ids in lag.archive.PAIRS if lag.archive.pair_name(ids)==pair)
            offsets = range(-16,17)
            positions = set(pos)
            support = sorted(t for t in positions if all(t+k in positions for k in offsets))
            for method in "AB":
                channels = []
                for sid in ids:
                    level = dict(zip(pos, part[f"C_{sid}"]))
                    if method=="B":
                        by_slot = {}
                        for i,value in level.items():
                            by_slot.setdefault(i%96,[]).append(value)
                        profile = {slot:median(values) for slot,values in by_slot.items()}
                        level = {i:value-profile[i%96] for i,value in level.items()}
                        trend = linear_regression(level.keys(),level.values())
                        level = {i:y-trend.intercept-trend.slope*i for i,y in level.items()}
                    channels.append(level)
                def episode(table):
                    return table[(table.pair==pair) & (table.selection==selection)
                                 & (table.method==method) & (table.step_min==15)]
                base_support = episode(audit).query("variant == 'base'").iloc[0]
                self.assertEqual(json.loads(base_support.anchor_offsets),support)
                lines = episode(curves).pivot(index="shift_min", columns="variant", values="r")
                expected = [correlation([channels[0][i] for i in support],
                                        [channels[1][i+k] for i in support]) for k in offsets]
                np.testing.assert_allclose(lines.base,expected,rtol=0,atol=1e-12)
                variants = lines.drop(columns=["envelope_min","envelope_max"]).dropna(axis=1)
                np.testing.assert_array_equal(lines.envelope_min,variants.min(axis=1))
                np.testing.assert_array_equal(lines.envelope_max,variants.max(axis=1))
                best_lower = variants.min(axis=1).max()
                competing = lines.index[variants.max(axis=1)>=best_lower-lag.archive.TIE].tolist()
                row = episode(summary).iloc[0]
                self.assertEqual(json.loads(row.competing),competing)
                self.assertEqual(json.loads(row.base_maxima),[int(lines.base.idxmax())])
                self.assertAlmostEqual(row.peak_r,max(expected),places=12)

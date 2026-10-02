"""Stage 0.2 tests: original records, independent sensor policies and real bins."""
from decimal import Decimal
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Research_log'))
import stage02_prepare as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp('2025-08-14')
        self.end = self.start + pd.Timedelta(days=1)
        self.exclusions = pd.DataFrame(columns=['level', 'start', 'end'])

    def basalt(self, values):
        times = pd.date_range(self.start, periods=len(values), freq='15min')
        frame = pd.concat([pd.DataFrame(dict(sensorid=sid, variableid=9,
                       localdatetime=times, datavalue=values)) for sid in (994,1010,1026)],ignore_index=True)
        frame['source_row'] = range(len(frame))
        return frame

    def air(self, values):
        n = len(values)
        return pd.DataFrame(dict(valueid=range(n), sensorid=1275, variableid=56,
            localdatetime=pd.date_range(self.start, periods=n, freq='h'), datavalue=values, source_row=range(n)))

    def test_basalt_range_boundaries_and_no_monthly_or_HA(self):
        raw = self.basalt([-9999., -1., 0., 1., 3000., 3000.1, np.nan])
        before = raw.copy(deep=True)
        with patch('chapter02.monthly_admission', side_effect=AssertionError('monthly called')), \
             patch('chapter02.calculate_trajectories', side_effect=AssertionError('H/A called')):
            views,audits=prep.prepare_basalt(raw,self.start,self.end,self.exclusions)
        pd.testing.assert_frame_equal(raw,before)
        for name,w in views.items():
            self.assertEqual(w.technically_retained.sum(),2)
            self.assertEqual(w.exclusion_reason.tolist(),['service_code','outside_0_3000','outside_0_3000','','','outside_0_3000','nonfinite'])
            self.assertEqual(audits[name].above_3000.sum(),1)
            self.assertTrue(w.loc[w.excluded,'datavalue'].isna().all())

    def test_conflicting_and_identical_duplicates_retain_every_raw_row(self):
        raw = self.basalt([400.,410.,420.])
        raw = pd.concat([raw,raw.iloc[[0,1]].assign(source_row=[100,101]),
                         raw.iloc[[1]].assign(datavalue=411.,source_row=102)],ignore_index=True)
        views,audits=prep.prepare_basalt(raw,self.start,self.end,self.exclusions)
        w,a=views['C_5'],audits['C_5']
        self.assertEqual(len(a),6)
        self.assertEqual(len(w),3)
        self.assertEqual(w.iloc[0].exact_duplicate_extra,1)
        self.assertEqual(w.iloc[1].exclusion_reason,'conflicting_timestamp')
        self.assertEqual(a.excluded.sum(),3)
        self.assertEqual(set(a.loc[a.excluded,'datavalue']),{410.,411.})

    def test_air_decimal_and_no_borrowed_upper_limit(self):
        raw=self.air([Decimal('15.38000'),Decimal('4000.00001'),Decimal('-1'),Decimal('0'),Decimal('-9999'),None])
        before=raw.copy(deep=True)
        w,a=prep.prepare_air(raw,self.start,self.end)
        pd.testing.assert_frame_equal(raw,before)
        self.assertEqual(w.technically_retained.sum(),4)
        self.assertEqual(w.doubtful.sum(),4)
        self.assertEqual(w.datavalue.iloc[1],Decimal('4000.00001'))
        self.assertIn('nonpositive_air_value',w.doubt_reason.iloc[2])
        self.assertEqual(w.exclusion_reason.iloc[4],'service_code')
        self.assertEqual(w.exclusion_reason.iloc[5],'nonfinite')
        self.assertEqual(a.datavalue.tolist(),raw.datavalue.tolist())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'air.parquet'
            w.to_parquet(path,index=False)
            pd.testing.assert_frame_equal(w,pd.read_parquet(path))

    def test_air_conflicting_times_not_arbitrarily_resolved(self):
        raw=self.air([Decimal('400'),Decimal('401')])
        raw.loc[1,'localdatetime']=raw.localdatetime.iloc[0]
        w,a=prep.prepare_air(raw,self.start,self.end)
        self.assertEqual(len(w),2)
        self.assertTrue(w.doubt_reason.str.contains('repeated_air_timestamp').all())
        self.assertTrue(w.technically_retained.all())

    def test_context_matches_full_cleaner_near_window_boundary(self):
        # An isolated spike outside the requested period can affect adjacency;
        # four real neighbours preserve that decision without +/-12h gating.
        raw=self.basalt([400.,401.,402.,1200.,403.,404.,405.,406.,407.,408.,409.,410.,411.,412.,413.,414.])
        s,e=self.start+pd.Timedelta(minutes=60),self.start+pd.Timedelta(minutes=150)
        full,_=prep.prepare_basalt(raw,self.start,self.end,self.exclusions)
        small,_=prep.prepare_basalt(raw,s,e,self.exclusions)
        for name in full:
            cols=['localdatetime','datavalue','exclusion_reason','doubt_reason','despike_window_eligible']
            pd.testing.assert_frame_equal(prep.clip(full[name],s,e)[cols].reset_index(drop=True),small[name][cols].reset_index(drop=True))

    def test_pointwise_spike_and_ambiguous_zero_MAD(self):
        raw=self.basalt([398.,399.,400.,800.,400.,401.,402.])
        w,_=prep.prepare_basalt(raw,self.start,self.end,self.exclusions)
        self.assertEqual(w['C_5'].exclusion_reason.iloc[3],'isolated_7_sample_MAD_spike')
        raw=self.basalt([400.,400.,400.,800.,400.,400.,400.])
        w,_=prep.prepare_basalt(raw,self.start,self.end,self.exclusions)
        self.assertEqual(w['C_5'].doubt_reason.iloc[3],'zero_MAD_deviation')
        self.assertTrue(w['C_5'].technically_retained.iloc[3])

    def test_bins_boundaries_count_real_rows_no_fill(self):
        b=self.basalt([400.,401.,402.,403.,404.])
        a=self.air([Decimal('400'),Decimal('401')])
        end=self.start+pd.Timedelta(hours=3)
        views,audits=prep.prepare(b,a,self.start,end,self.exclusions)
        grid=prep.count_grid(views,audits,self.start,end,1)
        self.assertEqual(grid.C_5_raw.tolist(),[4,1,0])
        self.assertEqual(grid.C_air_retained.tolist(),[1,1,0])
        self.assertEqual(grid.common_retained.tolist(),[True,True,False])
        self.assertFalse(grid.common_without_doubts.any())
        self.assertFalse(any('datavalue' in c for c in grid.columns))
        self.assertEqual(prep.grid_runs(grid,'retained').bins.tolist(),[2])
        self.assertEqual(len(prep.clip(a,self.start,self.start+pd.Timedelta(hours=1))),1)

    def test_all_excluded_is_empty_coverage_not_zero_concentration(self):
        views,audits=prep.prepare(self.basalt([4001.]*8),self.air([Decimal('400')]),self.start,self.end,self.exclusions)
        stats,steps,gaps=prep.summarize(views,audits,self.start,self.end)
        kept=stats[(stats.channel=='C_5') & (stats.state=='retained')].iloc[0]
        self.assertEqual(kept.rows,0)
        self.assertTrue(pd.isna(kept.max_internal_gap_s))
        grid=prep.count_grid(views,audits,self.start,self.end,2)
        self.assertEqual(grid.C_5_raw.sum(),8)
        self.assertEqual(grid.C_5_retained.sum(),0)
        self.assertFalse(grid.common_retained.any())
        self.assertTrue(prep.grid_runs(grid,'retained').empty)

    def test_empty_air_and_infinite_values(self):
        raw=self.air([Decimal('Infinity'),Decimal('-Infinity'),Decimal('NaN')])
        w,_=prep.prepare_air(raw,self.start,self.end)
        self.assertEqual(w.exclusion_reason.tolist(),['nonfinite']*3)
        w,_=prep.prepare_air(raw,self.end,self.end+pd.Timedelta(days=1))
        self.assertTrue(w.empty)

    def test_individual_exclusions_only_on_approved_dates(self):
        raw=self.basalt([400.,401.,402.])
        exc=pd.DataFrame([dict(level='D1',start=self.start,end=self.start)])
        views,_=prep.prepare_basalt(raw,self.start,self.end,exc)
        self.assertEqual(views['C_5'].exclusion_reason.iloc[0],'approved_individual_exclusion')
        self.assertTrue(views['C_20'].technically_retained.all())

    def test_candidates_are_daily_half_open_and_equal_zero_scores_tie(self):
        end=self.start+pd.Timedelta(days=32)
        views,audits=prep.prepare(self.basalt([4001.]*8),self.air([Decimal('400')]),self.start,end,self.exclusions)
        grids={h:prep.count_grid(views,audits,self.start,end,h) for h in (1,2)}
        with patch.object(prep,'START',self.start),patch.object(prep,'END',end):
            candidates=prep.candidate_windows(grids,views)
        self.assertEqual(len(candidates),3)
        self.assertEqual(candidates.coverage_rank.tolist(),[1,1,1])
        self.assertFalse(candidates.has_any_common_retained_bin.any())
        self.assertTrue((candidates.end_exclusive-candidates.start).eq(pd.Timedelta(days=30)).all())


if __name__=='__main__':
    unittest.main()

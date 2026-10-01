"""Approved monthly gates and their separation from Chapter 1 H/A."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Project_description.Research_log import chapter02 as ch

EMPTY = pd.DataFrame(columns=['level','start','end'])
SENSORS = {'D1':994,'D2':1010,'D3':1026}


def raw(values, times=None, sid=994):
    if times is None: times=pd.date_range('2019-01-01',periods=len(values),freq='15min')
    return pd.DataFrame(dict(localdatetime=times,datavalue=values,sensorid=sid,variableid=9))


def harmonic(start='2019-01-01', days=12, amplitude=20, offset=500, freq='15min'):
    times=pd.date_range(start,pd.Timestamp(start)+pd.Timedelta(days=days),freq=freq,inclusive='left')
    h=(times-times.normalize()).total_seconds()/3600
    return raw(offset+amplitude*np.cos(2*np.pi*(h-14)/24),times)


class AdmissionTests(unittest.TestCase):
    def clean(self, frame): return ch.clean_admission_measurements(frame,{'D1':994},EMPTY)

    def test_duplicate_decisions_preserve_source(self):
        source=raw([400,400,500,501,600],pd.to_datetime(['2019-01-01']*2+['2019-01-01 01:00']*2+['2019-01-01 02:00'],format='mixed'))
        before=source.copy(deep=True);work=self.clean(source)
        self.assertEqual(len(work),3)
        self.assertEqual(work.exact_duplicate_extra.tolist(),[1,0,0])
        self.assertTrue(np.isnan(work.datavalue.iloc[1]))
        self.assertEqual(work.exclusion_reason.iloc[1],'conflicting_timestamp')
        self.assertEqual(work.datavalue.iloc[2],600)
        pd.testing.assert_frame_equal(source,before)

    def test_band_and_service_boundaries(self):
        work=self.clean(raw([-10000,-9999,-1,0,.1,3000,3000.1,np.nan]))
        self.assertEqual(work.exclusion_reason.iloc[:2].tolist(),['service_code']*2)
        np.testing.assert_allclose(work.datavalue.iloc[4:6],[.1,3000])
        self.assertEqual(work.datavalue.notna().sum(),2)

    def test_one_spike_across_cadences(self):
        for freq in ['1min','5min','15min']:
            with self.subTest(freq=freq):
                times=pd.date_range('2019-01-01',periods=7,freq=freq)
                work=self.clean(raw([500,501,499,700,501,500,499],times))
                self.assertEqual(work.exclusion_reason.iloc[3],'isolated_7_sample_MAD_spike')
                self.assertEqual(work.datavalue.notna().sum(),6)

    def test_gap_or_cadence_boundary_is_not_crossed(self):
        for minutes in [[0,1,2,50,51,52,53],[0,1,2,3,8,13,18]]:
            work=self.clean(raw([500,501,499,700,501,500,499],pd.Timestamp('2019-01-01')+pd.to_timedelta(minutes,unit='min')))
            self.assertEqual(work.datavalue.iloc[3],700)
            self.assertFalse(work.despike_window_eligible.any())

    def test_invalid_record_is_not_jumped(self):
        work=self.clean(raw([500,501,-10000,700,501,500,499]))
        self.assertEqual(work.datavalue.iloc[3],700)
        self.assertFalse(work.despike_window_eligible.any())

    def test_zero_mad_and_multiple_peak_values_are_retained(self):
        work=self.clean(raw([500,500,500,700,500,500,500]))
        self.assertEqual(work.datavalue.iloc[3],700)
        self.assertEqual(work.doubt_reason.iloc[3],'zero_MAD_deviation')
        work=self.clean(raw([500,501,499,700,701,500,499]))
        self.assertEqual(work.datavalue.iloc[3],700)
        work=self.clean(raw([500]*100))
        self.assertFalse(work.exclusion_reason.ne('').any())
        self.assertFalse(work.doubt_reason.ne('').any())
        self.assertTrue(work.constant_run_hours.ge(2).all())

    def test_smooth_diurnal_maximum_not_removed(self):
        for freq in ['1min','5min','15min']:
            frame=harmonic(days=2,freq=freq)
            work=self.clean(frame)
            np.testing.assert_array_equal(work.datavalue,frame.datavalue)
            self.assertFalse(work.doubt_reason.ne('').any())

    def test_individual_exclusion_not_extended_to_other_dates(self):
        times=pd.to_datetime(['2017-11-12 07:00','2018-11-12 07:00'])
        e=pd.DataFrame([dict(level='D1',start=ch.STUCK_INTERVALS[0][0],end=ch.STUCK_INTERVALS[0][1])])
        work=ch.clean_admission_measurements(raw([500,500],times),{'D1':994},e)
        self.assertTrue(np.isnan(work.datavalue.iloc[0]))
        self.assertEqual(work.datavalue.iloc[1],500)

    def test_monthly_day_requires_bins_not_context_or_continuity(self):
        times=pd.Timestamp('2019-01-01')+pd.to_timedelta([0,1,6,7,12,13,18,19],unit='h')
        frame=raw(500+20*np.cos(2*np.pi*np.array([0,1,6,7,12,13,18,19])/24),times)
        fit=ch.monthly_day_fit(frame)
        self.assertAlmostEqual(fit['amplitude'],20)
        self.assertIsNone(ch.monthly_day_fit(frame.iloc[:-1]))
        self.assertIsNone(ch.monthly_day_fit(pd.concat([frame.iloc[:7],frame.iloc[[6]]])))

    def test_monthly_raw_amplitude_and_equal_day_weight(self):
        a=harmonic(days=10,amplitude=20,offset=500)
        b=harmonic('2019-01-11',days=1,amplitude=20,offset=1000,freq='1min')
        work=self.clean(pd.concat([a,b],ignore_index=True))
        months,days=ch.monthly_admission(work,{'D1':994},'2019-01-01','2019-02-01')
        self.assertEqual(months.status.iloc[0],'pass')
        self.assertAlmostEqual(months.median_concentration.iloc[0],500)
        self.assertAlmostEqual(months.median_raw_amplitude.iloc[0],20)
        self.assertEqual(months.valid_days.iloc[0],11)

    def test_monthly_amplitude_has_no_rolling_mean_subtraction(self):
        times=pd.date_range('2019-01-01',periods=96,freq='15min')
        fit=ch.monthly_day_fit(raw(500+2.5*np.arange(96),times))
        # Exact first discrete Fourier amplitude of a linear sequence.
        self.assertAlmostEqual(fit['amplitude'],2.5/np.sin(np.pi/96),places=9)

    def test_small_r_squared_is_not_a_monthly_exclusion(self):
        frame=harmonic(days=12)
        h=(frame.localdatetime-frame.localdatetime.dt.normalize()).dt.total_seconds()/3600
        frame.datavalue+=300*np.cos(2*np.pi*h/2)
        frame['doubt_reason']=''
        months,days=ch.monthly_admission(frame,{'D1':994},'2019-01-01','2019-02-01')
        self.assertTrue(days.r_squared.lt(.01).all())
        self.assertEqual(months.status.iloc[0],'pass')

    def test_months_with_no_rows_and_boundary_days(self):
        work=self.clean(harmonic('2019-01-25',days=14))
        months,_=ch.monthly_admission(work,{'D1':994},'2019-01-01','2019-04-01')
        self.assertEqual(months.month.tolist(),['2019-01','2019-02','2019-03'])
        self.assertEqual(months.valid_days.tolist(),[7,7,0])
        self.assertEqual(months.status.tolist(),['fail']*3)

    def test_thresholds_are_inclusive(self):
        for amplitude,offset,status in [(9,100,'pass'),(9,2000,'pass'),(7,500,'fail'),(20,99,'fail'),(20,2001,'fail')]:
            with self.subTest(amplitude=amplitude,offset=offset):
                work=self.clean(harmonic(amplitude=amplitude,offset=offset))
                months,_=ch.monthly_admission(work,{'D1':994},'2019-01-01','2019-02-01')
                self.assertEqual(months.status.iloc[0],status)

    def test_uncertainty_retained_only_when_gate_not_invariant(self):
        for amplitude,expected in [(20,'pass'),(7.99,'undetermined')]:
            work=self.clean(harmonic(amplitude=amplitude))
            # One explicitly unresolved datum per day. Keep/drop combinations
            # span the amplitude threshold only in the second example.
            mask=work.localdatetime.dt.hour.eq(14)&work.localdatetime.dt.minute.eq(0)
            work.loc[mask,'datavalue']+=3
            work.loc[mask,'doubt_reason']='ambiguous_excursion'
            months,_=ch.monthly_admission(work,{'D1':994},'2019-01-01','2019-02-01')
            self.assertEqual(months.status.iloc[0],expected)

    def test_final_context_and_all_three_monthly_pass(self):
        frame=harmonic(days=12)
        source=pd.concat([frame.assign(sensorid=sid) for sid in SENSORS.values()],ignore_index=True)
        work=ch.clean_admission_measurements(source,SENSORS,EMPTY)
        monthly,_=ch.monthly_admission(work,SENSORS,'2019-01-01','2019-01-13')
        cal=ch.final_admission_calendar(work,SENSORS,monthly,'2019-01-01','2019-01-13',EMPTY)
        self.assertTrue(monthly.status.eq('pass').all())
        self.assertEqual(cal.category.tolist(),['B']+['A']*10+['B'])
        table,fits=ch.calculate_trajectories(work,SENSORS,cal,EMPTY)
        self.assertEqual(table.calculation_status.eq('computed').sum(),10)
        np.testing.assert_allclose(fits.amplitude_ppm,20,atol=1e-7,rtol=1e-7)
        # Monthly failure is separate from temporal availability.
        monthly.loc[monthly.level.eq('D3'),'status']='fail'
        cal=ch.final_admission_calendar(work,SENSORS,monthly,'2019-01-01','2019-01-13',EMPTY)
        self.assertFalse(cal.category.eq('A').any())
        self.assertEqual(cal.technical_category.eq('A').sum(),10)

    def test_unresolved_context_uses_exact_windows_including_midnight_support(self):
        frame=harmonic(days=12)
        source=pd.concat([frame.assign(sensorid=sid) for sid in SENSORS.values()],ignore_index=True)
        work=ch.clean_admission_measurements(source,SENSORS,EMPTY)
        monthly,_=ch.monthly_admission(work,SENSORS,'2019-01-01','2019-01-13')
        # Jan 3 11:45 is outside the last Jan 2 fit window, but needed by
        # the Jan 3 midnight observation establishing complete Jan 2 coverage.
        flag=work.localdatetime.eq('2019-01-03 11:45') & work.level.eq('D1')
        work.loc[flag,'doubt_reason']='ambiguous_excursion'
        before=work.copy(deep=True)
        cal=ch.final_admission_calendar(work,SENSORS,monthly,'2019-01-01','2019-01-13',EMPTY).set_index('date')
        self.assertTrue(cal.loc['2019-01-02','unresolved_context'])
        self.assertEqual(cal.loc['2019-01-02','category'],'C')
        self.assertEqual(cal.loc['2019-01-05','category'],'A')
        pd.testing.assert_frame_equal(work,before)

    def test_final_plot_excludes_failed_and_undetermined_months(self):
        t=pd.DataFrame(dict(date=pd.date_range('2014-01-01',periods=3),calculation_status=['computed']*3,
                            admission_status=['pass','fail','undetermined']))
        for l in SENSORS:
            t[f'H_{l}']=[1.,2.,3.];t[f'A_{l}']=[20.,20.,20.]
            t[f'R2_{l}']=.001;t[f'phase_status_{l}']='not_assessed'
        for kind in ['H','A']:
            fig=ch.interactive_trajectory(t,kind,final=True)
            self.assertEqual(sum(len(trace.x) for trace in fig.data if trace.mode=='markers'),1)
            self.assertFalse(any(trace.mode=='lines' for trace in fig.data))
            self.assertIn('01.01.2014',fig.layout.title.text)

    def test_no_admitted_days_has_explicit_empty_views(self):
        t=pd.DataFrame(dict(date=pd.to_datetime(['2019-01-01']),calculation_status=['skipped'],admission_status=['fail']))
        for kind in ['H','A']:
            fig=ch.interactive_trajectory(t,kind,final=True)
            self.assertEqual(len(fig.data),0)
            self.assertIn('нет определённых',fig.layout.title.text)
            ch.plt.close(ch.plot_trajectory(t,kind,final=True))

    def test_enclosure_contains_actual_removal_subsets(self):
        frame=harmonic(days=1)
        frame['doubt_reason']=''
        frame.loc[::7,'doubt_reason']='ambiguous_excursion'
        frame.loc[::7,'datavalue']+=np.arange(len(frame.loc[::7]))
        fit=ch.monthly_day_fit(frame)
        bounds=ch.admission_day_bounds(frame,fit)
        indices=np.flatnonzero(frame.doubt_reason.ne(''))
        rng=np.random.default_rng(7401)
        for _ in range(40):
            keep=np.ones(len(frame),bool)
            keep[indices[rng.integers(0,2,len(indices)).astype(bool)]]=False
            actual=ch.monthly_day_fit(frame.loc[keep])
            self.assertIsNotNone(actual)
            self.assertLessEqual(bounds['a_lo'],actual['amplitude'])
            self.assertGreaterEqual(bounds['a_hi'],actual['amplitude'])
            self.assertLessEqual(bounds['c_lo'],actual['concentration'])
            self.assertGreaterEqual(bounds['c_hi'],actual['concentration'])

if __name__=='__main__': unittest.main()

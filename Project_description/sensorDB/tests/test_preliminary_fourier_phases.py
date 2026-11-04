"""Independent synthetic/unit/browser-engine tests. No database/archive needed."""
import cmath
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preliminary_fourier_phases import (
    IDS, PAIRS, aggregate_bins, broken_line, compute_windows, direct_fourier,
    html_payload, longest_missing_run, phase_difference, validation_examples,
    window_slices, wrap, complex_step, select_phase_jump, phase_jump_tables,
    GRID_START, SUBSURFACE_IDS, resolution_grid, resolution_windows,
    compare_resolution_windows, resolution_summary, resolution_extrema,
)


class FourierPhaseTests(unittest.TestCase):
    def setUp(self):
        self.t = (np.arange(360)+.5)/12
        self.u = np.cos(2*np.pi*self.t)
        self.v = np.cos(2*np.pi*(self.t-4/24))

    def delta(self, t, a, b):
        f = direct_fourier(t, np.column_stack([a,b]))
        return phase_difference(*f)

    def test_A_zero_phase(self):
        self.assertAlmostEqual(self.delta(self.t,self.u,self.u),0,places=11)

    def test_B_positive_four_hour_lag(self):
        self.assertAlmostEqual(self.delta(self.t,self.u,self.v),np.pi/3,delta=1e-10*np.pi/180)

    def test_C_reverse_negative_lag(self):
        self.assertAlmostEqual(self.delta(self.t,self.v,self.u),-np.pi/3,delta=1e-10*np.pi/180)

    def test_D_wrap_boundary(self):
        delayed = np.cos(2*np.pi*(self.t-14/24))
        self.assertAlmostEqual(self.delta(self.t,self.u,delayed),-5*np.pi/6,delta=1e-10*np.pi/180)
        self.assertEqual(wrap(np.pi),-np.pi)
        self.assertEqual(wrap(-np.pi),-np.pi)
        self.assertAlmostEqual(wrap(np.pi+np.pi/180),-np.pi+np.pi/180,delta=1e-11*np.pi/180)

    def test_E_common_missing_bins(self):
        # Entire days preserve quadrature balance, and recover the lag exactly.
        mask = ~((self.t>=10)&(self.t<13))
        self.assertAlmostEqual(self.delta(self.t[mask],self.u[mask],self.v[mask]),np.pi/3,delta=1e-10*np.pi/180)
        # An independent, fixed asymmetric mask produces finite sampling bias.
        mask = np.ones(360,dtype=bool)
        mask[[7,8,9,27,48,121,122,203,278,301,339]] = False
        t, u, v = self.t[mask], self.u[mask], self.v[mask]
        def reference(values):
            mean = math.fsum(float(y) for y in values)/len(values)
            return sum((float(y)-mean)*cmath.exp(-2j*math.pi*float(time))
                       for time,y in zip(t,values))/len(values)
        expected = cmath.phase(reference(u)*reference(v).conjugate())
        actual = self.delta(t,u,v)
        self.assertAlmostEqual(actual,expected,delta=1e-10*np.pi/180)
        self.assertLess(abs(actual-np.pi/3),np.pi/180)  # test-specific bound; NEVER a data admission rule
        self.assertGreater(abs(actual-np.pi/3),np.pi/18000)  # explicitly disprove exact gap invariance
        compressed = self.delta((np.arange(len(u))+.5)/12,u,v)
        self.assertGreater(abs(compressed-actual),np.pi/1800)

    def test_F_fft_cross_check_full_grid(self):
        x = np.column_stack([self.u,self.v])
        expected = np.fft.fft(x-np.mean(x,axis=0),axis=0)[30]/360
        # numpy FFT takes first sample as zero; actual bin centers start at 1h.
        expected *= np.exp(-2j*np.pi*self.t[0])
        np.testing.assert_allclose(direct_fourier(self.t,x),expected,atol=2e-14,rtol=0)

    def test_shared_origin_shift_and_constant_offsets(self):
        x = np.column_stack([self.u+400,self.v+700])
        f = direct_fourier(self.t,x)
        shifted = direct_fourier(self.t+17.37,x)
        np.testing.assert_allclose(shifted,f*np.exp(-2j*np.pi*17.37),atol=3e-14,rtol=0)
        self.assertAlmostEqual(phase_difference(*f),phase_difference(*shifted),places=11)

    def test_common_bins_means_counts_flags_and_no_fill(self):
        start,end = pd.Timestamp('2025-04-29'),pd.Timestamp('2025-04-30')
        frames = {}
        for j,s in enumerate(IDS):
            # One excluded spike, two retained in bin 0, one in bin 1.
            frames[s] = pd.DataFrame(dict(localdatetime=pd.to_datetime([
                '2025-04-29 00:00','2025-04-29 01:59','2025-04-29 01:30','2025-04-29 02:00']),
                datavalue=[1+j,5+j,9999,10+j],technically_retained=[True,True,False,True],
                doubtful=[False,True,False,False],plateau_review=[False]*4))
        frames[IDS[-1]] = frames[IDS[-1]].iloc[:3]
        grid = aggregate_bins(frames,start,end)
        self.assertEqual(len(grid),12)
        self.assertEqual(grid.complete.sum(),1)
        self.assertEqual(grid.iloc[0][f'mean_{IDS[0]}'],3)
        self.assertEqual(grid.iloc[0][f'n_{IDS[0]}'],2)
        self.assertTrue(grid.iloc[0].any_review)
        self.assertTrue(np.isnan(grid.iloc[2][f'mean_{IDS[0]}']))
        self.assertAlmostEqual(grid.iloc[0].t_days,1/24)
        self.assertEqual(grid.iloc[0].bin_center,pd.Timestamp('2025-04-29 01:00'))
        p,c = compute_windows(grid,1,1)
        self.assertEqual(c.iloc[0].complete_bins,1)
        self.assertEqual(c.iloc[0].longest_gap_hours,22)
        self.assertEqual(p.iloc[0].status,'INSUFFICIENT DATA')

    def synthetic_grid(self, days=90):
        t = (np.arange(days*12)+.5)/12
        start = pd.Timestamp('2025-04-29')
        frames = {}
        for j,s in enumerate(IDS):
            # Modulation gives different window results: catches a static HTML.
            values=450+j*20+10*np.cos(2*np.pi*t-j*.7)+3*np.cos(2*np.pi*.93*t+j)
            keep=np.ones(len(t),bool)
            keep[13+j::71+j]=False  # different sensor gaps -> intersection required
            frames[s]=pd.DataFrame(dict(localdatetime=start+pd.to_timedelta(t[keep],unit='D'),
                datavalue=values[keep],technically_retained=True,doubtful=(np.arange(keep.sum())%17==0),plateau_review=False))
        return aggregate_bins(frames,start,start+pd.Timedelta(days=days))

    def test_window_parameters_and_common_timestamps(self):
        grid=self.synthetic_grid()
        p,c=compute_windows(grid,30,1)
        window=grid.iloc[:360]; common=window.loc[window.complete]
        f=direct_fourier(common.t_days.to_numpy(),common[[f'mean_{s}' for s in IDS]].to_numpy())
        for j,s in enumerate(IDS):
            self.assertAlmostEqual(p.iloc[0][f'F_real_{s}'],f[j].real,places=12)
        self.assertEqual(len(window_slices(521*12,30,1)),492)
        self.assertEqual(len(window_slices(521*12,60,7)),66)
        self.assertLess(c.iloc[0].complete_bins,360)
        self.assertEqual(len(p),61)
        with self.assertRaises(ValueError): window_slices(12,2,1)
        with self.assertRaises(ValueError): window_slices(12,1,0)
        with self.assertRaises(ValueError): window_slices(12,1.5,1)

    def browser(self,payload,window,step):
        script = Path(__file__).resolve().parents[1]/'preliminary_fourier_phases.js'
        js = 'const fs=require("fs");const e=require('+json.dumps(str(script))+');const p=JSON.parse(fs.readFileSync(0,"utf8"));process.stdout.write(JSON.stringify(e.calculatePhases(p,'+str(window)+','+str(step)+')));'
        return json.loads(subprocess.check_output(['node','-e',js],input=json.dumps(payload),text=True))

    def test_javascript_recalculates_all_windows_like_python(self):
        grid=self.synthetic_grid()
        payload=html_payload(grid,{})
        for width,step in [(1,1),(7,2),(30,1),(60,7)]:
            actual=self.browser(payload,width,step)
            p,c=compute_windows(grid,width,step)
            self.assertEqual(len(actual),len(p))
            np.testing.assert_allclose([r['phase'] for r in actual],p[[x+'_rad' for x in PAIRS]],atol=1e-9*np.pi/180,rtol=0)
            self.assertEqual([r['n'] for r in actual],c.complete_bins.tolist())
            self.assertEqual([r['longestGapHours'] for r in actual],c.longest_gap_hours.tolist())
            self.assertEqual([r['doubtful'] for r in actual],c.doubtful_bins.tolist())
        self.assertGreater(abs(actual[0]['phase'][0]-self.browser(payload,30,1)[0]['phase'][0]),.01*np.pi/180)

    def test_undefined_empty_singleton_constant(self):
        for count in [0,1]:
            with self.assertRaisesRegex(ValueError,'INSUFFICIENT DATA'):
                direct_fourier(np.arange(count),np.ones((count,4)))
        f=direct_fourier(self.t,np.full((len(self.t),2),500))
        self.assertTrue(np.isnan(phase_difference(*f)))
        payload=dict(start='2025-04-29',t=(np.arange(12)+.5).tolist(),
                     y=[[None]*12 for _ in IDS],q=[[0]*12 for _ in IDS],n=[[0]*12 for _ in IDS])
        result=self.browser(payload,1,1)[0]
        self.assertEqual(result['phase'],[None]*3)
        self.assertEqual(result['longestGapHours'],24)
        payload['y']=[[500]*12 for _ in IDS]
        result=self.browser(payload,1,1)[0]
        self.assertEqual(result['phase'],[None]*3)
        self.assertIn('нулевой коэффициент',result['reason'])

    def test_wrap_plot_break_preserves_endpoints_and_gaps(self):
        x,y=broken_line([0,1,2,3,4],[np.pi-.01,-np.pi+.01,np.nan,2,2.1])
        np.testing.assert_allclose(y,[np.pi-.01,np.nan,-np.pi+.01,np.nan,2,2.1],equal_nan=True)
        self.assertEqual(x,[0,1,1,2,3,4])
        self.assertEqual(longest_missing_run([False,False,True,False]),2)

    def test_validation_table_has_all_six_required_checks(self):
        table=validation_examples()
        self.assertEqual([r['test'] for r in table],['A','B','C','D','E_whole_days','E_irregular','F'])
        self.assertLess(table[-1]['max_complex_error'],2e-14)

    def test_jump_branch_crossing_without_small_modulus(self):
        result = complex_step(cmath.exp(1j*(math.pi-.01)), cmath.exp(1j*(-math.pi+.01)))
        self.assertAlmostEqual(result['raw_increment_rad'], -2*math.pi+.02)
        self.assertAlmostEqual(result['circular_increment_rad'], .02)
        self.assertEqual(result['branch_turns'], -1)
        self.assertAlmostEqual(result['abs_before'], 1)
        self.assertAlmostEqual(result['abs_after'], 1)

    def test_jump_argument_change_and_conjugate_sign(self):
        u0, u1 = 2+3j, 1+2j
        v0, v1 = 2+1j, 2+.01j  # small imaginary part is not a small modulus
        du, dv, dq = [complex_step(a, b) for a,b in
                      [(u0,u1), (v0,v1), (u0*v0.conjugate(),u1*v1.conjugate())]]
        expected = cmath.phase(u1)-cmath.phase(u0)-cmath.phase(v1)+cmath.phase(v0)
        self.assertAlmostEqual(dq['circular_increment_rad'], expected)
        self.assertAlmostEqual(dq['circular_increment_rad'],
                               du['circular_increment_rad']-dv['circular_increment_rad'])
        self.assertEqual(dq['branch_turns'], 0)
        self.assertGreater(dv['abs_after'], 2)

    def test_jump_small_nonzero_coefficient_sensitivity(self):
        # Same absolute perturbation at two scales: no real-data cutoff implied.
        epsilon = 1e-8
        small = complex_step(epsilon+0j, epsilon+1j*epsilon)
        large = complex_step(1+0j, 1+1j*epsilon)
        self.assertTrue(small['phase_defined'])
        self.assertAlmostEqual(small['circular_increment_rad'], math.pi/4)
        self.assertAlmostEqual(large['circular_increment_rad'], math.atan(epsilon))
        self.assertAlmostEqual(small['relative_complex_change'], 1)
        self.assertAlmostEqual(large['relative_complex_change'], epsilon)

    def test_jump_zero_and_nonfinite_endpoints_are_undefined(self):
        for a,b in [(0j,1j), (1j,0j), (complex(np.nan,0),1j)]:
            result = complex_step(a,b)
            self.assertFalse(result['phase_defined'])
            self.assertTrue(np.isnan(result['circular_increment_rad']))
            self.assertIsNone(result['branch_turns'])
        # Opposite endpoints alone do not establish an intervening zero.
        result = complex_step(1+0j, -1+0j)
        self.assertTrue(result['phase_defined'])
        self.assertAlmostEqual(result['circular_increment_rad'], -math.pi)

    def test_jump_selection_ties_gaps_and_undefined_values(self):
        table = pd.DataFrame(dict(center=pd.to_datetime([
            '2025-01-01','2025-01-02','2025-01-03','2025-01-05','2025-01-06','2025-01-07']),
            **{'5_to_20_rad':[0, .5, 0, 3, np.nan, -3]}))
        self.assertEqual(select_phase_jump(table), (0,1))  # earliest exact tie; gap excluded
        with self.assertRaises(ValueError): select_phase_jump(table.iloc[::-1])
        with self.assertRaises(ValueError): select_phase_jump(table.iloc[[3,5]])

    def test_jump_tables_preserve_inputs_and_align_coverage(self):
        phases, coverage = compute_windows(self.synthetic_grid(), 30, 1)
        original = phases.copy(deep=True)
        local, quality, steps = phase_jump_tables(phases, coverage, neighbors=2)
        pd.testing.assert_frame_equal(phases, original)
        self.assertEqual(local.selected_endpoint.sum(), 2)
        self.assertEqual(local.window.tolist(), quality.window.tolist())
        for _,row in local.iterrows():
            u = complex(row.F_real_996, row.F_imag_996)
            v = complex(row.F_real_1012, row.F_imag_1012)
            expected = u*v.conjugate()
            self.assertAlmostEqual(row.Q_real, expected.real)
            self.assertAlmostEqual(row.Q_imag, expected.imag)
            self.assertAlmostEqual(row.Q_abs, abs(expected))
            self.assertAlmostEqual(row.Q_arg_rad, cmath.phase(expected))
        self.assertEqual(steps.coefficient.tolist(), ['F5','F20','Q'])
        with self.assertRaises(AssertionError): phase_jump_tables(phases, coverage.iloc[::-1])
        with self.assertRaises(ValueError): phase_jump_tables(phases, coverage.assign(window_days=60))

    def resolution_frames(self, days=30):
        times = pd.date_range(GRID_START+pd.Timedelta(minutes=7.5), periods=days*96, freq='15min')
        t = np.arange(days*96)/96 + 1/192
        frames = {}
        for j,sid in enumerate(SUBSURFACE_IDS):
            frames[sid] = pd.DataFrame(dict(localdatetime=times,
                datavalue=500+(4+2*j)*np.cos(2*np.pi*t-j*np.pi/3),
                technically_retained=True,doubtful=False,plateau_review=False))
        return frames

    def test_resolution_known_phase_sign_and_regular_aggregation(self):
        frames=self.resolution_frames()
        results=[]
        for minutes in (15,120):
            grid=resolution_grid(frames,minutes,GRID_START,GRID_START+pd.Timedelta(days=30))
            table=resolution_windows(grid,minutes)
            results.append(table)
            self.assertEqual(table.complete_bins.iloc[0],30*1440//minutes)
            for pair in PAIRS[1:]:
                self.assertAlmostEqual(table[pair+'_rad'].iloc[0],math.pi/3,places=11)
            a=complex(table.F_real_996.iloc[0],table.F_imag_996.iloc[0])
            b=complex(table.F_real_1012.iloc[0],table.F_imag_1012.iloc[0])
            self.assertAlmostEqual(phase_difference(b,a),-math.pi/3,places=11)
        attenuation=math.sin(math.pi/12)/(8*math.sin(math.pi/96))
        self.assertAlmostEqual(results[0].F_abs_996.iloc[0],2,places=11)
        self.assertAlmostEqual(results[1].F_abs_996.iloc[0],2*attenuation,places=11)
        comparison=compare_resolution_windows(*results,'aggregation')
        np.testing.assert_allclose(comparison.delta_rad,0,atol=1e-12,rtol=0)

    def test_resolution_circular_difference_across_branch(self):
        frame=resolution_windows(resolution_grid(self.resolution_frames(),15,GRID_START,
            GRID_START+pd.Timedelta(days=30)),15)
        left,right=frame.copy(),frame.copy()
        left['5_to_20_rad']=np.pi-.01
        right['5_to_20_rad']=-np.pi+.01
        right['20_to_50_rad']=np.nan
        comparison=compare_resolution_windows(left,right,'aggregation')
        self.assertAlmostEqual(comparison.iloc[0].delta_rad,-.02)
        self.assertFalse(comparison.iloc[1].comparable)
        self.assertTrue(np.isnan(comparison.iloc[1].delta_rad))
        summary=resolution_summary(comparison)
        self.assertEqual(summary.undefined_windows.tolist(),[0,1])
        with self.assertRaises(AssertionError):
            compare_resolution_windows(left,right.assign(center=right.center+pd.Timedelta(days=1)),'aggregation')

    def test_resolution_missing_cells_use_actual_centers(self):
        frames=self.resolution_frames()
        frames[996]=frames[996].drop([0,1,200])
        frames[1012]=frames[1012].drop([2,201])
        frames[1035]=frames[1035].drop([3])
        original=frames[996].copy(deep=True)
        grid=resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=30))
        self.assertFalse(grid.complete_subsurface.iloc[:4].any())
        self.assertTrue(np.isnan(grid.mean_996.iloc[0]))
        self.assertEqual(grid.n_996.iloc[0],0)
        result=resolution_windows(grid,15).iloc[0]
        self.assertEqual(result.missing_bins,6)
        self.assertEqual(result.longest_gap_hours,1)
        used=grid.loc[grid.complete_subsurface]
        z=[]
        for sid in SUBSURFACE_IDS:
            values=used[f'mean_{sid}'].to_numpy()
            mean=math.fsum(values)/len(values)
            z.append(sum((float(x)-mean)*cmath.exp(-2j*math.pi*float(t))
                         for t,x in zip(used.t_days,values))/len(values))
        self.assertAlmostEqual(result['5_to_20_rad'],cmath.phase(z[0]*z[1].conjugate()),places=11)
        pd.testing.assert_frame_equal(frames[996],original)

    def test_resolution_raw_observation_weighting_and_accepted_selector(self):
        times=pd.to_datetime(['2025-05-01 00:00','2025-05-01 00:05','2025-05-01 00:15',
                              '2025-05-01 00:30','2025-05-01 00:45','2025-05-01 02:00'])
        frames={s:pd.DataFrame(dict(localdatetime=times,datavalue=[1,3,8,900,800,10],
                technically_retained=[True,True,True,False,True,True],
                doubtful=[False,False,False,False,True,False],plateau_review=False)) for s in SUBSURFACE_IDS}
        fine=resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=1))
        coarse=resolution_grid(frames,120,GRID_START,GRID_START+pd.Timedelta(days=1))
        self.assertEqual(fine.mean_996.iloc[0],2)
        self.assertEqual(fine.mean_996.iloc[1],8)
        self.assertEqual(coarse.mean_996.iloc[0],4)  # (1+3+8)/3, NOT mean(2,8)
        self.assertTrue(np.isnan(coarse.mean_996.iloc[2]))
        self.assertEqual(coarse.mean_996.iloc[1],10)  # boundary belongs to next cell
        self.assertEqual(coarse.bin_center.iloc[0],GRID_START+pd.Timedelta(hours=1))
        self.assertEqual(fine.bin_center.iloc[0],GRID_START+pd.Timedelta(minutes=7.5))

    def test_resolution_air_restriction_is_separate_support_only(self):
        frames=self.resolution_frames()
        times=pd.date_range(GRID_START,periods=360,freq='2h')
        frames[1275]=pd.DataFrame(dict(localdatetime=times[12:],datavalue=400,
            technically_retained=True,doubtful=False,plateau_review=False))
        grid=resolution_grid(frames,120,GRID_START,GRID_START+pd.Timedelta(days=30))
        full=resolution_windows(grid,120)
        restricted=resolution_windows(grid,120,'with_air')
        self.assertEqual(full.complete_bins.iloc[0],360)
        self.assertEqual(restricted.complete_bins.iloc[0],348)
        self.assertEqual(restricted.air_removed_bins.iloc[0],12)
        self.assertEqual(restricted.longest_gap_hours.iloc[0],24)
        self.assertNotIn('F_abs_1275',restricted)
        compare=compare_resolution_windows(full,restricted,'air_support')
        np.testing.assert_allclose(compare.delta_rad,0,atol=1e-12,rtol=0)
        with self.assertRaises(ValueError):resolution_windows(grid,15,'with_air')

    def test_resolution_undefined_and_tiny_nonzero_without_new_cutoff(self):
        frames=self.resolution_frames()
        frames[1012]['datavalue']=500.
        table=resolution_windows(resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=30)),15)
        self.assertTrue(table[['5_to_20_rad','20_to_50_rad']].isna().all().all())
        self.assertEqual(table.F_abs_1012.iloc[0],0)
        frames=self.resolution_frames()
        for s in SUBSURFACE_IDS:frames[s]['datavalue']=(frames[s].datavalue-500)*1e-12
        table=resolution_windows(resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=30)),15)
        self.assertAlmostEqual(table['5_to_20_rad'].iloc[0],np.pi/3,places=11)
        for s in SUBSURFACE_IDS:frames[s]=frames[s].iloc[:1]
        table=resolution_windows(resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=30)),15)
        self.assertEqual(table['5_to_20_reason'].iloc[0],'fewer_than_two_common_cells')
        self.assertTrue(np.isnan(table['5_to_20_rad'].iloc[0]))

    def test_resolution_calendar_step_and_descriptive_extrema(self):
        frames=self.resolution_frames(32)
        fine=resolution_grid(frames,15,GRID_START,GRID_START+pd.Timedelta(days=32))
        table=resolution_windows(fine,15)
        self.assertEqual(len(table),3)
        self.assertEqual(table.start.iloc[-1],GRID_START+pd.Timedelta(days=2))
        self.assertEqual(table.end_exclusive.iloc[-1],GRID_START+pd.Timedelta(days=32))
        extrema=resolution_extrema(table.assign(variant='15min_subsurface'))
        self.assertEqual(set(extrema.metric),{f'F_abs_{s}' for s in SUBSURFACE_IDS})
        with self.assertRaises(ValueError):resolution_grid(frames,60)



class IntrabinTests(unittest.TestCase):
    """Concentration-only cases; no Fourier evaluation, archive or database."""
    def calculate(self, values, minutes=None, hours=2):
        from preliminary_fourier_phases import intrabin_statistics
        start=pd.Timestamp('2025-05-01')
        if minutes is None: minutes=np.arange(len(values))*15
        f=pd.DataFrame(dict(localdatetime=[start+pd.Timedelta(minutes=float(v)) for v in minutes],
            datavalue=values,technically_retained=True,doubtful=False,plateau_review=False))
        return intrabin_statistics({996:f},start,start+pd.Timedelta(hours=hours))

    def test_constant(self):
        c,a=self.calculate([400]*8);r=c.iloc[0]
        self.assertEqual(r.mean_ppm,400)
        for key in ['range_ppm','max_abs_deviation_ppm','last_minus_first_ppm','max_adjacent_abs_change_ppm','max_change_rate_ppm_h']:
            self.assertEqual(r[key],0)
        self.assertEqual(len(a),7);self.assertTrue(r.complete_15min_coverage)

    def test_linear(self):
        c,a=self.calculate([400+2*i for i in range(8)]);r=c.iloc[0]
        self.assertEqual(r.mean_ppm,407);self.assertEqual(r.range_ppm,14)
        self.assertEqual(r.max_abs_deviation_ppm,7);self.assertEqual(r.last_minus_first_ppm,14)
        self.assertEqual(r.max_change_rate_ppm_h,8)
        self.assertTrue(a.dt_seconds.eq(900).all())

    def test_single_peak(self):
        c,a=self.calculate([400,400,400,480,400,400,400,400]);r=c.iloc[0]
        self.assertEqual(r.mean_ppm,410);self.assertEqual(r.range_ppm,80)
        self.assertEqual(r.max_abs_deviation_ppm,70);self.assertEqual(r.last_minus_first_ppm,0)
        self.assertEqual(r.max_change_from,pd.Timestamp('2025-05-01 00:30'))

    def test_missing_and_insufficient(self):
        c,a=self.calculate([500],[15],hours=4)
        self.assertEqual(c.n.tolist(),[1,0]);self.assertEqual(len(a),0)
        self.assertEqual(c.iloc[0].mean_ppm,500);self.assertTrue(np.isnan(c.iloc[1].mean_ppm))
        for key in ['range_ppm','max_abs_deviation_ppm','last_minus_first_ppm','max_adjacent_abs_change_ppm','max_change_dt_seconds','max_change_rate_ppm_h']:
            self.assertTrue(c[key].isna().all())
        self.assertEqual(c.missing_15min_cells.tolist(),[7,8])
        self.assertTrue(c.insufficient_variation.all())

    def test_incomplete_gap(self):
        c,a=self.calculate([400,410,430,440],[0,15,60,105]);r=c.iloc[0]
        self.assertEqual(r.mean_ppm,420);self.assertEqual(r.range_ppm,40)
        self.assertEqual(r.occupied_15min_cells,4);self.assertEqual(r.longest_missing_15min_cells,2)
        self.assertEqual(r.max_change_dt_seconds,2700)
        self.assertAlmostEqual(r.max_change_rate_ppm_h,20/.75)
        self.assertEqual(a.missing_15min_cells_between.tolist(),[0,2,2])

    def test_half_open_boundaries(self):
        c,a=self.calculate([1,2,3,4,5],[0,119.9,120,239.9,240],hours=4)
        self.assertEqual(c.n.tolist(),[2,2]);self.assertEqual(c.mean_ppm.tolist(),[1.5,3.5])
        self.assertEqual(len(a),2)
        self.assertEqual(c.iloc[0].bin_end_exclusive,c.iloc[1].bin_start)

    def test_irregular_timestamps(self):
        c,a=self.calculate([100,110,106],[1/60,15+2/60,45+4/60]);r=c.iloc[0]
        self.assertEqual(a.dt_seconds.tolist(),[901,1802])
        self.assertAlmostEqual(r.mean_ppm,316/3)
        self.assertEqual(r.max_change_dt_seconds,901)
        self.assertAlmostEqual(r.max_change_rate_ppm_h,36000/901)
        self.assertEqual(r.first_time,pd.Timestamp('2025-05-01 00:00:01'))

    def test_actual_points_not_means_of_slots(self):
        c,a=self.calculate([0,0,90],[1,2,16]);r=c.iloc[0]
        self.assertEqual(r.mean_ppm,30);self.assertEqual(r.n,3)
        self.assertEqual(r.occupied_15min_cells,2);self.assertEqual(r.duplicate_slot_points,1)
        self.assertEqual(a.missing_15min_cells_between.tolist(),[0,0])

    def test_selection_earliest_ties_and_deduplication(self):
        from preliminary_fourier_phases import select_intrabin_examples
        c,_=self.calculate([400]*16,hours=4)
        c=pd.concat([c.assign(sensorid=s) for s in SUBSURFACE_IDS],ignore_index=True)
        e=select_intrabin_examples(c)
        self.assertEqual(len(e),1);self.assertEqual(e.iloc[0].bin_start,pd.Timestamp('2025-05-01'))
        self.assertEqual(e.iloc[0].selection_rules.count('maximum within-bin range'),3)



class TemporalResolutionTests(unittest.TestCase):
    def frames(self, periods=1344):
        times=pd.date_range('2025-05-01',periods=periods,freq='15min')+pd.Timedelta(seconds=1)
        return {sid:pd.DataFrame(dict(localdatetime=times,datavalue=np.arange(periods,dtype=float),
            technically_retained=True,doubtful=False,plateau_review=False)) for sid in SUBSURFACE_IDS}

    def test_residual_metrics_and_zero_sum(self):
        from preliminary_fourier_phases import concentration_residuals
        f=self.frames(8)
        p,s,b,g=concentration_residuals(f,GRID_START,GRID_START+pd.Timedelta(hours=2))
        np.testing.assert_allclose(s.MAE_ppm,2)
        np.testing.assert_allclose(s.RMSE_ppm,np.sqrt(5.25))
        np.testing.assert_allclose(s.max_abs_ppm,3.5)
        np.testing.assert_array_equal(b.sum_residual_ppm,0)
        for sid in SUBSURFACE_IDS:
            self.assertEqual(p.loc[p.sensorid.eq(sid),'localdatetime'].tolist(),f[sid].localdatetime.tolist())

    def test_missing_points_boundaries_and_weighting(self):
        from preliminary_fourier_phases import concentration_residuals
        f=self.frames(17)
        for sid in SUBSURFACE_IDS:
            f[sid]=f[sid].drop(index=[1,2,3,4,5,6])
            f[sid].loc[16,'localdatetime']=GRID_START+pd.Timedelta(hours=4)
        p,s,b,g=concentration_residuals(f,GRID_START,GRID_START+pd.Timedelta(hours=4))
        self.assertEqual(s.N.tolist(),[10]*3)
        # 2 points in first bin, 8 in second: metrics weight observations, not bins.
        self.assertAlmostEqual(s.MAE_ppm.iloc[0],(7+16)/10)
        self.assertAlmostEqual(s.RMSE_ppm.iloc[0],np.sqrt((24.5+42)/10))
        self.assertTrue(b.groupby('sensorid').n.apply(list).map(lambda x:x==[2,8]).all())
        self.assertLess(b.sum_residual_ppm.abs().max(),1e-12)
        self.assertFalse(p.localdatetime.eq(GRID_START+pd.Timedelta(hours=4)).any())

    def test_same_source_set_before_coarse_aggregation(self):
        from preliminary_fourier_phases import matched_resolution_grids
        f=self.frames(16)
        f[996].loc[0,'datavalue']=10000 # must not leak into coarse after other sensor loses cell
        f[1012]=f[1012].drop(index=0)
        f[1035].loc[3,'doubtful']=True
        fine,coarse,cohort=matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(hours=4))
        self.assertEqual(len(cohort),14)
        for sid in SUBSURFACE_IDS:
            self.assertEqual(coarse[f'n_{sid}'].tolist(),[6,8])
            self.assertEqual(coarse[f'mean_{sid}'].iloc[0],sum([1,2,4,5,6,7])/6)
        self.assertFalse(fine.complete_subsurface.iloc[0]);self.assertFalse(fine.complete_subsurface.iloc[3])
        self.assertEqual(cohort.time_996.iloc[0],GRID_START+pd.Timedelta(minutes=15,seconds=1))
        self.assertEqual(fine.bin_center.iloc[0],GRID_START+pd.Timedelta(minutes=7.5))
        self.assertEqual(coarse.bin_center.iloc[0],GRID_START+pd.Timedelta(hours=1))

    def test_14day_calendar_and_known_phase_sign(self):
        from preliminary_fourier_phases import matched_resolution_grids
        f=self.frames(16*96)
        t=(np.arange(16*96)+.5)/96
        for j,sid in enumerate(SUBSURFACE_IDS):f[sid].datavalue=np.cos(2*np.pi*t-j*np.pi/3)
        a,b,_=matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(days=16))
        wa=resolution_windows(a,15,window_days=14);wb=resolution_windows(b,120,window_days=14)
        self.assertEqual(len(wa),3);self.assertEqual(len(wb),3)
        self.assertTrue((wa.end_exclusive-wa.start).eq(pd.Timedelta(days=14)).all())
        self.assertTrue(wa.start.diff().dropna().eq(pd.Timedelta(days=1)).all())
        for pair in PAIRS[1:]:
            np.testing.assert_allclose(wa[pair+'_rad'],np.pi/3,atol=1e-12)
            np.testing.assert_allclose(wb[pair+'_rad'],np.pi/3,atol=1e-12)
        c=compare_resolution_windows(wa,wb,'test')
        np.testing.assert_allclose(c.delta_rad,0,atol=1e-12)
        self.assertTrue(wa.complete_bins.eq(1344).all());self.assertTrue(wb.complete_bins.eq(168).all())

    def test_circular_metric_zero_and_small_modules(self):
        from preliminary_fourier_phases import matched_resolution_grids,temporal_phase_summary
        f=self.frames()
        t=(np.arange(1344)+.5)/96
        for j,sid in enumerate(SUBSURFACE_IDS):f[sid].datavalue=1e-100*np.cos(2*np.pi*t-j*np.pi/3)
        a,b,_=matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(days=14))
        wa=resolution_windows(a,15,window_days=14);wb=resolution_windows(b,120,window_days=14)
        self.assertTrue(wa['5_to_20_rad'].notna().all())
        for pair in PAIRS[1:]:wa[pair+'_rad']=np.pi-.01;wb[pair+'_rad']=-np.pi+.02
        c=compare_resolution_windows(wa,wb,'test');s,_=temporal_phase_summary(c)
        np.testing.assert_allclose(c.delta_rad,-.03,atol=1e-14)
        np.testing.assert_allclose(s.MAE_rad,.03,atol=1e-14)
        for sid in SUBSURFACE_IDS:f[sid].datavalue=400.
        a,b,_=matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(days=14))
        wa=resolution_windows(a,15,window_days=14);wb=resolution_windows(b,120,window_days=14)
        c=compare_resolution_windows(wa,wb,'test');s,_=temporal_phase_summary(c)
        self.assertTrue(s.comparable_windows.eq(0).all());self.assertTrue(s.MAE_rad.isna().all())
        self.assertTrue(wa['5_to_20_reason'].eq('zero_or_nonfinite_coefficient').all())

    def test_actual_support_and_duplicate_slot_guard(self):
        from preliminary_fourier_phases import matched_resolution_grids
        f=self.frames()
        for sid in SUBSURFACE_IDS:f[sid]=f[sid].drop(index=range(8,16))
        a,b,_=matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(days=14))
        wa=resolution_windows(a,15,window_days=14);wb=resolution_windows(b,120,window_days=14)
        self.assertEqual(wa.complete_bins.iloc[0],1336);self.assertEqual(wb.complete_bins.iloc[0],167)
        self.assertEqual(wa.longest_gap_hours.iloc[0],2);self.assertEqual(wb.longest_gap_hours.iloc[0],2)
        self.assertEqual(wa.used_observations_996.iloc[0],wb.used_observations_996.iloc[0])
        f[996]=pd.concat([f[996],f[996].iloc[:1]])
        with self.assertRaises(ValueError):matched_resolution_grids(f,GRID_START,GRID_START+pd.Timedelta(days=14))


class DailyConcentrationScaleTests(unittest.TestCase):
    def calculate(self, times, values, days=3):
        from preliminary_fourier_phases import daily_concentration_ranges
        points=pd.DataFrame(dict(sensorid=996,localdatetime=[pd.Timestamp(t) for t in times],datavalue=values))
        return daily_concentration_ranges(points,GRID_START,GRID_START+pd.Timedelta(days=days))

    def test_calendar_boundaries_and_insufficient_days(self):
        d=self.calculate(['2025-05-01','2025-05-01 23:59:59','2025-05-02','2025-05-04'],[10,40,99,1000])
        self.assertEqual(d.n.tolist(),[2,1,0])
        self.assertEqual(d.D_ppm.iloc[0],30)
        self.assertTrue(d.D_ppm.iloc[1:].isna().all())
        self.assertEqual(d.longest_missing_cells.tolist(),[94,95,96])
        self.assertEqual(d.day_end_exclusive.iloc[-1],GRID_START+pd.Timedelta(days=3))

    def test_cell_coverage_is_not_point_count(self):
        from preliminary_fourier_phases import daily_concentration_ranges
        times=pd.date_range(GRID_START,periods=96,freq='15min')+pd.Timedelta(seconds=1)
        d=self.calculate(times,np.arange(96),days=1)
        self.assertTrue(d.full_day.iloc[0]);self.assertEqual(d.D_ppm.iloc[0],95)
        # 96 points in 95 cells: still incomplete, actual points define extrema.
        times=times.to_list();times[-1]=times[0]+pd.Timedelta(seconds=1)
        d=self.calculate(times,np.arange(96),days=1)
        self.assertEqual(d.n.iloc[0],96);self.assertEqual(d.occupied_15min_cells.iloc[0],95)
        self.assertFalse(d.full_day.iloc[0]);self.assertEqual(d.D_ppm.iloc[0],95)
        self.assertEqual(d.longest_missing_cells.iloc[0],1)

    def test_scale_ratio_and_descriptive_subsets(self):
        from preliminary_fourier_phases import daily_scale_summary
        d=self.calculate(['2025-05-01','2025-05-01 01:00','2025-05-02','2025-05-02 01:00'],[10,30,40,100])
        s,a,n=daily_scale_summary(d,pd.DataFrame(dict(sensorid=[996],MAE_ppm=[8.])))
        self.assertEqual(s.days.iloc[0],2);self.assertEqual(s.insufficient_days.iloc[0],1)
        self.assertEqual(s.D_typ_ppm.iloc[0],40);self.assertEqual(s.R.iloc[0],.2)
        self.assertEqual(a.days.tolist(),[2,0,2]);self.assertTrue(np.isnan(a.D_typ_ppm.iloc[1]))
        self.assertEqual(n.calendar_days.sum(),3)
        d.D_ppm=0
        s,_,_=daily_scale_summary(d,pd.DataFrame(dict(sensorid=[996],MAE_ppm=[0.])))
        self.assertTrue(np.isnan(s.R.iloc[0])) # no invented ratio when D_typ=0


if __name__=='__main__':
    unittest.main()

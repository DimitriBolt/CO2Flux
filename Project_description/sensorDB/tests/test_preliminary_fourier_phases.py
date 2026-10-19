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


if __name__=='__main__':
    unittest.main()

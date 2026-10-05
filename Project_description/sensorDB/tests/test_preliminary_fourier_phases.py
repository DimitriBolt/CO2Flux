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
    window_slices, wrap,
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


if __name__=='__main__':
    unittest.main()

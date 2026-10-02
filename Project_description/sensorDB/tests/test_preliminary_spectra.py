"""Independent numerical checks; no Oracle or archive needed."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from preliminary_spectra import gls, grid, window_starts, coverage


class PreliminarySpectraTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(20261002)
        self.t=np.sort(rng.uniform(0,60,1900))
        self.t=self.t[~((self.t>16)&(self.t<24))]

    def test_known_amplitude_shifted_times_and_gap(self):
        for f in (.17,1.0,2.37):
            y=720+31*np.cos(2*np.pi*f*self.t)+42*np.sin(2*np.pi*f*self.t)
            for shift in (0,19342.75):
                a,_=gls(self.t+shift,y,[f])
                self.assertAlmostEqual(a[0],np.hypot(31,42),places=7)

    def test_independent_lstsq_multicomponent(self):
        y=700+20*np.cos(2*np.pi*.31*self.t)+8*np.sin(2*np.pi*1.1*self.t)
        f=np.array([.17,.31,1,1.1,2.3])
        a,w=gls(self.t,y,f)
        for k,frequency in enumerate(f):
            angle=2*np.pi*frequency*self.t
            matrix=np.column_stack((np.ones(len(y)),np.cos(angle),np.sin(angle)))
            beta=np.linalg.lstsq(matrix,y,rcond=None)[0]
            self.assertAlmostEqual(a[k],np.hypot(beta[1],beta[2]),places=9)
            self.assertAlmostEqual(w[k],abs(np.exp(1j*angle).mean())**2,places=12)

    def test_frequency_recovery_without_oversampling(self):
        t=np.arange(0,60,1/12);t=t[(t<17)|(t>25)]
        f=grid(60);a,_=gls(t,500+15*np.cos(2*np.pi*.7*t),f)
        self.assertAlmostEqual(f[np.nanargmax(a)],.7)
        self.assertAlmostEqual(np.diff(f).min(),1/60)

    def test_no_fake_zero_for_absence_or_singularity(self):
        self.assertTrue(np.isnan(gls([],[],[1])[0][0]))
        self.assertTrue(np.isnan(gls([0,1,2],[1,2,3],[1])[0][0]))
        self.assertTrue(np.isnan(gls(np.arange(8),np.arange(8),[1])[0][0]))
        self.assertEqual(gls(self.t,np.full(len(self.t),400),[1])[0][0],0)

    def test_window_rebuild_and_gap_stats(self):
        self.assertEqual(len(window_starts(521,30,1)),492)
        self.assertEqual(len(window_starts(521,60,7)),66)
        self.assertEqual(len(grid(30)),179)
        self.assertEqual(len(grid(60)),359)
        self.assertTrue((grid(30)<6).all())
        self.assertAlmostEqual(coverage(np.array([0,.1,2,2.1]),0,3)['max_gap_h'],45.6)
        self.assertIn('no retained',coverage([],0,30)['insufficient'])

    def test_browser_worker_matches_python(self):
        for width in (1,30,60):
            t=self.t[self.t<width];y=700+20*np.cos(2*np.pi*.31*t)
            actual=self.browser_result(t,y,width)
            f=np.array(actual['f']);expected,_=gls(t,y,f);expected[f*(t[-1]-t[0])<1-1e-12]=np.nan
            np.testing.assert_allclose(np.array(actual['amplitude'],float),expected,rtol=1e-10,atol=1e-9,equal_nan=True)

    def test_browser_empty_window_has_no_spectrum(self):
        actual=self.browser_result(np.array([]),np.array([]),1)
        self.assertEqual(actual['stats']['n'],0)
        self.assertTrue(all(a is None for a in actual['amplitude']))

    def browser_result(self,t,y,width):
        template=Path(__file__).resolve().parents[1]/'preliminary_spectra.html'
        body=template.read_text().split('function workerBody(){',1)[1].split('\nfunction draw()',1)[0]
        payload=dict(init=[dict(t=t.tolist(),y=y.tolist(),q=[0]*len(t))])
        js='let onmessage; const postMessage=x=>process.stdout.write(JSON.stringify(x)); function workerBody(){'+body+'\nworkerBody();\n'
        js+='onmessage({data:'+json.dumps(payload)+'});\nonmessage({data:{width:'+str(width)+',lo:0,index:0,request:1}});'
        return json.loads(subprocess.check_output(['node','-e',js],text=True))['result'][0]


if __name__=='__main__':
    unittest.main()

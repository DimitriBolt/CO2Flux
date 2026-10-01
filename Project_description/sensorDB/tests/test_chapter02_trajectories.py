"""Accepted-day selection and topology of the Chapter 2 display."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Project_description.Research_log import chapter02 as ch


class TrajectoryTests(unittest.TestCase):
    def sample(self, constant=False):
        times = pd.date_range('2017-10-01', '2017-10-07', freq='5min', inclusive='left')
        sensors = {'D1': 994, 'D2': 1010, 'D3': 1026}
        raw = pd.concat([pd.DataFrame(dict(localdatetime=times, sensorid=sid,
            datavalue=500. + (0 if constant else 20*np.cos(2*np.pi*(times.hour+times.minute/60-peak)/24))))
            for sid,peak in zip(sensors.values(), [23., 2., 6.])], ignore_index=True)
        days = pd.date_range('2017-10-02', periods=3)
        cal = pd.DataFrame(dict(date=days, context_start=days-pd.Timedelta(days=1),
            context_end_exclusive=days+pd.Timedelta(days=2), category=['A','B','C'],
            reason=['accepted','missing data','pending decision']))
        return raw,sensors,cal,pd.DataFrame(columns=['level','start','end'])

    def test_only_a_fitted_with_context_and_no_input_mutation(self):
        args = self.sample()
        raw_before=args[0].copy(deep=True)
        with patch.object(ch.column, 'analyze_column', wraps=ch.column.analyze_column) as call:
            table, fits=ch.calculate_trajectories(*args)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(fits.date.unique().tolist(), [pd.Timestamp('2017-10-02')])
        np.testing.assert_allclose(fits.peak_hour_local, [23,2,6], atol=1e-10)
        np.testing.assert_allclose(fits.amplitude_ppm,20,atol=1e-10)
        self.assertEqual(table.calculation_status.tolist(), ['computed','skipped','skipped'])
        self.assertEqual(table.reason.tolist(),args[2].reason.tolist())
        pd.testing.assert_frame_equal(args[0],raw_before)

    def test_zero_amplitude_retains_amplitude_without_inventing_phase(self):
        table,fits=ch.calculate_trajectories(*self.sample(constant=True))
        self.assertTrue(fits.peak_hour_local.isna().all())
        self.assertTrue(fits.phase_status.eq('undefined').all())
        np.testing.assert_allclose(fits.amplitude_ppm,0,atol=1e-10)

    def test_unexpected_a_failure_is_recorded(self):
        args=list(self.sample())
        args[0]=pd.concat([args[0],args[0].iloc[[0]]],ignore_index=True)
        table,fits=ch.calculate_trajectories(*args)
        self.assertTrue(fits.empty)
        self.assertEqual(table.iloc[0].calculation_status,'failed')
        self.assertIn('Повторные',table.iloc[0].calculation_error)

    def test_midnight_link_uses_faces_instead_of_cube_chord(self):
        pieces=ch.cyclic_segments([23,4,5],[1,4,5])
        np.testing.assert_allclose(pieces, [[[23,4,5],[24,4,5]],[[0,4,5],[1,4,5]]])
        reverse=ch.cyclic_segments([1,4,5],[23,4,5])
        np.testing.assert_allclose(reverse, [[[1,4,5],[0,4,5]],[[24,4,5],[23,4,5]]])

    def test_antipodal_or_undefined_phase_has_no_arbitrary_link(self):
        self.assertEqual(ch.cyclic_segments([0,1,2],[12,1,2]), [])
        self.assertEqual(ch.cyclic_segments([np.nan,1,2],[3,1,2]), [])

    def test_calendar_gaps_are_not_joined(self):
        table=pd.DataFrame(dict(date=pd.to_datetime(['2018-01-01','2018-01-03','2018-01-04']),
            calculation_status=['computed']*3,H_D1=[1,2,3],H_D2=[4,5,6],H_D3=[7,8,9]))
        segments,_=ch.trajectory_segments(table,'H')
        self.assertEqual(len(segments),1)
        np.testing.assert_allclose(segments[0],[[2,5,8],[3,6,9]])

    def test_preliminary_groups_exclude_audit_refusals_without_bridging_dates(self):
        table = pd.DataFrame(dict(date=pd.to_datetime(['2013-09-29', '2013-09-30', '2013-10-01', '2013-10-02', '2013-10-04']),
            calculation_status=['computed']*5,
            admission_status=['confirmed', 'confirmed', 'not_admitted_audit', 'unconfirmed', 'unconfirmed']))
        for level in ('D1', 'D2', 'D3'):
            table[f'H_{level}'] = [23., 1., 4., 6., 9.]
            table[f'R2_{level}'] = [0., .3, .9, .001, .5]
            table[f'phase_status_{level}'] = 'not_assessed'
        before = table.copy(deep=True)
        fig = ch.interactive_trajectory(table, 'H', preliminary=True)
        markers = [trace for trace in fig.data if trace.mode == 'markers']
        self.assertEqual([trace.marker.symbol for trace in markers], ['circle', 'diamond-open'])
        self.assertEqual([list(trace.x) for trace in markers], [[23., 1.], [6., 9.]])
        self.assertEqual(sum(len(trace.x) for trace in markers), 4)  # even zero R² is retained
        lines = [trace for trace in fig.data if trace.mode == 'lines']
        self.assertEqual(len(lines), 1)
        self.assertEqual(list(lines[0].x), [23., 24., None, 0., 1., None])
        pd.testing.assert_frame_equal(table, before)

    def test_preliminary_undefined_phase_is_omitted_only_from_H(self):
        table = pd.DataFrame(dict(date=pd.date_range('2020-01-01', periods=2),
            calculation_status=['computed']*2, admission_status=['unconfirmed']*2))
        for level in ('D1', 'D2', 'D3'):
            table[f'H_{level}'] = [np.nan, 2.]
            table[f'A_{level}'] = [0., 3.]
            table[f'R2_{level}'] = [np.nan, .2]
            table[f'phase_status_{level}'] = ['undefined', 'not_assessed']
        h, a = [ch.interactive_trajectory(table, kind, preliminary=True) for kind in ('H', 'A')]
        self.assertEqual(sum(len(t.x) for t in h.data if t.mode == 'markers'), 1)
        self.assertEqual(sum(len(t.x) for t in a.data if t.mode == 'markers'), 2)
        self.assertIn('Неопределённые координаты: 1', h.layout.annotations[0].text)

if __name__ == '__main__':
    unittest.main()

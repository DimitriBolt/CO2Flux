"""Source identity preservation around the unchanged admission cleaner."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Project_description.Research_log import chapter02 as ch


EMPTY = pd.DataFrame(columns=['level', 'start', 'end'])
AIR = {'C_air': 1275}
AIR_METADATA = {'C_air': dict(variableid=56, units='umol/mol',
    table='LEO_WEST.DATAVALUESLICOR', sensorcode='LEO-W_4_0_1_LI-7000')}


def source(values, *, sid=1275, variableid=56):
    return pd.DataFrame(dict(
        localdatetime=pd.date_range('2025-01-01', periods=len(values), freq='2h'),
        datavalue=values, sensorid=sid, variableid=variableid))


class AdmissionMetadataTests(unittest.TestCase):
    def clean(self, frame, metadata=AIR_METADATA):
        return ch.clean_admission_measurements(frame, AIR, EMPTY,
                                               channel_metadata=metadata)

    def test_default_preserves_legacy_schema_and_variableid(self):
        frame = source([500, 501, 499, 700, 501, 500, 499])
        legacy = ch.clean_admission_measurements(frame, AIR, EMPTY)
        explicit = ch.clean_admission_measurements(frame, AIR, EMPTY,
                                                  channel_metadata=None)
        pd.testing.assert_frame_equal(legacy, explicit)
        self.assertTrue(legacy.variableid.eq(9).all())
        self.assertEqual(legacy.columns.tolist(), [
            'localdatetime', 'datavalue', 'source_rows', 'sensorid', 'variableid',
            'level', 'original_value', 'exact_duplicate_extra', 'exclusion_reason',
            'doubt_reason', 'despike_window_eligible', 'constant_run_hours'])

    def test_air_metadata_and_input_are_preserved(self):
        frame = source([500, 501, 499, 700, 501, 500, 499])
        frame['source_part'] = 'part-0001.parquet'
        frame['source_row_in_part'] = np.arange(len(frame))
        before = frame.copy(deep=True)
        metadata_before = deepcopy(AIR_METADATA)
        work = self.clean(frame)
        for field, expected in AIR_METADATA['C_air'].items():
            self.assertTrue(work[field].eq(expected).all())
        self.assertTrue(work.sensorid.eq(1275).all())
        self.assertTrue(work.level.eq('C_air').all())
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(AIR_METADATA, metadata_before)

    def test_metadata_does_not_change_any_numeric_decision(self):
        cases = [
            [500, 501, 499, 700, 501, 500, 499],
            [500, 500, 500, 700, 500, 500, 500],
            [500, 501, 499, 700, 701, 500, 499],
            [-10000, -9999, -1, 0, .1, 3000, 3000.1, np.nan],
        ]
        for values in cases:
            with self.subTest(values=values):
                frame = source(values)
                legacy = ch.clean_admission_measurements(frame, AIR, EMPTY)
                actual = self.clean(frame).drop(columns=['units', 'table', 'sensorcode'])
                actual['variableid'] = 9
                pd.testing.assert_frame_equal(actual, legacy)

    def test_two_channels_keep_distinct_variables_and_units(self):
        frame = pd.concat([source([400, 410]), source([500, 510], sid=996, variableid=9)],
                          ignore_index=True)
        sensors = dict(AIR, C_5=996)
        metadata = dict(AIR_METADATA, C_5=dict(variableid=9, units='ppm',
            table='LEO_WEST.DATAVALUES', sensorcode='LEO-W_4_1_1_GMM222'))
        work = ch.clean_admission_measurements(frame, sensors, EMPTY,
                                               channel_metadata=metadata)
        for level, sid in sensors.items():
            part = work.loc[work.level.eq(level)]
            self.assertTrue(part.sensorid.eq(sid).all())
            for field, expected in metadata[level].items():
                self.assertTrue(part[field].eq(expected).all())

    def test_duplicate_grouping_keeps_metadata(self):
        frame = source([400, 400, 500, 501])
        frame['localdatetime'] = pd.to_datetime([
            '2025-01-01', '2025-01-01', '2025-01-01 02:00', '2025-01-01 02:00'],
            format='mixed')
        work = self.clean(frame)
        self.assertEqual(work.source_rows.tolist(), [2, 2])
        self.assertEqual(work.exact_duplicate_extra.tolist(), [1, 0])
        self.assertEqual(work.exclusion_reason.tolist(), ['', 'conflicting_timestamp'])
        for field, expected in AIR_METADATA['C_air'].items():
            self.assertTrue(work[field].eq(expected).all())

    def test_missing_wrong_mixed_or_undefined_source_variableid_is_rejected(self):
        frame = source([400, 410])
        bad = [frame.drop(columns='variableid'), frame.assign(variableid=9)]
        for values in ([56, 9], [56, np.nan], [56, pd.NA], ['56', '56']):
            variant = frame.copy()
            variant['variableid'] = values
            bad.append(variant)
        for variant in bad:
            with self.subTest(variableid=variant.get('variableid')):
                with self.assertRaises(ValueError):
                    self.clean(variant)

    def test_metadata_requires_exact_levels_and_all_identity_fields(self):
        bad = [{}, dict(AIR_METADATA, C_5=AIR_METADATA['C_air']), {'other': AIR_METADATA['C_air']},
               {'C_air': None}]
        for field in ('variableid', 'units', 'table', 'sensorcode'):
            variant = deepcopy(AIR_METADATA)
            del variant['C_air'][field]
            bad.append(variant)
        for metadata in bad:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    self.clean(source([400, 410]), metadata)

    def test_metadata_rejects_invalid_variableid_and_blank_labels(self):
        for value in (True, np.bool_(True), 56., '56', None):
            metadata = deepcopy(AIR_METADATA)
            metadata['C_air']['variableid'] = value
            with self.subTest(variableid=value):
                with self.assertRaises(ValueError):
                    self.clean(source([400, 410]), metadata)
        for field in ('units', 'table', 'sensorcode'):
            for value in ('', '  ', None, 56):
                metadata = deepcopy(AIR_METADATA)
                metadata['C_air'][field] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.clean(source([400, 410]), metadata)


if __name__ == '__main__':
    unittest.main()

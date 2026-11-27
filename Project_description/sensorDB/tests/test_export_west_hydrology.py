"""Tests of record preservation and CO2-format compatibility; no Oracle access."""
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_west_hydrology as export


class HydrologyExportTests(unittest.TestCase):
    def test_records_service_values_duplicate_times_and_exact_decimal(self):
        a, b = datetime(2012, 1, 1), datetime(2026, 11, 26, 0, 0, 1)
        rows = [(3, 19, 6, b, Decimal("1.234567890123456789")),
                (2, 19, 6, a, Decimal("-9999")),
                (1, 19, 6, a, Decimal("-9999"))]
        raw = export.raw_batch(rows)
        self.assertEqual(raw["datavalue_oracle_decimal"].to_pylist()[0], "1.234567890123456789")
        frame, ordered = export.output_frame(raw.to_pandas())
        self.assertEqual(list(ordered.valueid), [1, 2, 3])
        self.assertEqual(list(frame.datavalue[:2]), [-9999., -9999.])
        self.assertEqual(len(frame), 3)
        self.assertFalse(frame.index.is_unique)
        self.assertEqual(list(frame.columns), ["datavalue"])
        self.assertEqual(frame.index.name, "localdatetime")
        self.assertEqual(str(frame.index.dtype), "datetime64[us]")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "channel.parquet"
            frame.to_parquet(path)
            pd.testing.assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
            schema = pq.ParquetFile(path).schema_arrow
            self.assertEqual(schema.names, ["datavalue", "localdatetime"])
            self.assertEqual(json.loads(schema.metadata[b"pandas"])["index_columns"], ["localdatetime"])

    def test_empty_channel_preserves_named_time_index(self):
        frame, _ = export.output_frame(export.raw_batch([]).to_pandas())
        self.assertEqual(len(frame), 0)
        self.assertEqual(str(frame.datavalue.dtype), "float64")
        self.assertEqual(str(frame.index.dtype), "datetime64[us]")
        self.assertEqual(frame.index.name, "localdatetime")

    def test_negative_potential_null_and_extremes_are_retained(self):
        t = datetime(2020, 1, 1, 1, 2, 3)
        raw = export.raw_batch([(1, 668, 7, t, Decimal("-11.4")),
                                (2, 668, 7, t, Decimal("1E20")),
                                (3, 668, 7, t, None)])
        frame, _ = export.output_frame(raw.to_pandas())
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.datavalue.iloc[0], -11.4)
        self.assertEqual(frame.datavalue.iloc[1], 1e20)
        self.assertTrue(pd.isna(frame.datavalue.iloc[2]))
        self.assertEqual(list(frame.index), [pd.Timestamp(t)] * 3)

    def test_month_bounds_cover_all_actual_history_and_both_tables(self):
        c = [dict(table="LEO_WEST.DATAVALUES", initial_rows=1,
                  initial_first="2012-12-17 01:00:01", initial_last="2026-11-26 16:45:03"),
             dict(table="LEO_WEST.DATAVALUESREU24", initial_rows=1,
                  initial_first="2024-06-26 08:45:01", initial_last="2026-11-26 16:45:03")]
        blocks = export.month_blocks(c)
        self.assertEqual(blocks[0], ("DATAVALUES", datetime(2012, 12, 1), datetime(2013, 1, 1)))
        self.assertEqual(blocks[-1], ("DATAVALUESREU24", datetime(2026, 11, 1), datetime(2026, 12, 1)))
        self.assertEqual(len(blocks), 168 + 30)

    def test_transaction_count_check_checkpoint_and_resume(self):
        t = datetime(2019, 1, 1, 0, 0, 1)
        rows = [(1, 19, 6, t, Decimal("-9999")), (2, 19, 6, t, Decimal("12.5"))]

        class Cursor:
            def __init__(self, count):
                self.count, self.sent, self.statements = count, False, []
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def execute(self, sql, *_): self.statements.append(sql)
            def fetchone(self): return (t,)
            def fetchall(self): return [(19, 6, self.count, t, t)]
            def fetchmany(self, *_):
                if self.sent: return []
                self.sent = True
                return rows

        class Connection:
            def __init__(self, count): self.cur = Cursor(count)
            def cursor(self): return self.cur
            def close(self): pass

        catalog = [dict(table="LEO_WEST.DATAVALUES", sensorid=19, variableid=6)]
        for expected, succeed in [(2, True), (3, False)]:
            with tempfile.TemporaryDirectory() as d:
                work = Path(d)
                conn = Connection(expected)
                args = (work, "DATAVALUES", datetime(2019, 1, 1), datetime(2019, 2, 1), catalog)
                with patch.object(export, "connect", return_value=conn):
                    if succeed:
                        report = export.download_block(*args)
                        self.assertEqual(report["rows"], 2)
                    else:
                        with self.assertRaisesRegex(ValueError, "count/bounds mismatch"):
                            export.download_block(*args)
                self.assertEqual(conn.cur.statements[0], "SET TRANSACTION READ ONLY")
                manifest = work / "blocks/DATAVALUES/2019-01/manifest.json"
                self.assertEqual(manifest.exists(), succeed)
                if succeed:
                    with patch.object(export, "connect", side_effect=AssertionError("Resume queried Oracle")):
                        self.assertEqual(export.download_block(*args)["rows"], 2)
                    path = manifest.parent / "19_6.parquet"
                    path.write_bytes(b"corrupt")
                    with self.assertRaisesRegex(ValueError, "Checkpoint integrity failure"):
                        export.download_block(*args)

    def test_complete_publication_includes_empty_channels_and_duplicate_report(self):
        t = datetime(2020, 1, 1, 0, 0, 1)
        with tempfile.TemporaryDirectory() as d:
            work = Path(d) / "source"
            work.mkdir()
            output = Path(d) / "published"
            output.mkdir()
            export.dump(work / "capture.json", {"base_commit": "synthetic"})
            export.dump(work / "runs.json", [])
            blockdir = work / "blocks/DATAVALUES/2020-01"
            blockdir.mkdir(parents=True)
            entries, catalog = [], []
            for vid, sid, values in [(6, 19, [Decimal("-9999"), Decimal("-9999")]),
                                     (7, 668, [Decimal("-11.4")]), (17, 1066, [])]:
                folder, suffix, unit, *_ = export.QUANTITIES[vid]
                c = dict(table="LEO_WEST.DATAVALUES", sensorid=sid, variableid=vid,
                         folder=folder, file=f"LEO-W_4_1_1_{suffix}.parquet", units=unit,
                         nodatavalue=-9999, initial_rows=len(values))
                catalog.append(c)
                if values:
                    path = blockdir / f"{sid}_{vid}.parquet"
                    pq.write_table(export.raw_batch([(i + 1, sid, vid, t, v) for i, v in enumerate(values)]), path)
                    entries.append(dict(sensorid=sid, variableid=vid, rows=len(values), file=path.name))
            blocks = [dict(table="LEO_WEST.DATAVALUES", start="2020-01-01 00:00:00",
                           end_exclusive="2020-02-01 00:00:00", oracle_snapshot="synthetic", channels=entries)]
            reports = export.build_outputs(work, catalog, blocks, output,
                                           filters=export.RowFilters(keep_service_values=True))
            self.assertEqual(sum(r["rows"] for r in reports), 3)
            self.assertEqual(reports[-1]["status"], "no_rows")
            duplicate_path = output / "LEO_West_VWC/duplicate_times.csv"
            self.assertEqual(pd.read_csv(duplicate_path).rows.iloc[0], 2)
            for c in catalog:
                frame = pd.read_parquet(output / c["folder"] / c["file"])
                self.assertEqual(len(frame), c["initial_rows"])
                self.assertEqual(frame.index.name, "localdatetime")
                self.assertEqual(list(frame.columns), ["datavalue"])

            reports = export.build_outputs(work, catalog, blocks, output, replace=True)
            self.assertEqual(sum(r["rows"] for r in reports), 1)
            self.assertEqual(reports[0]["removed_service_rows"], 2)
            self.assertEqual(reports[0]["status"], "filtered_out")
            self.assertTrue(pd.read_csv(duplicate_path).empty)

    def test_filters_use_half_open_time_and_keep_valid_duplicate_times(self):
        a, b = datetime(2020, 1, 1), datetime(2020, 1, 2)
        source = export.raw_batch([
            (1, 668, 7, a, Decimal("-9999")), (2, 668, 7, a, Decimal("-11.4")),
            (3, 668, 7, a, Decimal("-10")), (4, 668, 7, a, Decimal("-20")),
            (5, 668, 7, a, None), (6, 668, 7, b, Decimal("-10")),
        ]).to_pandas()
        filtered, counts = export.RowFilters(start=a, end=b, drop_nonfinite=True,
                                             min_value=-11.4, max_value=-10).apply(source, -9999)
        self.assertEqual(list(filtered.valueid), [2, 3])
        self.assertEqual(counts, dict(source_rows_in_window=5, removed_service_rows=1,
                                      removed_nonfinite_rows=1, removed_by_value_range=1))
        self.assertTrue(filtered.localdatetime.duplicated().any())


if __name__ == "__main__":
    unittest.main()

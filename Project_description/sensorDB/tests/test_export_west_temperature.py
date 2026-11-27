"""Temperature archive invariants with local synthetic Oracle batches."""
import csv
from datetime import datetime, timedelta
from decimal import Decimal
import io
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_west_hydrology as hydro
import export_west_temperature as app
import west_temperature_source as src


class TemperatureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.t = datetime(2025, 5, 1, 11, 13, 2, 123456)
        self.c = dict(
            table="LEO_WEST.DATAVALUES",
            sensorid=3,
            variableid=3,
            file="channel.parquet",
            quantity="soil",
            units="degC",
            available=True,
            nodatavalue=-888,
            initial_rows=10,
            sensor_type="5TM",
            x=1,
            y=4,
            depth_m=0.05,
            height_m=None,
        )
        values = [12, -888, None, float("nan"), float("inf"), float("-inf"), 0, -4, -4, 82]
        self.data = hydro.raw_batch(
            [(i, 3, 3, self.t + timedelta(seconds=min(i, 8)), v) for i, v in enumerate(values)]
        )
        self.data = self.data.append_column("source_null_test", self.data["datavalue"].is_null()).drop(
            ["source_null_test"]
        )
        self.part = src.write_channel(
            self.root, (3, 3), [self.data], {(3, 3): (10, self.t, self.t + timedelta(seconds=8))}, -888
        )

    def build(self, filters, parts=None, full=False):
        dup = io.StringIO()
        report, _ = app.build_channel(
            self.c,
            [(self.root, self.part)] if parts is None else parts,
            self.root,
            filters,
            csv.writer(dup),
            full,
        )
        return report, pd.read_parquet(self.root / self.c["file"]), dup.getvalue()

    def test_clean_balance_duplicates_and_original_times(self):
        r, frame, dups = self.build(app.RowFilters(drop_nonfinite=True), full=True)
        self.assertEqual(frame.datavalue.tolist(), [12, 0, -4, -4, 82])
        self.assertEqual(frame.index[0], self.t)
        self.assertEqual(frame.index[-2], frame.index[-1])
        self.assertIn("8;9", dups)
        self.assertEqual((r["source_service_rows"], r["null_rows"], r["nonfinite_nonnull_rows"]), (1, 1, 3))
        self.assertEqual(str(frame.index.dtype), "datetime64[us]")
        self.assertEqual(frame.index.name, "localdatetime")
        self.assertEqual(frame.dtypes.astype(str).to_dict(), {"datavalue": "float64"})

    def test_time_range_values_and_explicit_service_mode(self):
        r, frame, _ = self.build(
            app.RowFilters(
                self.t + timedelta(seconds=6),
                self.t + timedelta(seconds=8),
                drop_nonfinite=True,
                min_value=-4,
                max_value=0,
            )
        )
        self.assertEqual(frame.datavalue.tolist(), [0, -4])
        self.assertEqual(r["source_rows_in_window"], 2)
        _, frame, _ = self.build(
            app.RowFilters(keep_service_values=True, drop_nonfinite=True, min_value=-888, max_value=-888)
        )
        self.assertEqual(frame.datavalue.tolist(), [-888])

    def test_empty_all_service_and_user_filtered(self):
        r, empty, _ = self.build(app.RowFilters(drop_nonfinite=True), [])
        self.assertEqual((r["status"], str(empty.index.dtype)), ("no_source_rows", "datetime64[us]"))
        r, _, _ = self.build(
            app.RowFilters(
                start=self.t + timedelta(seconds=1), end=self.t + timedelta(seconds=2), drop_nonfinite=True
            )
        )
        self.assertEqual(r["status"], "only_service_values")
        r, _, _ = self.build(app.RowFilters(drop_nonfinite=True, min_value=99))
        self.assertEqual(r["status"], "filtered_out")
        self.c["available"] = False
        r, _, _ = self.build(app.RowFilters(drop_nonfinite=True), [])
        self.assertEqual(r["status"], "source_unavailable")

    def test_spatial_multi_channel_and_native_units(self):
        rows = [self.c, self.c | dict(sensorid=4, depth_m=0.2), self.c | dict(sensorid=5, depth_m=0.5)]
        args = app.parse(
            [
                "--x",
                "1",
                "--y-min",
                "4",
                "--y-max",
                "4",
                "--depth-m",
                ".05",
                "--depth-m",
                ".2",
                "--sensor-id",
                "3",
                "--sensor-id",
                "4",
                "--sensor-type",
                "5tm",
                "--source-table",
                "datavalues",
                "--variable-id",
                "3",
                "--quantity",
                "soil",
            ]
        )
        self.assertEqual(app.select(rows, args), rows[:2])
        with self.assertRaisesRegex(ValueError, "native unit"):
            app.select([self.c, self.c | dict(units="degK")], app.parse(["--min-value", "0"]))

    def test_filenames_and_protected_targets(self):
        rows = [self.c.copy(), self.c | dict(sensorid=4), self.c | dict(table="LEO_WEST.DATAVALUESVIZ")]
        app.unique_names(rows)
        self.assertEqual(len({r["file"] for r in rows}), 3)
        for path in (app.ROW, app.ROW / "LEO_West", app.RAW / "v1"):
            with self.assertRaises(ValueError):
                app.protect_output(path, True, app.RAW / "v1")
        app.protect_output(app.RAW / "selections" / "demo", True, app.RAW / "v1")

    def test_resumed_block_checks_integrity_without_oracle(self):
        t = "DATAVALUES"
        a, b = datetime(2025, 5, 1), datetime(2025, 6, 1)
        d = src.block_path(self.root, t, a)
        d.mkdir(parents=True)
        (self.root / "inventory").mkdir()
        src.dump(self.root / "inventory/capture.json", {})
        os.rename(self.root / self.part["file"], d / self.part["file"])
        src.dump(
            d / "manifest.json",
            dict(
                table=t,
                start=str(a),
                end=str(b),
                capture_sha256=src.sha(self.root / "inventory/capture.json"),
                channels=[self.part],
                rows=10,
            ),
        )
        with patch.object(src, "connect", side_effect=AssertionError("Offline resume contacted Oracle")):
            self.assertEqual(src.download_block(self.root, (t, a, b), {}, set(), {})["rows"], 10)
            with (d / self.part["file"]).open("ab") as f:
                f.write(b"broken")
            with self.assertRaises(Exception):
                src.download_block(self.root, (t, a, b), {}, set(), {})

    def test_physical_interpretation(self):
        loc = {"boxverticaldatum": "depth below the surface", "boxz": 0}
        sensor = {"sensorcode": "LEO-W_2_-3_LoadCellSurface_TC-E"}
        self.assertEqual(
            app.classify("DATAVALUES", "SurfaceTemp", "TC-E", sensor, loc, None)[:3], ("other", None, None)
        )
        sensor = {"sensorcode": "LEO-W_4_0_1_HMP60"}
        loc["boxverticaldatum"] = "height about the surface"
        self.assertEqual(
            app.classify("DATAVALUES", "AirTemp", "HMP60", sensor, loc, None)[:3], ("air", None, None)
        )
        b = dict(code=sensor["sensorcode"], height_cm=25)
        self.assertEqual(app.classify("DATAVALUES", "AirTemp", "HMP60", sensor, loc, b)[2], 0.25)

    def test_full_publish_offline_subset_and_independent_verification(self):
        import verify_west_temperature as audit

        work = self.root / "snapshot"
        row = self.root / "working"
        meta = work / "inventory"
        meta.mkdir(parents=True)
        cap = dict(tables=["DATAVALUES"], metadata_sha256={}, cutoff_local="2025-05-31 23:59:59")
        src.dump(meta / "capture.json", cap)
        src.dump(meta / "DATAVALUES.json", [dict(first_time=str(self.t), last_time=str(self.t))])
        plan = next(src.planned_blocks(work, cap))
        d = src.block_path(work, *plan[:2])
        d.mkdir(parents=True)
        # Oracle NUMBER cannot encode NaN/Inf; this independent test uses real NUMBER semantics.
        values = [-9999, None, 0, -2, Decimal("12.123456789012345678901")]
        data = hydro.raw_batch([(i, 3, 3, self.t + timedelta(seconds=i), v) for i, v in enumerate(values)])
        part = src.write_channel(
            d, (3, 3), [data], {(3, 3): (5, self.t, self.t + timedelta(seconds=4))}, -9999
        )
        sig = src.sha(meta / "capture.json")
        manifest = d / "manifest.json"
        src.dump(
            manifest,
            dict(
                table=plan[0],
                start=str(plan[1]),
                end=str(plan[2]),
                snapshot=str(self.t),
                capture_sha256=sig,
                reader_sha256="example",
                channels=[part],
                rows=5,
            ),
        )
        src.dump(
            work / "download_complete.json",
            dict(
                rows=5,
                blocks=1,
                manifests={str(manifest.relative_to(work)): src.sha(manifest)},
                capture_sha256=sig,
            ),
        )
        c = self.c | dict(initial_rows=5, nodatavalue=-9999)
        app.build(work, cap, [c], app.parse([]), row, True)
        folder = row / app.FOLDERS["soil"]
        src.dump(
            work / "export_complete.json",
            dict(channels=1, provenance_sha256={"soil": src.sha(folder / "provenance.json")}),
        )
        self.assertEqual(audit.verify(work, row)["retained"], 3)
        original = src.sha(folder / c["file"])
        out = self.root / "filtered"
        args = app.parse(["--local-only", "--min-value", "0", "--max-value", "0", "--output-dir", str(out)])
        with patch.object(src, "connect", side_effect=AssertionError("Offline contacted Oracle")):
            app.build(work, cap, [c], args, out, False)
        self.assertEqual(pd.read_parquet(out / app.FOLDERS["soil"] / c["file"]).datavalue.tolist(), [0])
        self.assertEqual(src.sha(folder / c["file"]), original)

    def test_interrupted_month_restarts_and_completed_month_is_reused(self):
        import pyarrow as pa
        from contextlib import nullcontext
        from unittest.mock import MagicMock

        cap = {"cutoff_local": "2025-05-31 23:59:59", "predicate": "variableid IN (3)"}
        meta = self.root / "inventory"
        meta.mkdir()
        src.dump(meta / "capture.json", cap)
        a, b = datetime(2025, 5, 1), datetime(2025, 6, 1)
        plan = ("DATAVALUES", a, b)
        batch = pa.table({"i": [101], "s": [3], "v": [3], "t": [self.t], "decimal": ["-2.25"]})
        cur = MagicMock()
        cur.__enter__.return_value = cur
        cur.fetchone.return_value = (self.t,)
        cur.__iter__.side_effect = lambda: iter([(3, 3, 1, self.t, self.t)])
        conn = MagicMock()
        conn.cursor.return_value = cur

        def fail():
            yield batch
            raise ConnectionError("injected interrupted Oracle fetch")

        conn.fetch_df_batches.side_effect = [fail(), iter([batch])]
        with patch.object(src, "connect", side_effect=lambda: nullcontext(conn)) as connect:
            args = (self.root, plan, cap, {(3, 3)}, {3: -9999})
            with self.assertRaises(ConnectionError):
                src.download_block(*args)
            directory = src.block_path(self.root, *plan[:2])
            self.assertFalse(directory.exists())
            first = src.download_block(*args)
            self.assertEqual(first["channels"][0]["valid"], 1)
            self.assertEqual(src.download_block(*args), first)
            self.assertEqual(connect.call_count, 2)
            self.assertEqual(cur.execute.call_args_list[0].args, ("SET TRANSACTION READ ONLY",))

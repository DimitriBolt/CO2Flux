"""No Oracle imports/connections: protect stage 0.2 counting and time boundaries."""
import unittest
from unittest.mock import Mock, patch
from decimal import Decimal
from pathlib import Path
import tempfile

import pandas as pd

from Project_description.Research_log import stage02_coverage as stage


class Stage02Tests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"localdatetime": pd.to_datetime([
            "2024-01-31 23:59:59", "2024-02-01 00:00:00", "2024-02-01 00:00:00",
            "2024-02-01 12:00:01", "2024-02-02 00:00:00"]),
            "datavalue": [400., -9999., None, 401., 402.]})

    def test_month_partition_no_boundary_loss(self):
        start, end = pd.Timestamp("2024-01-31 12:30"), pd.Timestamp("2024-03-01")
        windows = list(stage.month_windows(start, end))
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)
        self.assertEqual(windows[0][1], windows[1][0])
        counts = [stage.window_metrics(self.frame(), a, b)["rows"] for a, b in windows]
        self.assertEqual(counts, [1, 4])

    def test_no_cleaning_rounding_or_exact_match_requirement(self):
        f = self.frame()
        before = f.copy(deep=True)
        result = stage.window_metrics(f, pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-02"))
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["nonnull_values"], 2)
        self.assertEqual(result["max_internal_gap_s"], 43201)
        self.assertEqual(result["trailing_gap_s"], 43199)
        pd.testing.assert_frame_equal(before, f)

    def test_empty_and_singleton_do_not_imply_zero_gap_coverage(self):
        f = self.frame()
        empty = stage.window_metrics(f, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-02-01"))
        one = stage.window_metrics(f, pd.Timestamp("2024-02-02"), pd.Timestamp("2024-02-03"))
        self.assertEqual(empty["rows"], 0)
        self.assertIsNone(empty["max_internal_gap_s"])
        self.assertIsNone(one["max_internal_gap_s"])
        self.assertEqual(one["trailing_gap_s"], 86400)

    def test_multiset_comparison_retains_duplicates_and_null(self):
        old = self.frame()
        new = pd.concat([old, old.iloc[[1]]], ignore_index=True)
        result = stage.compare_observations(old, new)
        self.assertEqual(result["matched_old_rows"], len(old))
        self.assertEqual(result["additional_new_rows"], 1)
        self.assertEqual(result["missing_old_rows"], 0)
        result = stage.compare_observations(new, old)
        self.assertEqual(result["missing_old_rows"], 1)

    def test_service_null_duplicate_statistics_retained(self):
        actual = stage.local_stats(self.frame())
        self.assertEqual(actual["rows"], 5)
        self.assertEqual(actual["null_values"], 1)
        self.assertEqual(actual["service_le_minus9999"], 1)
        self.assertEqual(actual["duplicate_time_extras"], 1)

    def test_days_without_observations_remain_in_calendar(self):
        calendar = pd.DataFrame({"common_nonnull_day": [False, True, True, False]},
                                index=pd.date_range("2024-01-01", periods=4))
        result = stage.common_runs(calendar)
        self.assertEqual(result.days.tolist(), [1, 2, 1])
        self.assertEqual(result.all_four_have_nonnull_each_day.tolist(), [False, True, False])
        self.assertEqual(result.iloc[1].end_exclusive, pd.Timestamp("2024-01-04"))

    def test_query_error_is_not_retried(self):
        cur = Mock()
        cur.execute.side_effect = RuntimeError("access denied")
        with self.assertRaises(RuntimeError):
            stage.bounded_select(cur, "SELECT column_name FROM all_tab_columns")
        self.assertEqual(cur.execute.call_count, 1)

    def test_calendar_and_vector_windows_keep_offsets_and_missing_days(self):
        channels = {}
        for i, name in enumerate(("C_air", "C_5", "C_20", "C_35")):
            frame = self.frame()
            frame.localdatetime += pd.Timedelta(seconds=i)
            channels[name] = frame
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(stage, "START", pd.Timestamp("2024-01-31 12:30")), \
                patch.object(stage, "END", pd.Timestamp("2024-03-05")):
            folder = Path(tmp) / "diagnostics"
            result = stage.diagnostics(channels, folder, {})
            # The 1–3 second offsets move the Jan 31 record across midnight.
            self.assertEqual(result["common_days"], 2)
            calendar = pd.read_csv(folder / "daily_calendar.csv")
            self.assertEqual(len(calendar), 34)
            self.assertEqual(calendar.C_air_rows.sum(), 5)
            self.assertTrue(calendar.C_air_rows.eq(0).any())
            windows = pd.read_csv(folder / "windows.csv", parse_dates=["start"], date_format="mixed")
            row = windows.loc[windows.channel.eq("C_air") & windows.days.eq(1)
                              & windows.start.eq(pd.Timestamp("2024-02-01"))].iloc[0]
            self.assertEqual(row.rows, 3)
            self.assertEqual(row.max_internal_gap_s, 43201)
            self.assertEqual(set(windows.days), {1, 7, 30})

    def test_extraction_with_fake_oracle_preserves_ids_nulls_and_duplicates(self):
        frame = self.frame()
        frame["valueid"] = [Decimal(i) for i in range(1, 6)]
        frame["sensorid"] = Decimal(1275)
        frame["variableid"] = Decimal(56)
        frame["datavalue"] = [Decimal("400.00001"), Decimal(-9999), None, Decimal(401), Decimal(402)]
        names = ["valueid", "datavalue", "localdatetime", "sensorid", "variableid"]
        meta = {"columns": [{"name": n.upper(), "type": "DATE" if n == "localdatetime" else "NUMBER"} for n in names]}
        conn = Mock()
        raw_queries = []

        def select(cur, query, params):
            part = frame.loc[frame.localdatetime.ge(params["start_time"]) & frame.localdatetime.lt(params["end_time"])]
            raw_queries.append(query)
            return list(part[names].itertuples(index=False, name=None))

        def stats(cur, start, end):
            return stage.local_stats(frame.loc[frame.localdatetime.ge(start) & frame.localdatetime.lt(end)])

        with tempfile.TemporaryDirectory() as tmp, patch.object(stage, "check_inputs"), \
                patch.object(stage, "connect", return_value=conn), patch.object(stage, "metadata", return_value=meta), \
                patch.object(stage, "month_windows", return_value=iter([
                    (pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-01")),
                    (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-03-01"))])), \
                patch.object(stage, "stats", side_effect=stats), patch.object(stage, "bounded_select", side_effect=select):
            result = stage.extract(Path(tmp) / "archive")
            actual = pd.read_parquet(Path(tmp) / "archive/raw.parquet")
            self.assertEqual(result["rows"], 5)
            self.assertEqual(actual.datavalue.iloc[0], Decimal("400.00001"))
            self.assertEqual(result["totals"]["null_values"], 1)
            self.assertEqual(result["totals"]["duplicate_time_extras"], 1)
            self.assertEqual(result["totals"]["service_le_minus9999"], 1)
            self.assertEqual(actual.valueid.nunique(), 5)
            self.assertTrue(conn.close.called)
            for query in raw_queries:
                self.assertNotIn("DISTINCT", query)
                self.assertNotIn("GROUP BY", query)
                self.assertNotIn("datavalue<", query)


if __name__ == "__main__":
    unittest.main()

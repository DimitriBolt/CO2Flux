"""Published admission evidence must not become an invented liveness calendar."""
import unittest

import pandas as pd

from Project_description.Research_log.prepare_chapter02 import archive_admission


class ArchiveAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.inventory = pd.DataFrame({
            "audit_code": ["W_R4_C-4_D1", "W_R4_C-4_D2", "W_R4_C-4_D3"],
            "first_alive": ["2013-09"] * 3,
            "last_alive": ["2023-06", "2020-12", "2021-09"],
        })
        self.blocks = [("2017-10-01", "2020-09-01", "published joint block")]

    def status(self, month):
        return archive_admission(month, self.inventory, "2026-05", self.blocks)[0]

    def test_first_last_never_fill_unpublished_internal_months(self):
        for month in ("2013-10", "2014-08", "2017-09", "2020-09", "2020-12"):
            with self.subTest(month=month):
                self.assertEqual(self.status(month), "unconfirmed")

    def test_common_first_and_explicit_joint_block_are_confirmed(self):
        for month in ("2013-09", "2017-10", "2020-08"):
            self.assertEqual(self.status(month), "confirmed")
        self.inventory.loc[0, "first_alive"] = "2013-08"
        self.assertEqual(self.status("2013-09"), "unconfirmed")

    def test_negative_last_evidence_ends_at_audit_boundary(self):
        for month in ("2021-01", "2026-05"):
            self.assertEqual(self.status(month), "not_admitted_audit")
        for month in ("2026-06", "2026-09"):
            self.assertEqual(self.status(month), "unconfirmed")


if __name__ == "__main__":
    unittest.main()

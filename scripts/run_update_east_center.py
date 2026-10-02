"""Refresh existing East/Center CO2 rows without changing West or support rows."""
from __future__ import annotations

import update_co2_sheet as u


def main(*, workbook_path=None) -> None:
    u.main(workbook_path=workbook_path, slopes={"LEO East", "LEO Center"})


if __name__ == "__main__":
    main()

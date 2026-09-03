from __future__ import annotations

import pandas as pd

from ml_scan.data.calendar import NSECalendar


def test_hourly_index_has_seven_bars() -> None:
    cal = NSECalendar()
    idx = cal.hourly_index(pd.Timestamp("2024-01-02", tz="Asia/Kolkata"))
    assert len(idx) == 7

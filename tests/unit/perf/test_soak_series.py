import datetime as dt

import pytest

from scripts.perf.soak_series import (
    parse_users_series,
    parse_durations_series,
    parse_duration_to_seconds,
    build_plan,
    build_run_dir,
    format_snapshot_name,
)
from pathlib import Path


pytestmark = pytest.mark.unit


def test_parse_series_args():
    assert parse_users_series("100, 500,1000") == [100, 500, 1000]
    durs = parse_durations_series("1d,3h,45m,30s")
    assert durs == ["1d", "3h", "45m", "30s"]
    # spot check seconds
    assert int(parse_duration_to_seconds("1d")) == 86400
    assert int(parse_duration_to_seconds("3h")) == 10800
    assert int(parse_duration_to_seconds("45m")) == 2700
    assert int(parse_duration_to_seconds("30s")) == 30


def test_plan_generation_tmp(tmp_path: Path):
    users = [100, 500]
    durs = ["1h", "2h"]
    plan = build_plan(users, durs, profile="soak", out_root=tmp_path, fixed_ts="20250101-010203")
    assert len(plan) == 4
    # Deterministic run_dir names
    assert str(plan[0].run_dir).endswith("soak-u100-20250101-010203")
    assert str(plan[1].run_dir).endswith("soak-u100-20250101-010203")
    assert str(plan[2].run_dir).endswith("soak-u500-20250101-010203")
    # Seconds parsed
    assert int(plan[0].seconds) == 3600


def test_snapshot_filename():
    d = dt.datetime(2025, 1, 2, 3, 4, 5)
    assert format_snapshot_name(d, "md") == "summary-20250102-03.md"
    assert format_snapshot_name(d, "html") == "summary-20250102-03.html"


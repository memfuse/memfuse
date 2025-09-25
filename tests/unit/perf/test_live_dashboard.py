import threading
import time
import urllib.request
from pathlib import Path

import pytest

from scripts.perf.live_dashboard import (
    parse_locust_history_csv,
    parse_db_metrics_jsonl,
    serve,
)


pytestmark = pytest.mark.unit


def test_build_timeseries_from_csv():
    csv_text = (
        "Timestamp,User Count,Requests/s,Fail Ratio\n"
        "2025-01-01 00:00:00,10,5.0,0.0\n"
        "2025-01-01 00:01:00,12,6.5,0.1\n"
    )
    out = parse_locust_history_csv(csv_text)
    assert out["timestamps"][0].startswith("2025-01-01")
    assert out["users"] == [10, 12]
    assert out["rps"] == [5.0, 6.5]
    assert out["fail_ratio"] == [0.0, 0.1]


def test_build_timeseries_from_jsonl():
    jsonl = (
        '{"ts":"t1","metrics":{"connections_by_state":{"active":2},"idle_in_transaction":0}}\n'
        '{"ts":"t2","metrics":{"connections_by_state":{"active":3,"idle":1},"idle_in_transaction":1}}\n'
    )
    out = parse_db_metrics_jsonl(jsonl)
    assert out["states"] == ["active", "idle"]
    assert out["timeline"][0]["states"]["active"] == 2
    assert out["timeline"][1]["idle_in_tx"] == 1


def test_server_serves_files(tmp_path: Path):
    # Prepare a simple file in the directory
    f = tmp_path / "foo.txt"
    f.write_text("hello", encoding="utf-8")

    httpd = serve(tmp_path, host="127.0.0.1", port=0)  # auto port
    host, port = httpd.server_address
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        time.sleep(0.1)
        with urllib.request.urlopen(f"http://{host}:{port}/foo.txt") as resp:
            body = resp.read().decode("utf-8")
            assert body == "hello"
    finally:
        httpd.shutdown()
        httpd.server_close()
        t.join(timeout=2)


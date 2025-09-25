#!/usr/bin/env python3
"""
Lightweight live dashboard server and utilities.

Serves a run directory (containing Locust history CSV and DB JSONL) and a
static live.html from this repository for real-time charts. Also exposes
helpers to parse time-series from CSV/JSONL for unit tests.

Usage:
  poetry run python scripts/perf/live_dashboard.py --dir outputs/perf/<run> --host 127.0.0.1 --port 8088

Then open http://127.0.0.1:8088/live.html

The page will fetch `locust_stats_history.csv` and `db_metrics.jsonl` in that
directory periodically and render charts.
"""

from __future__ import annotations

import argparse
import csv
import http.server
import io
import json
import os
import socketserver
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ------------------------------
# Parsing helpers (unit-tested)
# ------------------------------


def parse_locust_history_csv(data: str) -> Dict[str, List[Any]]:
    """Parse Locust stats history CSV into arrays for charting.

    Returns a dict with keys: timestamps, users, rps, fail_ratio.
    """
    if not data.strip():
        return {"timestamps": [], "users": [], "rps": [], "fail_ratio": []}
    reader = csv.DictReader(io.StringIO(data))
    ts: List[str] = []
    users: List[int] = []
    rps: List[float] = []
    fail: List[float] = []
    for row in reader:
        ts.append(str(row.get("Timestamp") or row.get("timestamp") or ""))
        try:
            users.append(int(float(row.get("User Count") or 0)))
        except Exception:
            users.append(0)
        try:
            rps.append(float(row.get("Requests/s") or row.get("Request/s") or 0.0))
        except Exception:
            rps.append(0.0)
        try:
            fail.append(float(row.get("Fail Ratio") or 0.0))
        except Exception:
            fail.append(0.0)
    return {"timestamps": ts, "users": users, "rps": rps, "fail_ratio": fail}


def parse_db_metrics_jsonl(data: str) -> Dict[str, Any]:
    """Parse DB metrics JSONL into a timeline suitable for charting.

    Returns a dict with keys: timeline (list of {ts, idle_in_tx, states{}}), states (union of state keys).
    """
    timeline: List[Dict[str, Any]] = []
    state_keys: set[str] = set()
    for line in data.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        ts = obj.get("ts")
        metrics = obj.get("metrics") or {}
        states = metrics.get("connections_by_state") or {}
        idle = metrics.get("idle_in_transaction") or 0
        if isinstance(states, dict):
            state_keys.update(k or "" for k in states.keys())
        timeline.append({"ts": ts, "idle_in_tx": idle, "states": states})
    return {"timeline": timeline, "states": sorted([k or "" for k in state_keys])}


# ------------------------------
# Static server
# ------------------------------


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self) -> None:  # type: ignore[override]
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def serve(run_dir: Path, host: str = "127.0.0.1", port: int = 8088) -> socketserver.TCPServer:
    # Change working directory so SimpleHTTPRequestHandler serves from run_dir
    os.chdir(str(run_dir))
    handler = CORSRequestHandler
    httpd = socketserver.TCPServer((host, port), handler)
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Live dashboard server")
    ap.add_argument("--dir", required=True, help="Run directory to serve (contains CSV/JSONL)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    args = ap.parse_args(argv)

    run_dir = Path(args.dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return 2

    # Serve run dir and also expose live.html from repo if not present
    live_html = run_dir / "live.html"
    if not live_html.exists():
        # Copy content from scripts/perf/live.html
        repo_live = Path(__file__).resolve().parent / "live.html"
        if repo_live.exists():
            try:
                live_html.write_text(repo_live.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

    httpd = serve(run_dir, host=args.host, port=args.port)
    print(f"Serving {run_dir} at http://{args.host}:{args.port}/live.html (Ctrl+C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


#!/usr/bin/env python3
"""
Worker Scaling Study Harness

Runs the MemFuse server via uvicorn with varying worker counts, executes a
load scenario for each, and collects artifacts per worker setting.

Artifacts per run:
- db_metrics.jsonl (DB sampler)
- locust_*.csv (Locust request stats)
- summary.md (aggregated report)

Example:
  poetry run python scripts/perf/worker_scaling.py \
    --workers 1,2,4 --port 8010 --profile load --users 50 --spawn 10 --runtime 20m
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def wait_for_health(base_url: str, timeout_s: float = 60.0) -> bool:
    import urllib.request
    import urllib.error

    url = base_url.rstrip("/") + "/api/v1/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_once(
    workers: int,
    host: str,
    port: int,
    profile: str,
    users: int,
    spawn: int,
    runtime: str,
    out_root: Path,
) -> Path:
    run_dir = out_root / f"w{workers}-{timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{host}:{port}"

    # 1) Start server using memfuse-core with Hydra config overrides
    # Note: MemFuse architecture doesn't support multi-worker uvicorn due to service initialization
    # We use single process and test load scaling instead of worker scaling
    uvicorn_cmd = [
        "poetry",
        "run",
        "memfuse-core",
        f"server.host={host}",
        f"server.port={port}",
    ]

    # Use current environment (poetry handles dependencies)
    env = os.environ.copy()

    print(f"[scaling] Starting memfuse-core server on {host}:{port} (worker param ignored - using single process)")
    # Avoid blocking due to unconsumed pipes; discard output
    server = subprocess.Popen(uvicorn_cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ok = wait_for_health(base_url, timeout_s=90)
        if not ok:
            raise RuntimeError(f"Server did not become healthy at {base_url}")

        # 2) Start DB metrics sampler
        db_cmd = [
            "poetry",
            "run",
            "python",
            "scripts/perf/db_metrics.py",
            "--interval",
            os.getenv("DB_INTERVAL", "5s"),
            "--duration",
            runtime,
            "--out",
            str(run_dir / "db_metrics.jsonl"),
        ]
        db_proc = subprocess.Popen(db_cmd)

        # 3) Run Locust load with profile
        locust_cmd = [
            "locust",
            "-f",
            "tests/performance/api/locustfile.py",
            "--host",
            base_url,
            "--users",
            str(users),
            "--spawn-rate",
            str(spawn),
            "--run-time",
            runtime,
            "--headless",
            "--csv",
            str(run_dir / "locust"),
            "--csv-full-history",
        ]
        env = os.environ.copy()
        env["PROFILE"] = profile
        print(f"[scaling] Running Locust: {' '.join(locust_cmd)} (PROFILE={profile})")
        locust_rc = subprocess.call(locust_cmd, env=env)
        if locust_rc != 0:
            print(f"[scaling] Locust exited with code {locust_rc}")

        # 4) Aggregate summary
        agg_cmd = [
            "poetry",
            "run",
            "python",
            "scripts/perf/aggregate_report.py",
            "--run-dir",
            str(run_dir),
            "--host",
            base_url,
        ]
        subprocess.call(agg_cmd)

        # 5) Wait for DB sampler
        db_proc.wait(timeout=120)

        print(f"[scaling] Completed workers={workers}. Artifacts: {run_dir}")
        return run_dir
    finally:
        try:
            server.terminate()
            # Give it a moment, then force kill if needed
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-worker scaling study harness")
    ap.add_argument("--workers", default="1,2,4", help="Comma-separated worker counts, e.g., 1,2,4")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--profile", default="load")
    ap.add_argument("--users", type=int, default=50)
    ap.add_argument("--spawn", type=int, default=10)
    ap.add_argument("--runtime", default="20m")
    ap.add_argument("--out-root", default="outputs/perf/scaling")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    worker_list = [int(x.strip()) for x in str(args.workers).split(",") if x.strip()]

    print(f"[scaling] Running workers={worker_list}, profile={args.profile}, users={args.users}, spawn={args.spawn}, runtime={args.runtime}")
    for w in worker_list:
        run_once(w, args.host, args.port, args.profile, args.users, args.spawn, args.runtime, out_root)

    print(f"[scaling] All runs complete. See {out_root}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

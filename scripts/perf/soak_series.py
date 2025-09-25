#!/usr/bin/env python3
"""
Soak Series Harness: run a matrix of (users x durations) sequentially.

Features:
- Parses user counts and durations (e.g., 1d, 3h, 45m)
- Builds timestamped run directories and invokes profile_run.py for each
- Optional hourly snapshots: periodically generate summary-{YYYYMMDD-HH}.{md,html}
- Dry-run mode for planning and unit tests

Example:
  poetry run python scripts/perf/soak_series.py \
    --host http://localhost:8000 --profile soak \
    --users-series 100,500,1000 --durations 1d,3h \
    --spawn 10 --db-interval 60s --snapshot-hourly
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def parse_duration_to_seconds(value: str) -> float:
    s = str(value).strip().lower()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 3600.0
    if s.endswith("d"):
        return float(s[:-1]) * 86400.0
    return float(s)


def parse_users_series(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


def parse_durations_series(s: str) -> List[str]:
    # Keep as strings (e.g., '1d', '3h') for display; convert to seconds when needed
    out: List[str] = []
    for part in (s or "").split(","):
        part = part.strip()
        if part:
            out.append(part)
    return out


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_run_dir(root: Path, profile: str, users: int, fixed_ts: str | None = None) -> Path:
    ts = fixed_ts or now_ts()
    return root / f"{profile}-u{users}-{ts}"


def format_snapshot_name(dt: datetime, ext: str = "md") -> str:
    ts = dt.strftime("%Y%m%d-%H")
    return f"summary-{ts}.{ext}"


@dataclass
class PlanItem:
    users: int
    duration: str  # original token, e.g., '1d'
    seconds: float
    run_dir: Path


def build_plan(users: List[int], durations: List[str], profile: str, out_root: Path, fixed_ts: str | None = None) -> List[PlanItem]:
    plan: List[PlanItem] = []
    for u in users:
        for d in durations:
            secs = parse_duration_to_seconds(d)
            rd = build_run_dir(out_root, profile, u, fixed_ts=fixed_ts)
            plan.append(PlanItem(users=u, duration=d, seconds=secs, run_dir=rd))
    return plan


def run_profile_once(item: PlanItem, host: str, profile: str, spawn: float, db_interval: str, allow_failures: bool, seed: bool) -> int:
    cmd = [
        sys.executable,
        "scripts/perf/profile_run.py",
        "--host",
        host,
        "--profile",
        profile,
        "--users",
        str(item.users),
        "--spawn",
        str(spawn),
        "--runtime",
        item.duration,
        "--db-interval",
        db_interval,
        "--run-dir",
        str(item.run_dir),
    ]
    if allow_failures:
        cmd.append("--allow-failures")
    if seed:
        cmd.append("--seed")
    env = os.environ.copy()
    proc = subprocess.Popen(cmd)
    return proc.wait()


def snapshot_worker(run_dir: Path, host: str, stop_event: threading.Event, interval_s: int = 3600) -> None:
    while not stop_event.wait(timeout=interval_s):
        try:
            # Write hourly MD and HTML snapshots
            subprocess.run([
                sys.executable,
                "scripts/perf/aggregate_report.py",
                "--run-dir",
                str(run_dir),
                "--host",
                host,
            ], check=False)
            subprocess.run([
                sys.executable,
                "scripts/perf/aggregate_report.py",
                "--run-dir",
                str(run_dir),
                "--host",
                host,
                "--format",
                "html",
            ], check=False)
        except Exception:
            pass


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Soak series harness")
    ap.add_argument("--host", default=os.getenv("HOST", "http://localhost:8765"))
    ap.add_argument("--profile", default=os.getenv("PROFILE", "soak"))
    ap.add_argument(
        "--users-series",
        default=os.getenv("USERS_SERIES", "500"),
        help="Comma-separated users list (default: 500). Example: 100,500,1000",
    )
    ap.add_argument("--durations", default="30m", help="Comma-separated durations, default 30m (e.g., 1d,3h,45m)")
    ap.add_argument("--spawn", type=float, default=float(os.getenv("SPAWN", "10")))
    ap.add_argument("--db-interval", default=os.getenv("DB_INTERVAL", "60s"))
    ap.add_argument("--out-root", default="outputs/perf/series")
    ap.add_argument("--snapshot-hourly", action="store_true")
    ap.add_argument("--allow-failures", action="store_true")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    users = parse_users_series(args.users_series)
    durations = parse_durations_series(args.durations)
    plan = build_plan(users, durations, args.profile, out_root)

    if args.dry_run:
        print("Plan:")
        for i, it in enumerate(plan, 1):
            print(f" {i}. users={it.users} duration={it.duration} ({it.seconds:.0f}s) run_dir={it.run_dir}")
        return 0

    rc_total = 0
    for it in plan:
        it.run_dir.mkdir(parents=True, exist_ok=True)
        stop_evt = threading.Event()
        t = None
        if args.snapshot_hourly:
            t = threading.Thread(target=snapshot_worker, args=(it.run_dir, args.host, stop_evt), daemon=True)
            t.start()
        rc = run_profile_once(it, args.host, args.profile, args.spawn, args.db_interval, args.allow_failures, args.seed)
        if t is not None:
            stop_evt.set()
            t.join(timeout=5)
        if rc != 0:
            print(f"Run failed for users={it.users} duration={it.duration} with code {rc}")
            rc_total = rc
            # Continue to next to complete the matrix
    return rc_total


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""
One-shot profiler for MemFuse (seed → run → report) with progress/ETA.

This script orchestrates a single performance run:
  1) Optional dataset seeding (via scripts/perf/seed_data.py)
  2) Capture run metadata (config/env/git)
  3) Start DB metrics sampler for the duration of the run
  4) Run Locust headless with selected profile and parameters
  5) Aggregate a Markdown summary in the run directory

Outputs live in a timestamped run dir under outputs/perf/<profile>-YYYYMMDD-HHMMSS

Examples
  # Load profile, no seeding
  poetry run python scripts/perf/profile_run.py \
    --host http://localhost:8000 --profile load --users 50 --spawn 10 --runtime 20m

  # With seeding
  poetry run python scripts/perf/profile_run.py \
    --host http://localhost:8000 --profile load --users 50 --spawn 10 --runtime 20m \
    --seed --seed-users 10 --seed-agents 2 --seed-sessions-per-user 3 \
    --seed-messages-per-session 50 --seed-msg-size-profile mixed --seed-concurrency 8
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_duration_to_seconds(value: str) -> float:
    """Parse duration strings like '30', '30s', '5m', '2h' to seconds (float)."""
    s = str(value).strip().lower()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    if s.endswith("h"):
        return float(s[:-1]) * 3600.0
    return float(s)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_subprocess(cmd: List[str], env: Optional[Dict[str, str]] = None) -> int:
    """Run a subprocess and stream its output to the console.

    Args:
        cmd: Command + args.
        env: Environment overrides.
    Returns:
        Process return code.
    """
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    return proc.wait()


def spawn_background(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    log_path: Optional[Path] = None,
) -> Tuple[subprocess.Popen, Optional[object]]:
    """Spawn a background process.

    If log_path is provided, redirect stdout/stderr to that file to avoid
    pipe buffering deadlocks. Returns (proc, log_handle) so caller can keep
    the handle alive and close it when done.
    """
    if log_path is not None:
        log_handle = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return proc, log_handle
    else:
        proc = subprocess.Popen(
            cmd,
            stdout=None,
            stderr=None,
            env=env,
        )
        return proc, None


def print_spinner(prefix: str, proc: subprocess.Popen) -> int:
    """Simple spinner while proc is running. Returns proc return code."""
    spinner = ["-", "\\", "|", "/"]
    i = 0
    try:
        while proc.poll() is None:
            sys.stdout.write(f"\r{prefix} {spinner[i % 4]}")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)
        rc = proc.wait()
        sys.stdout.write("\r" + " " * (len(prefix) + 2) + "\r")
        sys.stdout.flush()
        return rc
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        return proc.wait()


def render_progress_bar(elapsed: float, total: float, width: int = 40) -> str:
    ratio = 0.0 if total <= 0 else min(1.0, max(0.0, elapsed / total))
    done = int(ratio * width)
    remaining = width - done
    bar = "#" * done + "-" * remaining
    pct = int(ratio * 100)
    eta = max(0.0, total - elapsed)
    return f"[{bar}] {pct:3d}%  ETA {int(eta)}s"


def progress_loop(total_seconds: float, desc: str) -> None:
    start = time.time()
    while True:
        now = time.time()
        elapsed = now - start
        bar = render_progress_bar(elapsed, total_seconds)
        sys.stdout.write(f"\r{desc} {bar}")
        sys.stdout.flush()
        if elapsed >= total_seconds:
            break
        time.sleep(0.2)
    sys.stdout.write("\n")
    sys.stdout.flush()


@dataclass
class SeedOptions:
    enabled: bool = False
    users: int = 3
    agents: int = 1
    sessions_per_user: int = 2
    messages_per_session: int = 20
    msg_size_profile: str = "mixed"
    concurrency: int = 6
    per_agent_sessions: bool = False


def build_run_dir(profile: str, base: Path) -> Path:
    return base / f"{profile}-{now_ts()}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="One-shot MemFuse profiler")
    ap.add_argument("--host", default=os.getenv("HOST", "http://localhost:8000"))
    ap.add_argument("--profile", default=os.getenv("PROFILE", "load"))
    ap.add_argument("--users", type=int, default=int(os.getenv("USERS", "50")))
    ap.add_argument("--spawn", type=float, default=float(os.getenv("SPAWN", "10")))
    ap.add_argument("--runtime", default=os.getenv("RUNTIME", "20m"))
    ap.add_argument("--db-interval", default=os.getenv("DB_INTERVAL", "5s"))
    ap.add_argument("--db-dsn", default=os.getenv("DB_DSN", ""), help="Override DB DSN for sampler (or use POSTGRES_* env vars)")
    ap.add_argument("--run-dir", default="", help="Optional run dir; default uses timestamp")

    # Seeding
    ap.add_argument("--seed", action="store_true", help="Run dataset seeding before the profile")
    ap.add_argument("--seed-users", type=int, default=int(os.getenv("SEED_USERS", "3")))
    ap.add_argument("--seed-agents", type=int, default=int(os.getenv("SEED_AGENTS", "1")))
    ap.add_argument("--seed-sessions-per-user", type=int, default=int(os.getenv("SEED_SESSIONS_PER_USER", "2")))
    ap.add_argument("--seed-messages-per-session", type=int, default=int(os.getenv("SEED_MESSAGES_PER_SESSION", "20")))
    ap.add_argument(
        "--seed-msg-size-profile",
        choices=["short", "mixed", "long"],
        default=os.getenv("SEED_MSG_SIZE_PROFILE", "mixed"),
    )
    ap.add_argument("--seed-concurrency", type=int, default=int(os.getenv("SEED_CONCURRENCY", "6")))
    ap.add_argument("--per-agent-sessions", action="store_true")

    ap.add_argument("--allow-failures", action="store_true", help="Don't fail run if Locust has request errors (pass --exit-code-on-error 0)")
    args = ap.parse_args(argv)

    host = args.host.rstrip("/")
    profile = args.profile
    run_root = Path("outputs") / "perf"
    ensure_dir(run_root)
    run_dir = Path(args.run_dir) if args.run_dir else build_run_dir(profile, run_root)
    ensure_dir(run_dir)

    runtime_s = parse_duration_to_seconds(args.runtime)
    db_interval = args.db_interval

    # 0) Seed (optional)
    if args.seed:
        seed_cmd = [
            sys.executable,
            "scripts/perf/seed_data.py",
            "--base-url",
            host,
            "--users",
            str(args.seed_users),
            "--agents",
            str(args.seed_agents),
            "--sessions-per-user",
            str(args.seed_sessions_per_user),
            "--messages-per-session",
            str(args.seed_messages_per_session),
            "--msg-size-profile",
            args.seed_msg_size_profile,
            "--concurrency",
            str(args.seed_concurrency),
        ]
        if args.per_agent_sessions:
            seed_cmd.append("--per-agent-sessions")

        # Pass through API key envs if present
        seed_env = os.environ.copy()
        # Visual spinner during seed
        print("🌱 Seeding dataset…")
        seed_log = run_dir / "seed.log"
        seed_proc, seed_handle = spawn_background(seed_cmd, env=seed_env, log_path=seed_log)
        rc_seed = print_spinner("Seeding", seed_proc)
        if seed_handle is not None:
            try:
                seed_handle.close()
            except Exception:
                pass
        if rc_seed != 0:
            print(f"❌ Seeding failed with code {rc_seed}")
            return rc_seed
        print("✅ Seeding complete")

    # 1) Capture run metadata
    meta_cmd = [
        sys.executable,
        "scripts/perf/capture_run_metadata.py",
        "--run-dir",
        str(run_dir),
        "--host",
        host,
    ]
    print("🧾 Capturing run metadata…")
    rc_meta = run_subprocess(meta_cmd)
    if rc_meta != 0:
        print(f"❌ Metadata capture failed with code {rc_meta}")
        return rc_meta

    # 2) Start DB metrics sampler (background)
    # Build DB sampler command with DSN (fallbacks to sensible defaults)
    def _default_dsn() -> str:
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "54321")  # default overridden per requirement
        db = os.getenv("POSTGRES_DB", "memfuse")
        user = os.getenv("POSTGRES_USER", "postgres")
        pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
        parts = [f"host={host}", f"port={port}", f"dbname={db}", f"user={user}"]
        if pwd:
            parts.append(f"password={pwd}")
        return " ".join(parts)

    dsn = args.db_dsn or os.getenv("DB_DSN") or _default_dsn()

    db_cmd = [
        sys.executable,
        "scripts/perf/db_metrics.py",
        "--interval",
        db_interval,
        "--duration",
        args.runtime,
        "--out",
        str(run_dir / "db_metrics.jsonl"),
        "--dsn",
        dsn,
    ]
    print(
        f"🗄️  Starting DB sampler every {db_interval} for {args.runtime} → {run_dir/'db_metrics.jsonl'}"
    )
    db_log = run_dir / "db_sampler.log"
    db_proc, db_handle = spawn_background(db_cmd, env=os.environ.copy(), log_path=db_log)

    # 3) Run Locust headless
    locust_cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        "tests/performance/api/locustfile.py",
        "--host",
        host,
        "--users",
        str(args.users),
        "--spawn-rate",
        str(args.spawn),
        "--run-time",
        args.runtime,
        "--headless",
        "--csv",
        str(run_dir / "locust"),
        "--csv-full-history",
    ]
    env_locust = os.environ.copy()
    env_locust["PROFILE"] = profile
    if args.allow_failures:
        locust_cmd.extend(["--exit-code-on-error", "0"])
    print(
        f"🚀 Running profile '{profile}' for {args.runtime} @ users={args.users} spawn={args.spawn}"
    )

    # Launch Locust and render a time-based progress bar until done.
    locust_log = run_dir / "locust.log"
    locust_proc, locust_handle = spawn_background(locust_cmd, env=env_locust, log_path=locust_log)
    # Progress bar loop in parallel
    progress_loop(total_seconds=runtime_s, desc="⏳ Load test")
    rc_locust = locust_proc.wait()
    if rc_locust != 0:
        print(f"❌ Locust exited with code {rc_locust}")
        # Surface last lines from locust log for quick diagnosis
        try:
            with open(locust_log, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
            tail = "".join(lines[-50:]) if lines else "(empty log)"
            print("--- locust.log tail ---\n" + tail + "\n-----------------------")
        except Exception:
            pass
        # Try to terminate DB sampler
        try:
            db_proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        _ = db_proc.wait(timeout=10)
        # Close log handles
        for h in (db_handle, locust_handle):
            try:
                h and h.close()
            except Exception:
                pass
        return rc_locust

    # 4) Wait for DB sampler to finish naturally
    try:
        rc_db = db_proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            db_proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        rc_db = db_proc.wait()

    if rc_db != 0:
        print(f"⚠️  DB sampler exited with code {rc_db}")
    # Close log handles
    for h in (db_handle, locust_handle):
        try:
            h and h.close()
        except Exception:
            pass

    # 5) Aggregate report
    agg_cmd = [
        sys.executable,
        "scripts/perf/aggregate_report.py",
        "--run-dir",
        str(run_dir),
        "--host",
        host,
    ]
    print("📊 Aggregating summary report…")
    rc_agg = run_subprocess(agg_cmd)
    if rc_agg != 0:
        print(f"❌ Aggregation failed with code {rc_agg}")
        return rc_agg
    # Also generate HTML report
    agg_cmd_html = agg_cmd + ["--format", "html"]
    rc_agg_html = run_subprocess(agg_cmd_html)
    if rc_agg_html != 0:
        print(f"⚠️  HTML aggregation failed with code {rc_agg_html}")

    print("✅ Done. Artifacts:")
    print(f"  - Run dir: {run_dir}")
    print(f"  - Summary: {run_dir/'summary.md'}")
    print(f"  - HTML: {run_dir/'summary.html'}")
    print(f"  - Locust CSVs: {run_dir/'locust_stats.csv'} + history/failures")
    print(f"  - DB metrics: {run_dir/'db_metrics.jsonl'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

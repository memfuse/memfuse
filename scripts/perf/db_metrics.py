#!/usr/bin/env python3
"""
DB Metrics Sampler for MemFuse

Periodically samples PostgreSQL metrics and writes JSONL records for
correlation with HTTP load tests.

Metrics collected per sample:
- connections_by_state (pg_stat_activity)
- wait_events (type/event counts from pg_stat_activity)
- idle_in_transaction count
- deadlocks_total (pg_stat_database)
- top_queries (pg_stat_statements, limited to top by total_time)

Usage examples:
  # Use env vars for DSN and write to default path every 5s for 10m
  poetry run python scripts/perf/db_metrics.py --interval 5s --duration 10m

  # Explicit DSN and custom output path
  poetry run python scripts/perf/db_metrics.py \
    --dsn "host=localhost port=5432 dbname=memfuse user=postgres password=postgres" \
    --interval 3 --duration 600 --out outputs/perf/db_metrics.jsonl

Environment variables for DSN (used if --dsn not provided):
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import psycopg
except Exception as e:  # pragma: no cover
    print("psycopg is required to run db_metrics.py. Install it in your env.", file=sys.stderr)
    raise


def parse_duration(value: str) -> float:
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
    # default: seconds
    return float(s)


def build_default_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "memfuse")
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "")
    # Use key=value DSN to avoid URI encoding issues
    parts = [
        f"host={host}",
        f"port={port}",
        f"dbname={db}",
        f"user={user}",
    ]
    if pwd:
        parts.append(f"password={pwd}")
    return " ".join(parts)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_connections_by_state(cur) -> Dict[str, int]:
    cur.execute(
        """
        select coalesce(state,'') as state, count(*) as cnt
        from pg_stat_activity
        where datname = current_database()
        group by state
        order by cnt desc
        """
    )
    return {row[0] or "": int(row[1]) for row in cur.fetchall()}


def fetch_wait_events(cur) -> List[Dict[str, Any]]:
    cur.execute(
        """
        select coalesce(wait_event_type,''), coalesce(wait_event,''), count(*) as cnt
        from pg_stat_activity
        where datname = current_database() and wait_event_type is not null
        group by wait_event_type, wait_event
        order by cnt desc
        limit 20
        """
    )
    return [
        {"type": row[0] or "", "event": row[1] or "", "count": int(row[2])}
        for row in cur.fetchall()
    ]


def fetch_idle_in_tx(cur) -> int:
    cur.execute(
        """
        select count(*) from pg_stat_activity
        where datname = current_database() and state = 'idle in transaction'
        """
    )
    return int(cur.fetchone()[0])


def fetch_deadlocks_total(cur) -> int:
    cur.execute(
        """
        select coalesce(deadlocks, 0)
        from pg_stat_database
        where datname = current_database()
        """
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def fetch_top_queries(cur) -> List[Dict[str, Any]]:
    """Fetch top queries from pg_stat_statements. Gracefully handle absence."""
    try:
        # Prefer filtering to current database if dbid/datname available
        try:
            cur.execute(
                """
                select queryid, calls, total_time, rows
                from pg_stat_statements
                order by total_time desc
                limit 20
                """
            )
        except Exception:
            # Some installations require filtering by database OID
            cur.execute(
                """
                select s.queryid, s.calls, s.total_time, s.rows
                from pg_stat_statements s
                join pg_database d on d.oid = s.dbid
                where d.datname = current_database()
                order by s.total_time desc
                limit 20
                """
            )
        rows = cur.fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            queryid, calls, total_time, rows_out = r
            calls = int(calls or 0)
            total_time = float(total_time or 0.0)
            mean_time = total_time / calls if calls else 0.0
            out.append(
                {
                    "queryid": int(queryid) if queryid is not None else None,
                    "calls": calls,
                    "total_time_ms": round(total_time, 3),
                    "mean_time_ms": round(mean_time, 3),
                    "rows": int(rows_out or 0),
                }
            )
        return out
    except Exception as e:
        # Extension not available or permission issues
        return []


def sample_once(conn: psycopg.Connection) -> Dict[str, Any]:
    with conn.cursor() as cur:
        metrics = {
            "connections_by_state": fetch_connections_by_state(cur),
            "wait_events": fetch_wait_events(cur),
            "idle_in_transaction": fetch_idle_in_tx(cur),
            "deadlocks_total": fetch_deadlocks_total(cur),
            "top_queries": fetch_top_queries(cur),
        }
    return metrics


def connect_with_autocommit(dsn: str, timeout: int) -> psycopg.Connection:
    conn = psycopg.connect(dsn, connect_timeout=timeout)
    conn.autocommit = True
    return conn


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Postgres metrics sampler (JSONL)")
    parser.add_argument("--dsn", type=str, default=os.getenv("DB_DSN", ""), help="psycopg DSN string")
    parser.add_argument("--interval", type=str, default=os.getenv("DB_SAMPLER_INTERVAL", "5s"), help="sampling interval (e.g., 5s, 1m)")
    parser.add_argument("--duration", type=str, default=os.getenv("DB_SAMPLER_DURATION", "10m"), help="total duration (e.g., 10m, 2h)")
    parser.add_argument("--out", type=str, default=os.getenv("DB_SAMPLER_OUT", "outputs/perf/db_metrics.jsonl"), help="output JSONL path")
    parser.add_argument("--connect-timeout", type=str, default=os.getenv("DB_CONNECT_TIMEOUT", "5s"), help="DB connect timeout (e.g., 3s, 1m)")
    args = parser.parse_args(argv)

    interval_s = parse_duration(args.interval)
    duration_s = parse_duration(args.duration)
    connect_timeout_s = parse_duration(args.connect_timeout)
    connect_timeout_param = max(1, int(round(connect_timeout_s)))
    out_path = args.out
    ensure_dir(out_path)

    dsn = args.dsn or build_default_dsn()

    # Open connection (reconnect on failure)
    conn: Optional[psycopg.Connection] = None
    start = time.time()
    next_tick = start
    with open(out_path, "a", encoding="utf-8") as f:
        while True:
            now = time.time()
            if now - start >= duration_s:
                break

            record: Dict[str, Any] = {"ts": iso_now(), "ok": True, "metrics": {}, "errors": []}
            try:
                if conn is None or conn.closed:
                    conn = connect_with_autocommit(dsn, connect_timeout_param)
                metrics = sample_once(conn)
                record["metrics"] = metrics
            except Exception as e:  # pragma: no cover
                record["ok"] = False
                record["errors"].append(str(e))
                # attempt reconnect next loop
                try:
                    if conn is not None and not conn.closed:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        conn.close()
                except Exception:
                    pass
                conn = None

            f.write(json.dumps(record) + "\n")
            f.flush()

            # sleep until next tick
            next_tick += interval_s
            sleep_for = max(0.0, next_tick - time.time())
            time.sleep(sleep_for)

    try:
        if conn is not None and not conn.closed:
            conn.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

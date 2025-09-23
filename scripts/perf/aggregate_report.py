#!/usr/bin/env python3
"""
Aggregate Locust and DB sampler outputs into a concise report (Markdown or HTML).

Inputs (under --run-dir):
  - locust_* CSV files (from --csv prefix)
  - db_metrics.jsonl (from scripts/perf/db_metrics.py)
  - Optional: config.json, env.json, git.json, profile.json

Outputs:
  - Markdown: summary.md
  - HTML: summary.html (with charts via Chart.js)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape
except Exception:  # pragma: no cover - optional for HTML
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    select_autoescape = None  # type: ignore


def find_locust_prefix(run_dir: Path) -> Optional[Path]:
    # Look for *_stats.csv (but not *_stats_history.csv)
    candidates = []
    for p in run_dir.glob("*_stats.csv"):
        if p.name.endswith("_stats.csv") and not p.name.endswith("_stats_history.csv"):
            candidates.append(p)
    if not candidates:
        return None
    # Choose first
    return candidates[0].with_name(candidates[0].name[:-10])  # strip _stats.csv


def parse_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(str(v)))
    except Exception:
        return default


def parse_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(str(v))
    except Exception:
        return default


def load_locust_stats(prefix: Path) -> Dict[str, Any]:
    stats_file = prefix.with_name(prefix.name + "_stats.csv")
    history_file = prefix.with_name(prefix.name + "_stats_history.csv")
    failures_file = prefix.with_name(prefix.name + "_failures.csv")

    summary: Dict[str, Any] = {
        "endpoints": [],
        "aggregated": None,
        "history": [],
        "failures": [],
    }

    def first_of(row: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in row and row[k] != "":
                return row[k]
        return None

    if stats_file.exists():
        with stats_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Name") or row.get("name") or ""
                endpoint = {
                    "type": row.get("Type") or row.get("type"),
                    "name": name,
                    # Support Locust variants: "Requests" vs "Request Count"
                    "requests": parse_int(first_of(row, ["Requests", "Request Count", "# requests", "Total Request Count"])),
                    "failures": parse_int(first_of(row, ["Failures", "Failure Count", "# failures", "Total Failure Count"])),
                    # rps column stable, but accept singular variant
                    "rps": parse_float(first_of(row, ["Requests/s", "Request/s"])),
                    "avg_ms": parse_float(first_of(row, ["Average Response Time", "Avg", "Average Response time"])),
                    "p50_ms": parse_float(row.get("50%")),
                    "p95_ms": parse_float(row.get("95%")),
                    "p99_ms": parse_float(row.get("99%")),
                    "min_ms": parse_float(first_of(row, ["Min Response Time", "Min"])),
                    "max_ms": parse_float(first_of(row, ["Max Response Time", "Max"])),
                }
                if name.lower() in ("aggregated", "total"):
                    summary["aggregated"] = endpoint
                else:
                    summary["endpoints"].append(endpoint)

    if history_file.exists():
        with history_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Keep a light history for overview (timestamp, user count, rps, fail ratio)
                summary["history"].append(
                    {
                        "time": row.get("Timestamp") or row.get("timestamp"),
                        "users": parse_int(row.get("User Count")),
                        "rps": parse_float(row.get("Requests/s")),
                        "fail_ratio": parse_float(row.get("Fail Ratio")),
                    }
                )

    if failures_file.exists():
        with failures_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                summary["failures"].append(
                    {
                        "method": row.get("Method"),
                        "name": row.get("Name"),
                        "method_name": row.get("Method Name") or "",
                        "occurrences": parse_int(row.get("Occurrences")),
                    }
                )

    # Sort endpoints by request count desc
    summary["endpoints"].sort(key=lambda x: x["requests"], reverse=True)
    return summary


def load_db_metrics(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "db_metrics.jsonl"
    out: Dict[str, Any] = {
        "samples": 0,
        "connections_by_state": {},
        "connections_by_state_max": {},
        "idle_in_tx_samples": 0,
        "idle_in_tx_max": 0,
        "wait_events": {},  # (type,event) -> count
        "top_queries": {},  # queryid -> {calls,sum_total_ms,sum_mean_ms,rows}
    }
    if not path.exists():
        return out

    def add_state(state: str, cnt: int):
        out["connections_by_state"][state] = out["connections_by_state"].get(state, 0) + cnt
        out["connections_by_state_max"][state] = max(out["connections_by_state_max"].get(state, 0), cnt)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                js = json.loads(line)
            except Exception:
                continue

            out["samples"] += 1
            # Keep per-sample for charts
            ts = js.get("ts")
            if ts:
                out.setdefault("timeline", []).append({"ts": ts})
            metrics = js.get("metrics") or {}

            states = metrics.get("connections_by_state") or {}
            for s, cnt in states.items():
                try:
                    add_state(s or "", int(cnt))
                except Exception:
                    pass
            # Also record per-sample states for charts
            if ts and isinstance(states, dict):
                sample_entry = out.get("timeline", [])[-1]
                sample_entry["states"] = states

            idle = metrics.get("idle_in_transaction")
            if isinstance(idle, int):
                if idle > 0:
                    out["idle_in_tx_samples"] += 1
                out["idle_in_tx_max"] = max(out["idle_in_tx_max"], idle)
                if ts:
                    out.get("timeline", [])[-1]["idle_in_tx"] = idle

            waits = metrics.get("wait_events") or []
            for w in waits:
                key = (w.get("type") or "", w.get("event") or "")
                out["wait_events"][key] = out["wait_events"].get(key, 0) + int(w.get("count") or 0)

            # Aggregate top queries across samples roughly by summing
            for tq in metrics.get("top_queries") or []:
                qid = tq.get("queryid")
                if qid is None:
                    continue
                agg = out["top_queries"].setdefault(qid, {"calls": 0, "sum_total_ms": 0.0, "rows": 0})
                agg["calls"] += int(tq.get("calls") or 0)
                agg["sum_total_ms"] += float(tq.get("total_time_ms") or 0.0)
                agg["rows"] += int(tq.get("rows") or 0)

    return out


def fmt_pct(v: float) -> str:
    return f"{v*100:.2f}%"


def render_markdown(
    run_dir: Path,
    locust: Dict[str, Any],
    db: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Performance Summary\n")
    # Run metadata
    lines.append("## Run Metadata")
    if meta:
        if meta.get("profile") and meta["profile"].get("name"):
            lines.append(f"- Profile: `{meta['profile']['name']}`")
        if meta.get("host"):
            lines.append(f"- Host: `{meta['host']}`")
        if meta.get("git") and meta["git"].get("commit"):
            lines.append(f"- Git: `{meta['git']['commit']}` ({meta['git'].get('branch','')})")
        lines.append("")

    # HTTP summary
    lines.append("## HTTP Summary")
    agg = locust.get("aggregated") or {}
    total_req = int(agg.get("requests") or 0)
    total_fail = int(agg.get("failures") or 0)
    fail_rate = (total_fail / total_req) if total_req else 0.0
    lines.append(
        f"- Total Requests: {total_req}  |  Failures: {total_fail} ({fmt_pct(fail_rate)})"
    )
    if agg:
        lines.append(
            f"- Latency (ms): p50={agg.get('p50_ms',0):.1f}, p95={agg.get('p95_ms',0):.1f}, p99={agg.get('p99_ms',0):.1f}, avg={agg.get('avg_ms',0):.1f}"
        )
        lines.append(f"- RPS: {agg.get('rps',0):.2f}")
    lines.append("")

    # Top endpoints table
    eps = locust.get("endpoints") or []
    if eps:
        lines.append("### Top Endpoints (by requests)")
        lines.append("| Name | Requests | Fail% | p50 | p95 | p99 | Avg | RPS |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for e in eps[:15]:
            req = int(e.get("requests") or 0)
            fail = int(e.get("failures") or 0)
            fr = (fail / req) if req else 0.0
            lines.append(
                f"| {e.get('name','')} | {req} | {fr*100:.2f}% | {e.get('p50_ms',0):.1f} | {e.get('p95_ms',0):.1f} | {e.get('p99_ms',0):.1f} | {e.get('avg_ms',0):.1f} | {e.get('rps',0):.2f} |"
            )
        lines.append("")

    # DB summary
    lines.append("## Database Summary")
    samples = int(db.get("samples") or 0)
    if samples == 0:
        lines.append("- No DB metrics available")
    else:
        # Average connections by state
        lines.append(f"- Samples: {samples}")
        lines.append("### Connections by State (avg / max)")
        lines.append("| State | Avg | Max |")
        lines.append("|---|---:|---:|")
        for state, sum_cnt in sorted(db.get("connections_by_state", {}).items()):
            avg = sum_cnt / samples
            mx = db.get("connections_by_state_max", {}).get(state, 0)
            lines.append(f"| {state or '(none)'} | {avg:.1f} | {mx} |")
        lines.append("")
        # Idle in tx
        lines.append(
            f"- Idle-in-transaction: max={db.get('idle_in_tx_max',0)}, samples_with_idle={db.get('idle_in_tx_samples',0)}"
        )
        # Wait events
        waits = db.get("wait_events", {})
        if waits:
            lines.append("### Top Wait Events")
            lines.append("| Type | Event | Count |")
            lines.append("|---|---|---:|")
            top_waits = sorted(waits.items(), key=lambda kv: kv[1], reverse=True)[:10]
            for (wt, ev), cnt in top_waits:
                lines.append(f"| {wt} | {ev} | {cnt} |")
            lines.append("")
        # Top queries
        tqs = db.get("top_queries", {})
        if tqs:
            lines.append("### Top Queries (aggregated)")
            lines.append("| QueryID | Calls | Total ms | Mean ms | Rows |")
            lines.append("|---:|---:|---:|---:|---:|")
            # Compute mean from totals
            topq = sorted(
                (
                    (qid, data)
                    for qid, data in tqs.items()
                ),
                key=lambda kv: kv[1].get("sum_total_ms", 0.0),
                reverse=True,
            )[:10]
            for qid, d in topq:
                calls = int(d.get("calls", 0))
                tot = float(d.get("sum_total_ms", 0.0))
                mean = (tot / calls) if calls else 0.0
                rows = int(d.get("rows", 0))
                lines.append(f"| {qid} | {calls} | {tot:.1f} | {mean:.2f} | {rows} |")
            lines.append("")

    # Observations (basic heuristics)
    lines.append("## Observations")
    notes: List[str] = []
    if agg:
        if fail_rate > 0.01:
            notes.append(f"Failure rate above 1% ({fmt_pct(fail_rate)}). Investigate failing endpoints.")
        if (agg.get("p95_ms") or 0) > 2000:
            notes.append("High p95 latency (>2000ms). Check DB waits and pool sizing.")
    if db.get("idle_in_tx_max", 0) > 0:
        notes.append("Idle-in-transaction observed. Look for long transactions / missing commits.")
    waits_sorted = sorted(db.get("wait_events", {}).items(), key=lambda kv: kv[1], reverse=True)
    if waits_sorted:
        top_wait = waits_sorted[0]
        if (top_wait[0][0] or "").lower() in ("lock", "lwlock"):
            notes.append("Lock waits observed. Investigate contention or transaction scope.")
    if not notes:
        notes.append("No obvious red flags detected by heuristics.")
    for n in notes:
        lines.append(f"- {n}")

    return "\n".join(lines) + "\n"


def _load_template_env() -> Optional[Environment]:
    if Environment is None:
        return None
    template_dir = Path(__file__).resolve().parent / "templates"
    loader = FileSystemLoader(str(template_dir))
    env = Environment(
        loader=loader,
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env


def build_chart_data(locust: Dict[str, Any], db: Dict[str, Any]) -> Dict[str, Any]:
    # Locust history: time, users, rps, fail_ratio
    hist = locust.get("history") or []
    locust_ts = [{
        "ts": h.get("time"),
        "users": h.get("users", 0),
        "rps": h.get("rps", 0.0),
        "fail_ratio": h.get("fail_ratio", 0.0),
    } for h in hist if h.get("time")]

    # DB timeline: ts, states (dict), idle_in_tx
    db_timeline = db.get("timeline") or []
    # Collect union of states for stacked chart
    state_keys = set()
    for s in db_timeline:
        if isinstance(s.get("states"), dict):
            state_keys.update(s["states"].keys())
    state_keys = {k or "(none)" for k in state_keys}

    return {
        "locust": locust_ts,
        "db": {
            "timeline": db_timeline,
            "states": sorted(list(state_keys)),
        },
    }


def render_html(
    run_dir: Path,
    locust: Dict[str, Any],
    db: Dict[str, Any],
    meta: Dict[str, Any],
) -> str:
    env = _load_template_env()
    if env is None:
        raise RuntimeError("Jinja2 not available; install to render HTML report")
    template = env.get_template("perf_report.html.j2")

    # Build data context
    agg = locust.get("aggregated") or {}
    # Prepare DB top waits and queries
    waits = db.get("wait_events") or {}
    top_waits = []
    if isinstance(waits, dict):
        for (wt, ev), cnt in sorted(waits.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            top_waits.append({"type": wt, "event": ev, "count": cnt})
    tq = db.get("top_queries") or {}
    top_queries = []
    if isinstance(tq, dict):
        items = sorted(tq.items(), key=lambda kv: kv[1].get("sum_total_ms", 0.0), reverse=True)[:10]
        for qid, d in items:
            calls = int(d.get("calls", 0))
            tot = float(d.get("sum_total_ms", 0.0))
            mean = (tot / calls) if calls else 0.0
            top_queries.append({
                "queryid": qid,
                "calls": calls,
                "total_ms": tot,
                "mean_ms": mean,
                "rows": int(d.get("rows", 0)),
            })
    context = {
        "meta": meta,
        "agg": agg,
        "endpoints": locust.get("endpoints") or [],
        "failures": locust.get("failures") or [],
        "db": db,
        "db_top_waits": top_waits,
        "db_top_queries": top_queries,
        "charts": build_chart_data(locust, db),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "run_dir": str(run_dir),
    }
    return template.render(**context)


def load_meta(run_dir: Path, host: Optional[str]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    if host:
        meta["host"] = host
    for name in ("config.json", "env.json", "git.json", "profile.json"):
        p = run_dir / name
        if p.exists():
            try:
                meta[name.split(".")[0]] = json.loads(p.read_text())
            except Exception:
                pass
    # Flatten profile
    if "profile" in meta and isinstance(meta["profile"], dict):
        prof = meta["profile"]
        if prof.get("data") and isinstance(prof["data"], dict):
            # keep only essentials to avoid bloating the report
            essentials = {
                "name": prof.get("name"),
                "source": prof.get("source"),
                "weights": prof["data"].get("weights"),
                "think_time_ms": prof["data"].get("think_time_ms"),
                "message_size": prof["data"].get("message_size"),
            }
            meta["profile"] = essentials
    return meta


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate perf artifacts into a summary")
    ap.add_argument("--run-dir", required=True, help="Directory containing perf artifacts")
    ap.add_argument("--out", help="Output path (default: summary.md or summary.html)")
    ap.add_argument("--format", choices=["md", "html"], default="md", help="Output format")
    ap.add_argument("--host", help="Server host shown in report (optional)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    prefix = find_locust_prefix(run_dir)
    locust = load_locust_stats(prefix) if prefix else {"endpoints": [], "aggregated": None, "history": [], "failures": []}
    db = load_db_metrics(run_dir)
    meta = load_meta(run_dir, args.host)

    if args.format == "md":
        content = render_markdown(run_dir, locust, db, meta)
        out_path = Path(args.out) if args.out else (run_dir / "summary.md")
    else:
        content = render_html(run_dir, locust, db, meta)
        out_path = Path(args.out) if args.out else (run_dir / "summary.html")

    out_path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

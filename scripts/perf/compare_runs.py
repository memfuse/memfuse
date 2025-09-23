#!/usr/bin/env python3
"""
Compare two perf run directories and flag regressions based on thresholds.

Default regression rules (configurable via CLI):
- p95 increase > 30% (aggregated and per-endpoint)
- avg latency increase > 20% (aggregated and per-endpoint)
- failure rate in B > 1%

Exit codes:
- 0: No regression
- 2: Regression detected

Usage:
  poetry run python scripts/perf/compare_runs.py \
    --run-a outputs/perf/load-20250922-170000 \
    --run-b outputs/perf/load-20250922-180000 \
    --max-p95-increase-pct 30 --max-avg-increase-pct 20 --max-fail-rate-pct 1
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_locust_prefix(run_dir: Path) -> Optional[Path]:
    candidates = []
    for p in run_dir.glob("*_stats.csv"):
        if p.name.endswith("_stats.csv") and not p.name.endswith("_stats_history.csv"):
            candidates.append(p)
    if not candidates:
        return None
    return candidates[0].with_name(candidates[0].name[:-10])


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


@dataclass
class EndpointStat:
    name: str
    requests: int
    failures: int
    rps: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    @property
    def fail_rate(self) -> float:
        return (self.failures / self.requests) if self.requests else 0.0


@dataclass
class RunStats:
    aggregated: Optional[EndpointStat]
    endpoints: Dict[str, EndpointStat]  # by name


def load_run_stats(run_dir: Path) -> RunStats:
    prefix = find_locust_prefix(run_dir)
    endpoints: Dict[str, EndpointStat] = {}
    aggregated: Optional[EndpointStat] = None
    if not prefix:
        return RunStats(aggregated=None, endpoints={})

    stats_file = prefix.with_name(prefix.name + "_stats.csv")
    if not stats_file.exists():
        return RunStats(aggregated=None, endpoints={})

    with stats_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Name") or row.get("name") or ""
            def first_of(keys: list[str]) -> Any:
                for k in keys:
                    if k in row and row[k] != "":
                        return row[k]
                return None
            s = EndpointStat(
                name=name,
                requests=parse_int(first_of(["Requests", "Request Count", "# requests", "Total Request Count"])),
                failures=parse_int(first_of(["Failures", "Failure Count", "# failures", "Total Failure Count"])),
                rps=parse_float(first_of(["Requests/s", "Request/s"])),
                avg_ms=parse_float(first_of(["Average Response Time", "Avg", "Average Response time"])),
                p50_ms=parse_float(row.get("50%")),
                p95_ms=parse_float(row.get("95%")),
                p99_ms=parse_float(row.get("99%")),
                min_ms=parse_float(first_of(["Min Response Time", "Min"])),
                max_ms=parse_float(first_of(["Max Response Time", "Max"])),
            )
            if name.lower() in ("aggregated", "total"):
                aggregated = s
            else:
                endpoints[name] = s

    return RunStats(aggregated=aggregated, endpoints=endpoints)


def pct_increase(a: float, b: float) -> float:
    if a <= 0:
        return 0.0 if b <= 0 else 1.0  # treat as 100% increase when baseline was ~0
    return (b - a) / a


def compare_runs(
    a: RunStats,
    b: RunStats,
    max_p95_inc: float,
    max_avg_inc: float,
    max_fail_rate: float,
) -> Tuple[bool, str]:
    """Return (is_regression, report_markdown)."""
    lines: List[str] = []
    regress = False

    lines.append("# Regression Report\n")
    lines.append("## Aggregated")
    if not a.aggregated or not b.aggregated:
        lines.append("- Missing aggregated stats in one or both runs.")
    else:
        A, B = a.aggregated, b.aggregated
        fail_rate_a = A.fail_rate
        fail_rate_b = B.fail_rate
        p95_inc = pct_increase(A.p95_ms, B.p95_ms)
        avg_inc = pct_increase(A.avg_ms, B.avg_ms)
        lines.append(
            f"- Requests: A={A.requests} → B={B.requests}; Fail rate: A={fail_rate_a*100:.2f}% → B={fail_rate_b*100:.2f}%"
        )
        lines.append(
            f"- Latency p95(ms): A={A.p95_ms:.1f} → B={B.p95_ms:.1f} ({p95_inc*100:.1f}%); Avg(ms): A={A.avg_ms:.1f} → B={B.avg_ms:.1f} ({avg_inc*100:.1f}%)"
        )
        if fail_rate_b > max_fail_rate:
            regress = True
            lines.append(f"  - REGRESSION: fail rate {fail_rate_b*100:.2f}% > {max_fail_rate*100:.2f}%")
        if p95_inc > max_p95_inc:
            regress = True
            lines.append(f"  - REGRESSION: p95 increase {p95_inc*100:.1f}% > {max_p95_inc*100:.1f}%")
        if avg_inc > max_avg_inc:
            regress = True
            lines.append(f"  - REGRESSION: avg increase {avg_inc*100:.1f}% > {max_avg_inc*100:.1f}%")

    lines.append("\n## Endpoint-level (top regressions)")
    common = set(a.endpoints.keys()) & set(b.endpoints.keys())
    endpoint_issues: List[Tuple[str, float, float]] = []  # (name, p95_inc, avg_inc)
    for name in sorted(common):
        ea, eb = a.endpoints[name], b.endpoints[name]
        if ea.requests < 10 or eb.requests < 10:
            continue  # ignore very low volume endpoints
        p95_inc = pct_increase(ea.p95_ms, eb.p95_ms)
        avg_inc = pct_increase(ea.avg_ms, eb.avg_ms)
        fr_b = eb.fail_rate
        issues = []
        if fr_b > max_fail_rate:
            issues.append(f"fail {fr_b*100:.2f}% > {max_fail_rate*100:.2f}%")
            regress = True
        if p95_inc > max_p95_inc:
            issues.append(f"p95 +{p95_inc*100:.1f}% > {max_p95_inc*100:.1f}%")
            regress = True
        if avg_inc > max_avg_inc:
            issues.append(f"avg +{avg_inc*100:.1f}% > {max_avg_inc*100:.1f}%")
            regress = True
        if issues:
            endpoint_issues.append((name, p95_inc, avg_inc))
            lines.append(
                f"- {name}: A p95={ea.p95_ms:.1f}→B {eb.p95_ms:.1f}, A avg={ea.avg_ms:.1f}→B {eb.avg_ms:.1f}  | Issues: {', '.join(issues)}"
            )

    if not endpoint_issues:
        lines.append("- No endpoint-level regressions exceeding thresholds.")

    return regress, "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compare two perf runs and detect regressions")
    ap.add_argument("--run-a", required=True, help="Baseline run directory")
    ap.add_argument("--run-b", required=True, help="New run directory")
    ap.add_argument("--max-p95-increase-pct", type=float, default=30.0)
    ap.add_argument("--max-avg-increase-pct", type=float, default=20.0)
    ap.add_argument("--max-fail-rate-pct", type=float, default=1.0)
    ap.add_argument("--out", help="Optional path to write comparison report")
    args = ap.parse_args(argv)

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)
    a = load_run_stats(run_a)
    b = load_run_stats(run_b)

    regress, report = compare_runs(
        a,
        b,
        max_p95_inc=args.max_p95_increase_pct / 100.0,
        max_avg_inc=args.max_avg_increase_pct / 100.0,
        max_fail_rate=args.max_fail_rate_pct / 100.0,
    )

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    else:
        print(report)

    return 2 if regress else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

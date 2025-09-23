#!/usr/bin/env python3
"""
Capture run metadata (config/env/git) into a perf run directory.

Outputs written under --run-dir:
  - config.json  (effective server config if available; fallback to composed config)
  - env.json     (selected environment variables, with secrets masked)
  - git.json     (commit, branch, tag, dirty, remotes)
  - profile.json (selected Locust profile details if discoverable)

Usage:
  poetry run python scripts/perf/capture_run_metadata.py --run-dir outputs/perf/<run>
  poetry run python scripts/perf/capture_run_metadata.py --run-dir <dir> --host http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _mask_value(key: str, value: str) -> str:
    k = key.lower()
    if any(s in k for s in ("password", "secret", "token", "key")):
        return "***"
    return value


def capture_env() -> Dict[str, Any]:
    keys = []
    # Focused snapshot to avoid dumping all env
    for k in os.environ.keys():
        kl = k.upper()
        if (
            kl.startswith("POSTGRES_")
            or kl.startswith("PG_")
            or kl in {"API_PREFIX", "ENTITY_PREFIX", "PROFILE", "PROFILE_PATH", "MSG_SIZE_PROFILE"}
        ):
            keys.append(k)
    snap: Dict[str, Any] = {}
    for k in sorted(set(keys)):
        snap[k] = _mask_value(k, os.environ.get(k, ""))
    return snap


def capture_git() -> Dict[str, Any]:
    def run(cmd: list[str]) -> Optional[str]:
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            return out
        except Exception:
            return None

    data = {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "tag": run(["git", "describe", "--tags", "--always"]),
        "dirty": bool(run(["git", "status", "--porcelain"])) if run(["git", "status", "--porcelain"]) is not None else None,
        "remote": run(["git", "remote", "-v"]),
    }
    return data


def fetch_server_config(host: str) -> Optional[Dict[str, Any]]:
    import urllib.request
    import urllib.error

    url = host.rstrip("/") + "/api/v1/health"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            js = json.loads(body.decode("utf-8"))
            # Expected ApiResponse shape {status, code, data: {config: {...}, ...}}
            data = js.get("data") or {}
            cfg = data.get("config")
            if cfg:
                # Include version fields if present
                v = {
                    k: data.get(k)
                    for k in ("version", "python_version")
                    if k in data
                }
                return {"config": cfg, **({"version": v} if v else {})}
    except Exception:
        return None
    return None


def compose_config_fallback() -> Optional[Dict[str, Any]]:
    try:
        # Compose Hydra config from repo config folder
        from hydra import initialize, compose
        from omegaconf import OmegaConf

        # Resolve config path relative to this file
        repo_root = Path(__file__).resolve().parents[2]
        config_path = str(repo_root / "config")
        with initialize(config_path=config_path, job_name="capture_config"):
            cfg = compose(config_name="config")
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        return None


def capture_profile() -> Optional[Dict[str, Any]]:
    profile = os.getenv("PROFILE")
    profile_path = os.getenv("PROFILE_PATH")
    if profile_path and Path(profile_path).exists():
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                return {"name": profile or Path(profile_path).stem, "source": profile_path, "data": json.load(f)}
        except Exception:
            return {"name": profile or Path(profile_path).stem, "source": profile_path, "data": None}
    elif profile:
        # Try default bundled profiles
        default = Path(__file__).resolve().parents[1] / "tests" / "performance" / "api" / "profiles" / f"{profile}.json"
        if default.exists():
            try:
                with open(default, "r", encoding="utf-8") as f:
                    return {"name": profile, "source": str(default), "data": json.load(f)}
            except Exception:
                return {"name": profile, "source": str(default), "data": None}
        return {"name": profile}
    return None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Capture perf run metadata")
    ap.add_argument("--run-dir", required=True, help="Directory to write metadata files")
    ap.add_argument("--host", default=os.getenv("HOST", "http://localhost:8000"), help="Base server host for health/config")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1) Config: prefer server health endpoint, fallback to Hydra composition
    cfg: Optional[Dict[str, Any]] = fetch_server_config(args.host)
    if cfg is None:
        fallback = compose_config_fallback()
        if fallback is not None:
            cfg = {"config": fallback}
        else:
            cfg = {"error": "Unable to capture config from server or compose fallback"}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    # 2) Env snapshot (masked)
    env = capture_env()
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))

    # 3) Git metadata
    git = capture_git()
    (run_dir / "git.json").write_text(json.dumps(git, indent=2))

    # 4) Profile (if any)
    prof = capture_profile()
    if prof is not None:
        (run_dir / "profile.json").write_text(json.dumps(prof, indent=2))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


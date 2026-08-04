#!/usr/bin/env python3
"""Run a bounded concurrent HTTP soak and enforce success-rate/latency SLOs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class RequestResult:
    success: bool
    latency_ms: float
    status_code: int | None = None
    error: str | None = None


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def build_metrics_report(
    results: list[RequestResult],
    *,
    requested_count: int | None = None,
    min_success_rate: float,
    max_p95_ms: float,
) -> dict:
    requested = len(results) if requested_count is None else requested_count
    successful = sum(result.success for result in results)
    success_rate = successful / requested if requested else 0.0
    latencies = [result.latency_ms for result in results]
    p95 = percentile(latencies, 0.95)
    failures = []
    if len(results) != requested:
        failures.append(f"completed {len(results)} of {requested} requests")
    if success_rate < min_success_rate:
        failures.append(f"success rate {success_rate:.4f} is below {min_success_rate:.4f}")
    if p95 is None or p95 > max_p95_ms:
        actual = "unavailable" if p95 is None else f"{p95:.3f}ms"
        failures.append(f"p95 latency {actual} exceeds {max_p95_ms:.3f}ms")
    latency = {
        "min": round(min(latencies), 3) if latencies else None,
        "mean": round(statistics.fmean(latencies), 3) if latencies else None,
        "p50": round(percentile(latencies, 0.50), 3) if latencies else None,
        "p95": round(p95, 3) if p95 is not None else None,
        "p99": round(percentile(latencies, 0.99), 3) if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }
    return {
        "requested_requests": requested,
        "completed_requests": len(results),
        "successful_requests": successful,
        "failed_requests": requested - successful,
        "success_rate": success_rate,
        "latency_ms": latency,
        "status_codes": dict(sorted(Counter(str(r.status_code) for r in results if r.status_code is not None).items())),
        "error_counts": dict(sorted(Counter(r.error for r in results if r.error).items())),
        "thresholds": {"min_success_rate": min_success_rate, "max_p95_ms": max_p95_ms},
        "passed": not failures,
        "failures": failures,
    }


def _request(url: str, timeout: float, expected: set[int]) -> RequestResult:
    started = time.perf_counter()
    status = None
    error = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CogDoc-reliability-soak/1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            response.read(65536)
        success = status in expected
        if not success:
            error = f"unexpected HTTP status {status}"
    except urllib.error.HTTPError as exc:
        status = exc.code
        success = status in expected
        error = None if success else f"HTTPError: {exc.code}"
    except Exception as exc:  # Each failed attempt must become a metric, not abort the run.
        success = False
        error = f"{type(exc).__name__}: {exc}"
    return RequestResult(success, (time.perf_counter() - started) * 1000, status, error)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--requests", dest="request_count", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--startup-timeout", type=float, default=0.0)
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument("--max-p95-ms", type=float, default=750.0)
    parser.add_argument("--expected-status", type=int, action="append")
    parser.add_argument("--json", type=Path, default=Path("artifacts/reliability/soak.json"))
    args = parser.parse_args()
    if args.request_count <= 0 or args.concurrency <= 0 or args.timeout <= 0:
        parser.error("requests, concurrency, and timeout must be positive")
    if not 0 <= args.min_success_rate <= 1 or args.max_p95_ms <= 0 or args.startup_timeout < 0:
        parser.error("thresholds are outside their valid ranges")

    expected = set(args.expected_status or [200])
    started_wall = datetime.now(timezone.utc)
    started = time.monotonic()
    warmup = {"enabled": args.startup_timeout > 0, "ready": True, "attempts": 0}
    if args.startup_timeout > 0:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            warmup["attempts"] += 1
            probe = _request(args.url, args.timeout, expected)
            if probe.success:
                warmup["ready"] = True
                break
            warmup["ready"] = False
            warmup["last_error"] = probe.error
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))

    results = []
    if warmup["ready"]:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(lambda _: _request(args.url, args.timeout, expected), range(args.request_count)))
    metrics = build_metrics_report(
        results,
        requested_count=args.request_count,
        min_success_rate=args.min_success_rate,
        max_p95_ms=args.max_p95_ms,
    )
    if not warmup["ready"]:
        metrics["failures"].insert(0, "endpoint did not become ready before startup timeout")
        metrics["passed"] = False
    report = {
        "schema_version": 1,
        "started_at": started_wall.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(time.monotonic() - started, 6),
        "url": args.url,
        "concurrency": args.concurrency,
        "request_timeout_seconds": args.timeout,
        "expected_statuses": sorted(expected),
        "warmup": warmup,
        **metrics,
    }
    _write_json(args.json, report)
    outcome = "passed" if report["passed"] else "failed"
    print(
        f"API soak {outcome}: success_rate={report['success_rate']:.4f} "
        f"p95_ms={report['latency_ms']['p95']} report={args.json}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

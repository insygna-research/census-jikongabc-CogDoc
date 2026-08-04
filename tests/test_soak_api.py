from scripts.soak_api import RequestResult, build_metrics_report


def _result(success=True, latency=10.0, status=200):
    return RequestResult(success, latency, status, None if success else "request failed")


def test_soak_metrics_pass_when_both_thresholds_hold():
    report = build_metrics_report(
        [_result() for _ in range(10)],
        min_success_rate=0.99,
        max_p95_ms=20,
    )
    assert report["passed"] is True
    assert report["success_rate"] == 1.0
    assert report["latency_ms"]["p95"] == 10.0


def test_soak_metrics_fail_success_rate_threshold():
    report = build_metrics_report(
        [_result() for _ in range(8)] + [_result(False) for _ in range(2)],
        min_success_rate=0.9,
        max_p95_ms=20,
    )
    assert report["passed"] is False
    assert any("success rate" in failure for failure in report["failures"])


def test_soak_metrics_fail_p95_threshold():
    report = build_metrics_report(
        [_result(latency=10) for _ in range(18)] + [_result(latency=500) for _ in range(2)],
        min_success_rate=1.0,
        max_p95_ms=100,
    )
    assert report["passed"] is False
    assert any("p95 latency" in failure for failure in report["failures"])

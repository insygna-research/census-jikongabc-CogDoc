import json
import sys

import pytest
import scripts.eval_suite as eval_suite


# 写入质量评测集。
def _write_quality_eval(path):
    rows = [
        {
            "case_type": "router",
            "layer": "easy",
            "query": "请总结 a.pdf",
            "expected_task_type": "summary",
        },
        {
            "case_type": "citation",
            "layer": "hard",
            "answer": "内容来自文档。[a.pdf:P1]",
            "expected_valid": True,
            "docs": [
                {
                    "text": "内容来自文档。",
                    "meta": {"source": "a.pdf", "page": 1, "chunk_id": "c1"},
                }
            ],
        },
        {"case_type": "faithfulness", "layer": "no-answer", "is_faithful": True},
        {
            "case_type": "router",
            "layer": "compare",
            "query": "对比 a.pdf 和 b.pdf",
            "expected_task_type": "compare",
        },
        {
            "case_type": "router",
            "layer": "multi-turn",
            "query": "那它有什么限制",
            "expected_task_type": "qa",
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


# 写入检索评测集。
def _write_retrieval_eval(path):
    rows = [
        {"query": "单文档", "expected_sources": ["a.pdf"], "layer": "single-source"},
        {
            "query": "跨文档",
            "expected_sources": ["a.pdf", "b.pdf"],
            "layer": "multi-source",
        },
        {"query": "无答案", "expected_sources": [], "layer": "no-answer"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )


# 验证组合评测不跑检索也能生成门禁结果。
def test_eval_suite_builds_gate_without_retrieval(tmp_path):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)

    report = eval_suite.build_report(
        quality_eval_set=quality_path,
        retrieval_eval_set=retrieval_path,
        run_retrieval=False,
        k_values=[1, 3],
        rerank=False,
    )

    assert report["gate"]["passed"] is True
    assert report["quality_report"]["coverage"]["is_coverage_complete"] is True
    assert report["retrieval_report"]["skipped"] is True
    assert report["retrieval_report"]["coverage"]["is_coverage_complete"] is True


# 验证组合评测会在覆盖不完整时失败。
def test_eval_suite_fails_incomplete_coverage(tmp_path):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    _write_quality_eval(quality_path)
    retrieval_path.write_text(
        json.dumps(
            {
                "query": "单文档",
                "expected_sources": ["a.pdf"],
                "layer": "single-source",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = eval_suite.build_report(
        quality_eval_set=quality_path,
        retrieval_eval_set=retrieval_path,
        run_retrieval=False,
        k_values=[1, 3],
        rerank=False,
    )

    assert report["gate"]["passed"] is False
    assert report["gate"]["checks"]["quality_coverage"] is True
    assert report["gate"]["checks"]["retrieval_coverage"] is False


# 验证组合评测入口会写出报告。
def test_eval_suite_main_writes_json_report(tmp_path, monkeypatch, capsys):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    report_path = tmp_path / "report.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--quality-eval-set",
            str(quality_path),
            "--retrieval-eval-set",
            str(retrieval_path),
            "--json",
            str(report_path),
        ],
    )

    assert eval_suite.main() == 0
    out = capsys.readouterr().out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert "整体结果: 通过" in out
    assert payload["gate"]["passed"] is True
    assert payload["retrieval_report"]["skipped"] is True


# 验证组合基线能识别质量回退。
def test_eval_suite_baseline_detects_quality_regression(tmp_path):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    report = eval_suite.build_report(
        quality_eval_set=quality_path,
        retrieval_eval_set=retrieval_path,
        run_retrieval=False,
        k_values=[1, 3],
        rerank=False,
    )
    baseline = {
        "quality_report": {
            "aggregate": {
                "router_rule_accuracy": 1.0,
                "citation_accuracy": 1.0,
            }
        },
        "retrieval_report": {"aggregate": {"mrr": 1.0}},
    }
    report["quality_report"]["aggregate"]["router_rule_accuracy"] = 0.0
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False), encoding="utf-8"
    )

    result = eval_suite.compare_baseline(report, baseline_path)

    assert result["regressed"] is True
    assert result["quality"]["rows"][0]["status"] == "regressed"
    assert result["retrieval"]["skipped"] is True


# 验证组合基线忽略旧报告缺失的新门禁指标。
def test_eval_suite_baseline_ignores_new_metric_missing_from_old_baseline(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    report = {
        "quality_report": {
            "aggregate": {"old_metric": 1.0, "new_metric": 0.0},
            "baseline_gated_metrics": ["old_metric", "new_metric"],
        },
        "retrieval_report": {"skipped": True},
    }
    baseline = {"quality_report": {"aggregate": {"old_metric": 1.0}}}
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False), encoding="utf-8"
    )

    result = eval_suite.compare_baseline(report, baseline_path)

    assert result["regressed"] is False
    assert [row["metric"] for row in result["quality"]["rows"]] == ["old_metric"]


# 验证组合基线按双方门禁指标交集对比。
def test_eval_suite_baseline_uses_gated_metric_intersection(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    report = {
        "quality_report": {
            "aggregate": {"old_metric": 1.0, "new_metric": 0.0},
            "baseline_gated_metrics": ["old_metric", "new_metric"],
        },
        "retrieval_report": {"skipped": True},
    }
    baseline = {
        "quality_report": {
            "aggregate": {"old_metric": 1.0, "new_metric": 1.0},
            "baseline_gated_metrics": ["old_metric"],
        }
    }
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False), encoding="utf-8"
    )

    result = eval_suite.compare_baseline(report, baseline_path)

    assert result["regressed"] is False
    assert [row["metric"] for row in result["quality"]["rows"]] == ["old_metric"]


# 验证组合入口基线回退返回非零。
def test_eval_suite_main_returns_nonzero_on_baseline_regression(
    tmp_path, monkeypatch
):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    baseline = {
        "quality_report": {
            "aggregate": {
                "router_rule_accuracy": 1.0,
                "citation_accuracy": 2.0,
            }
        }
    }
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--quality-eval-set",
            str(quality_path),
            "--retrieval-eval-set",
            str(retrieval_path),
            "--baseline",
            str(baseline_path),
        ],
    )

    assert eval_suite.main() == 1


# 验证组合入口缺失基线返回稳定错误码。
def test_eval_suite_main_returns_two_when_baseline_missing(tmp_path, monkeypatch):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    baseline_path = tmp_path / "missing.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--quality-eval-set",
            str(quality_path),
            "--retrieval-eval-set",
            str(retrieval_path),
            "--baseline",
            str(baseline_path),
        ],
    )

    assert eval_suite.main() == 2


# 验证组合报告写入会创建文件。
def test_eval_suite_write_report_creates_file(tmp_path):
    report_path = tmp_path / "nested" / "report.json"
    payload = {"gate": {"passed": True}}

    eval_suite.write_report(payload, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == payload
    assert not report_path.with_suffix(".json.tmp").exists()


# 验证组合入口可更新基线。
def test_eval_suite_main_updates_baseline(tmp_path, monkeypatch, capsys):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    baseline_path = tmp_path / "baseline.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--quality-eval-set",
            str(quality_path),
            "--retrieval-eval-set",
            str(retrieval_path),
            "--update-baseline",
            str(baseline_path),
        ],
    )

    assert eval_suite.main() == 0
    out = capsys.readouterr().out
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "组合基线已更新" in out
    assert payload["gate"]["passed"] is True


# 验证组合入口使用默认路径更新基线。
def test_eval_suite_main_updates_default_baseline(tmp_path, monkeypatch):
    quality_path = tmp_path / "quality.jsonl"
    retrieval_path = tmp_path / "retrieval.jsonl"
    baseline_path = tmp_path / "default_baseline.json"
    _write_quality_eval(quality_path)
    _write_retrieval_eval(retrieval_path)
    monkeypatch.setattr(eval_suite, "DEFAULT_BASELINE_PATH", baseline_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--quality-eval-set",
            str(quality_path),
            "--retrieval-eval-set",
            str(retrieval_path),
            "--update-baseline",
        ],
    )

    assert eval_suite.main() == 0
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert payload["gate"]["passed"] is True


# 验证组合入口拒绝同时对比并更新基线。
def test_eval_suite_main_rejects_baseline_and_update(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_suite.py",
            "--baseline",
            str(baseline_path),
            "--update-baseline",
            str(baseline_path),
        ],
    )

    with pytest.raises(SystemExit) as exc:
        eval_suite.main()
    assert exc.value.code == 2

import json
import sys

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

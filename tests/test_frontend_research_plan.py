from cogdoc.frontend.app import (
    _build_edited_research_requirements,
    _research_contract_key,
    _research_requirement_editor_lines,
)


def _legacy_requirement():
    return {
        "question": "报名对象有哪些限制？",
        "retrieval_query": "报名对象有哪些限制？",
        "recovery_query": "报名对象有哪些限制？",
    }


def test_research_requirement_editor_upgrades_equal_legacy_queries():
    question_text, retrieval_text, recovery_text = _research_requirement_editor_lines(
        [_legacy_requirement()]
    )

    assert question_text == "报名对象有哪些限制？"
    assert retrieval_text == question_text
    assert _research_contract_key(recovery_text) != _research_contract_key(
        retrieval_text
    )


def test_changed_requirement_rebuilds_two_distinct_queries():
    original = [_legacy_requirement()]
    _, old_retrieval, old_recovery = _research_requirement_editor_lines(original)

    requirements = _build_edited_research_requirements(
        "参赛年龄上限是多少？",
        old_retrieval,
        old_recovery,
        original,
    )

    assert requirements[0]["retrieval_query"] == "参赛年龄上限是多少？"
    assert _research_contract_key(
        requirements[0]["recovery_query"]
    ) != _research_contract_key(requirements[0]["retrieval_query"])
    assert "参赛年龄上限是多少" in requirements[0]["recovery_query"]


def test_manual_query_edits_are_kept_but_equivalent_recovery_is_diversified():
    original = [_legacy_requirement()]
    requirements = _build_edited_research_requirements(
        "报名对象有哪些限制？",
        "参赛人员 资格限制",
        "ＣＯＭＰＥＴＩＴＯＲ 条件",
        original,
    )
    assert requirements[0]["retrieval_query"] == "参赛人员 资格限制"
    assert requirements[0]["recovery_query"] == "ＣＯＭＰＥＴＩＴＯＲ 条件"

    diversified = _build_edited_research_requirements(
        "报名对象有哪些限制？",
        "ＡＢＣ 资格",
        "abc 资格",
        original,
    )
    assert _research_contract_key(
        diversified[0]["recovery_query"]
    ) != _research_contract_key(diversified[0]["retrieval_query"])

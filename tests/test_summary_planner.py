from cogdoc.agents.summary_planner import SectionPlannerAgent


# 验证 default summary sections are stable 场景。
def test_default_summary_sections_are_stable():
    # Summary MVP 默认输出固定结构化章节。
    result = SectionPlannerAgent.plan_sections({"query": "总结这篇文档"})

    titles = [section["title"] for section in result["summary_section_plans"]]
    assert titles == [
        "背景与目标",
        "方案与流程",
        "规则与要求",
        "价值与产出",
        "限制与注意事项",
    ]
    assert result["steps_trace"][0]["step_name"] == "summary_section_planner"


# 验证 custom summary sections override defaults 场景。
def test_custom_summary_sections_override_defaults():
    # 调用方可显式传入章节标题。
    result = SectionPlannerAgent.plan_sections(
        {"summary_section_titles": ["背景", "贡献", "风险"]}
    )

    assert [section["title"] for section in result["summary_section_plans"]] == [
        "背景",
        "贡献",
        "风险",
    ]
    assert [section["section_id"] for section in result["summary_section_plans"]] == [
        "custom_1",
        "custom_2",
        "custom_3",
    ]


# 验证 empty custom sections fall back to defaults 场景。
def test_empty_custom_sections_fall_back_to_defaults():
    # 空自定义章节不应产生空规划。
    result = SectionPlannerAgent.plan_sections({"summary_section_titles": ["", "  "]})

    assert len(result["summary_section_plans"]) == 5

from typing import List
from graph.state import SummarySectionPlan


DEFAULT_SUMMARY_SECTIONS: List[SummarySectionPlan] = [
    {
        "section_id": "research_question",
        "title": "研究问题",
        "instruction": "概括文档要解决的核心问题、任务背景与目标，即“为什么做、做什么”；方法与结果留给后续章节。",
    },
    {
        "section_id": "method",
        "title": "方法与方案",
        "instruction": "概括文档采用的方法、系统设计、关键流程或技术路线，即“怎么做”；不写实验数据与最终结论。",
    },
    {
        "section_id": "experiment_or_requirement",
        "title": "实验与要求",
        "instruction": "概括实验设置、评价指标、提交要求或约束条件，侧重可操作的条件与衡量方式；不重复方法本身。",
    },
    {
        "section_id": "conclusion",
        "title": "结论与价值",
        "instruction": "概括文档已得出的结论、预期成果与应用价值；只写结论性判断，不复述方法过程与实验细节。",
    },
    {
        "section_id": "limitations",
        "title": "局限与注意事项",
        "instruction": "概括文档明确提到的限制、风险、注意事项或未覆盖范围；仅限文档显式说明的内容，不要自行推断。",
    },
]


class SectionPlannerAgent:
    @staticmethod
    def plan_sections(state: dict) -> dict:
        # Summary MVP 先使用稳定固定章节，避免规划阶段引入幻觉。
        query = state.get("query", "")
        requested_sections = state.get("summary_section_titles", [])

        if requested_sections:
            plans: List[SummarySectionPlan] = [
                {
                    "section_id": f"custom_{idx + 1}",
                    "title": str(title),
                    "instruction": f"围绕用户摘要意图提炼“{title}”相关内容。",
                }
                for idx, title in enumerate(requested_sections)
                if str(title).strip()
            ]
            if plans:
                return {"summary_section_plans": plans}

        plans = [
            SummarySectionPlan(
                section_id = section["section_id"],
                title = section["title"],
                instruction = section["instruction"],
            )
            for section in DEFAULT_SUMMARY_SECTIONS
        ]
        return {
            "summary_section_plans": plans,
            "steps_trace": [
                {
                    "step_name": "summary_section_planner",
                    "input_summary": query,
                    "output_summary": "，".join(plan["title"] for plan in plans),
                }
            ],
        }

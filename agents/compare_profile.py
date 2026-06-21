from typing import List
from langchain_core.messages import HumanMessage, SystemMessage
from agents.qa_generator import Generator
from agents.summary_generator import (
    generate_section_cell,
    tokenize_for_section,
)
from graph.state import CompareDimensionPlan, CompareCell, DocumentProfile, RetrievedDoc, SummarySectionPlan


COMPARE_CELL_CONTEXT_CHUNKS = 4
# 本地模式缩小维度和上下文，避免 Ollama 多轮生成时触发内存不足。
LOCAL_COMPARE_CELL_CONTEXT_CHUNKS = 2
LOCAL_COMPARE_DIMENSION_IDS = {"method", "data", "metrics", "limitations"}


DEFAULT_COMPARE_DIMENSIONS: List[CompareDimensionPlan] = [
    {
        "dimension_id": "method",
        "title": "方法",
        "instruction": "概括该文档描述的核心方法、技术路线、系统设计或实施流程。",
    },
    {
        "dimension_id": "data",
        "title": "数据",
        "instruction": "概括该文档使用或要求的数据、输入、资料来源、样本、场景或对象。",
    },
    {
        "dimension_id": "metrics",
        "title": "指标",
        "instruction": "概括该文档中的评价指标、验收标准、评分规则、实验设置或衡量方式。",
    },
    {
        "dimension_id": "strengths",
        "title": "优点",
        "instruction": "概括该文档明确提到的优势、价值、收益、创新点或适用收益。",
    },
    {
        "dimension_id": "limitations",
        "title": "限制",
        "instruction": "概括该文档明确提到的限制、风险、前提条件、注意事项或未覆盖范围。",
    },
    {
        "dimension_id": "scenarios",
        "title": "适用场景",
        "instruction": "概括该文档明确适用的任务、用户、环境、业务场景或使用边界。",
    },
]


def default_compare_dimensions(is_local: bool = False) -> List[CompareDimensionPlan]:
    # 本地模式只保留核心维度，控制 LLM 调用次数。
    dimensions = [
        CompareDimensionPlan(
            dimension_id = dimension["dimension_id"],
            title = dimension["title"],
            instruction = dimension["instruction"],
        )
        for dimension in DEFAULT_COMPARE_DIMENSIONS
    ]   
    if is_local:
        return [dimension for dimension in dimensions if dimension["dimension_id"] in LOCAL_COMPARE_DIMENSION_IDS]
    return dimensions


def _dimension_as_section_plan(dimension: CompareDimensionPlan) -> SummarySectionPlan:
    # Compare 维度复用 Summary section 生成器的字段契约。
    return SummarySectionPlan(
        section_id = dimension["dimension_id"],
        title = dimension["title"],
        instruction = dimension["instruction"],
    )


def _compare_messages(source: str, plan: SummarySectionPlan, query: str, context: str):
    # 首轮提示禁止模型自造引用，引用由共享 helper 统一绑定。
    return [
        SystemMessage(
            content = (
                "你是一位严谨的技术方案对比助手。你的任务是：仅依据给定的 <Document> 标签文本，"
                "为当前文档的指定对比维度写一段中文短描述。\n\n"
                "【硬性约束】\n"
                "1. 只能描述当前文档在当前维度下的信息，禁止跨文档综合或下结论。\n"
                "2. 只能使用 <Document> 标签内的信息，禁止引入标签外知识、常识或推测。\n"
                "3. 不要输出引用标签、页码、文件名或 <Document> 标签，程序会在生成后自动绑定引用。\n"
                "4. 不要输出维度标题、列表编号、规则解释或额外说明。\n\n"
                "【无依据时】\n"
                "- 只有当所有给定片段都没有任何可归入该维度的信息时，才输出：文档中未明确说明。\n"
                "- 若片段中有目标、流程、规则、指标、对象、收益、限制或适用条件，必须据实归纳。\n\n"
                "【输出】只输出 1-2 句中文短描述，尽量不超过 120 个中文字符；"
                "或输出“文档中未明确说明”。"
            )
        ),
        HumanMessage(
            content = (
                f"【目标文档】{source}\n"
                f"【对比维度】{plan['title']}\n"
                f"【维度聚焦】{plan['instruction']}\n"
                f"【用户对比意图】{query}\n\n"
                f"【参考资料开始】\n{context}\n【参考资料结束】\n\n"
                "请只描述该文档在该维度下的信息。"
            )
        ),
    ]


def _compare_retry_messages(source: str, plan: SummarySectionPlan, query: str, context: str):
    # 本地模型误判无依据时，用更直接的事实提炼提示重试一次。
    return [
        SystemMessage(
            content = (
                "你是一位严谨的中文资料整理助手。下面片段已经由程序筛选为与指定对比维度相关。"
                "请从片段中提炼事实，不要因为文档类型不是论文而回答“文档中未明确说明”。\n\n"
                "【要求】只能依据 <Document> 标签内文字；能找到相关事实就写 1-2 句；"
                "不要输出标题、引用、页码或解释。"
            )
        ),
        HumanMessage(
            content = (
                f"【目标文档】{source}\n"
                f"【对比维度】{plan['title']}\n"
                f"【维度说明】{plan['instruction']}\n"
                f"【用户意图】{query}\n\n"
                f"【参考资料开始】\n{context}\n【参考资料结束】\n\n"
                "请重新提炼该维度描述。"
            )
        ),
    ]


class DocumentProfileAgent:
    @staticmethod
    def build_profiles(state: dict) -> dict:
        # 每篇文档按相同维度生成 profile，供后续对齐输出。
        docs_by_source: dict[str, List[RetrievedDoc]] = state.get("compare_docs_by_source", {})
        sources: List[str] = state.get("compare_sources", list(docs_by_source.keys()))
        dimensions: List[CompareDimensionPlan] = state.get("compare_dimensions") or default_compare_dimensions(is_local = state.get("is_local", False))
        query = state.get("query", "")
        is_local = state.get("is_local", False)

        if len(docs_by_source) < 2 or not dimensions:
            return {"document_profiles": [], "compare_dimensions": dimensions}

        llm = Generator._get_client(is_local = is_local)
        profiles: List[DocumentProfile] = []

        for source in sources:
            docs = docs_by_source.get(source, [])
            if not docs:
                continue

            doc_tokens = [(doc, tokenize_for_section(doc["text"])) for doc in docs]
            cells: List[CompareCell] = []

            for dimension in dimensions:
                plan = _dimension_as_section_plan(dimension)
                # 每个 cell 走 Summary 共享的证据绑定和无依据规则。
                content, evidence = generate_section_cell(
                    llm,
                    source,
                    plan,
                    docs,
                    query,
                    is_local,
                    doc_tokens,
                    _compare_messages,
                    _compare_retry_messages,
                    max_chunks = LOCAL_COMPARE_CELL_CONTEXT_CHUNKS if is_local else COMPARE_CELL_CONTEXT_CHUNKS,
                )

                cells.append(
                    CompareCell(
                        dimension_id = dimension["dimension_id"],
                        source = source,
                        content = content,
                        evidence = evidence,
                    )
                )

            profiles.append(DocumentProfile(source = source, cells = cells))

        return {
            "compare_dimensions": dimensions,
            "document_profiles": profiles,
            "steps_trace": [
                {
                    "step_name": "compare_document_profile",
                    "input_summary": query,
                    "output_summary": f"{len(profiles)} documents x {len(dimensions)} dimensions",
                }
            ],
        }
